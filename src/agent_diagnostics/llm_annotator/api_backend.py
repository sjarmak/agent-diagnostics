"""Anthropic Python SDK backend (synchronous single-trial and async batch)."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from agent_diagnostics.annotation_cache import DEFAULT_CACHE_DIR, cache_key
from agent_diagnostics.llm_annotator.models import _resolve_model_api
from agent_diagnostics.llm_annotator.parsing import (
    _is_error,
    _load_json,
    _read_text,
    _to_annotation_result,
    _unwrap_result,
)
from agent_diagnostics.llm_annotator.prompt import (
    _ANNOTATION_SCHEMA,
    _taxonomy_yaml,
    build_prompt,
    truncate_trajectory,
    validate_categories,
)
from agent_diagnostics.types import AnnotationError, AnnotationResult

logger = logging.getLogger(__name__)


def annotate_trial_api(
    trial_dir: str | Path,
    signals: dict,
    model: str = "haiku",
) -> list[dict]:
    """Annotate a single trial using the Anthropic API directly.

    Requires ``ANTHROPIC_API_KEY`` in the environment.
    """
    # Import cache helpers via the package so test mocks on
    # ``agent_diagnostics.llm_annotator.get_cached`` / ``put_cached`` apply.
    from agent_diagnostics import llm_annotator as _pkg

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic SDK required -- install with: pip install agent-observatory[llm]"
        ) from None

    trial_dir = Path(trial_dir)
    instruction = _read_text(trial_dir / "agent" / "instruction.txt", max_chars=4000)
    traj = _load_json(trial_dir / "agent" / "trajectory.json")
    truncated = truncate_trajectory(traj, first_n=30, last_n=10)
    taxonomy = _taxonomy_yaml()

    prompt = build_prompt(instruction, truncated, signals, taxonomy)
    model_id = _resolve_model_api(model)

    # Cache lookup
    key = cache_key(prompt, model_id)
    cached = _pkg.get_cached(DEFAULT_CACHE_DIR, key)
    if cached is not None:
        logger.debug("Cache hit for %s (key=%s)", trial_dir, key[:12])
        return cached

    # Tool definition for structured output via tool-use
    annotate_tool = {
        "name": "annotate",
        "description": "Return taxonomy category annotations for the trial.",
        "input_schema": _ANNOTATION_SCHEMA,
    }

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(  # type: ignore[call-overload]
            model=model_id,
            max_tokens=2048,
            tools=[annotate_tool],
            tool_choice={"type": "tool", "name": "annotate"},
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract categories from the tool_use content block
        categories: list[dict] = []
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "annotate":
                tool_input = block.input
                if isinstance(tool_input, dict):
                    categories = tool_input.get("categories", [])
                break

        if not isinstance(categories, list):
            logger.warning("LLM returned unexpected type for %s", trial_dir)
            annotation: AnnotationResult = AnnotationError(
                reason="LLM returned unexpected type"
            )
            return _unwrap_result(annotation)
        validated = validate_categories(categories, trial_dir)
        annotation = _to_annotation_result(validated)
        _pkg.put_cached(
            DEFAULT_CACHE_DIR, key, validated, is_error=_is_error(annotation)
        )
        return _unwrap_result(annotation)
    except Exception as exc:
        logger.error("API error annotating %s: %s", trial_dir, exc)
        annotation = AnnotationError(reason=f"API error: {exc}")
        return _unwrap_result(annotation)


def annotate_batch_api(
    trials: list[str | Path],
    signals_list: list[dict],
    model: str = "haiku",
    max_concurrent: int = 5,
) -> list[list[dict]]:
    """Annotate a batch of trials using the Anthropic API with concurrency."""
    if len(trials) != len(signals_list):
        raise ValueError(
            f"trials ({len(trials)}) and signals_list ({len(signals_list)}) "
            "must have the same length"
        )

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic SDK required -- install with: pip install agent-observatory[llm]"
        ) from None

    sem = asyncio.Semaphore(max_concurrent)
    taxonomy = _taxonomy_yaml()

    async def _annotate_one(
        idx: int,
        trial_dir: Path,
        signals: dict,
        aclient: Any,
    ) -> tuple[int, list[dict]]:
        async with sem:
            instruction = _read_text(
                trial_dir / "agent" / "instruction.txt",
                max_chars=4000,
            )
            traj = _load_json(trial_dir / "agent" / "trajectory.json")
            truncated = truncate_trajectory(traj, first_n=30, last_n=10)
            prompt = build_prompt(instruction, truncated, signals, taxonomy)
            model_id = _resolve_model_api(model)

            try:
                message = await aclient.messages.create(
                    model=model_id,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    categories = parsed.get("categories", [])
                elif isinstance(parsed, list):
                    categories = parsed
                else:
                    return idx, []
                return idx, validate_categories(categories, trial_dir)
            except json.JSONDecodeError as exc:
                logger.error(
                    "JSON parse error for trial %d (%s): %s",
                    idx,
                    trial_dir,
                    exc,
                )
                return idx, []
            except Exception as exc:
                logger.error(
                    "API error for trial %d (%s): %s",
                    idx,
                    trial_dir,
                    exc,
                )
                return idx, []

    async def _run_all() -> list[list[dict]]:
        aclient = anthropic.AsyncAnthropic()
        results: list[list[dict]] = [[] for _ in trials]
        tasks = [
            _annotate_one(i, Path(t), s, aclient)
            for i, (t, s) in enumerate(zip(trials, signals_list))
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            idx, cats = await coro
            results[idx] = cats
            done += 1
            print(f"  [{done}/{len(trials)}] annotated", file=sys.stderr)
        return results

    return asyncio.run(_run_all())
