"""Claude Code CLI subprocess backend."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path

from agent_diagnostics.annotation_cache import DEFAULT_CACHE_DIR, cache_key
from agent_diagnostics.llm_annotator.models import _resolve_model_alias
from agent_diagnostics.llm_annotator.parsing import (
    _is_error,
    _load_json,
    _parse_claude_response,
    _read_text,
    _short_exc,
    _to_annotation_result,
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


def _find_claude_cli() -> str:
    """Locate the claude CLI binary."""
    path = shutil.which("claude")
    if path is None:
        raise FileNotFoundError(
            "claude CLI not found on PATH. "
            "Install it: https://docs.anthropic.com/en/docs/claude-code"
        )
    return path


def _claude_cli_argv(claude_bin: str, schema_str: str, model_alias: str, prompt: str) -> list[str]:
    """Build the argv for a ``claude -p`` invocation with structured JSON output."""
    return [
        claude_bin,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        schema_str,
        "--model",
        model_alias,
        "--no-session-persistence",
        prompt,
    ]


def annotate_trial_claude_code(
    trial_dir: str | Path,
    signals: dict,
    model: str = "haiku",
) -> AnnotationResult:
    """Annotate a single trial by spawning a ``claude -p`` subprocess.

    Uses the currently authenticated Claude Code account -- no API key needed.

    Returns an :class:`AnnotationResult` — :class:`AnnotationOk` with the
    validated categories, :class:`AnnotationNoCategoriesFound` when the model
    returned an empty list, or :class:`AnnotationError` carrying the failure
    reason (subprocess rc, timeout, parse error, etc.).
    """
    # Import cache helpers via the package so that tests patching
    # ``agent_diagnostics.llm_annotator.get_cached`` / ``put_cached`` affect
    # this call site at runtime.
    from agent_diagnostics import llm_annotator as _pkg

    claude_bin = _find_claude_cli()
    trial_dir = Path(trial_dir)

    instruction = _read_text(trial_dir / "agent" / "instruction.txt", max_chars=4000)
    traj = _load_json(trial_dir / "agent" / "trajectory.json")
    truncated = truncate_trajectory(traj, first_n=30, last_n=10)
    taxonomy = _taxonomy_yaml()
    prompt = build_prompt(instruction, truncated, signals, taxonomy)
    model_alias = _resolve_model_alias(model)

    # Cache lookup
    key = cache_key(prompt, model_alias)
    cached = _pkg.get_cached(DEFAULT_CACHE_DIR, key)
    if cached is not None:
        logger.debug("Cache hit for %s (key=%s)", trial_dir, key[:12])
        return _to_annotation_result(cached)

    schema_str = json.dumps(_ANNOTATION_SCHEMA)

    try:
        proc_result = subprocess.run(
            _claude_cli_argv(claude_bin, schema_str, model_alias, prompt),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc_result.returncode != 0:
            logger.error(
                "claude CLI failed for %s (rc=%d): %s",
                trial_dir,
                proc_result.returncode,
                proc_result.stderr[:500],
            )
            return AnnotationError(reason=f"claude CLI rc={proc_result.returncode}")

        envelope = json.loads(proc_result.stdout)
        categories = _parse_claude_response(envelope)
        if categories is None:
            return AnnotationError(reason="Failed to parse claude response")
        validated = validate_categories(categories, trial_dir)
        annotation = _to_annotation_result(validated)
        _pkg.put_cached(DEFAULT_CACHE_DIR, key, validated, is_error=_is_error(annotation))
        return annotation

    except subprocess.TimeoutExpired:
        logger.error("claude CLI timed out for %s", trial_dir)
        return AnnotationError(reason="claude CLI timed out")
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse claude CLI output for %s: %s", trial_dir, exc)
        return AnnotationError(reason=f"JSON decode error: {_short_exc(exc)}")
    except Exception as exc:
        logger.error("Error running claude CLI for %s: %s", trial_dir, exc)
        return AnnotationError(reason=f"Unexpected error: {_short_exc(exc)}")


async def _annotate_one_claude_code(
    idx: int,
    trial_dir: Path,
    signals: dict,
    model: str,
    sem: asyncio.Semaphore,
    claude_bin: str,
    taxonomy: str,
) -> tuple[int, AnnotationResult]:
    """Async wrapper: spawn one claude -p subprocess under a semaphore.

    Returns ``(idx, AnnotationResult)`` so errors propagate to the caller
    with their reason intact instead of being silently unwrapped to ``[]``.
    """
    async with sem:
        instruction = _read_text(trial_dir / "agent" / "instruction.txt", max_chars=4000)
        traj = _load_json(trial_dir / "agent" / "trajectory.json")
        truncated = truncate_trajectory(traj, first_n=30, last_n=10)
        prompt = build_prompt(instruction, truncated, signals, taxonomy)
        model_alias = _resolve_model_alias(model)
        schema_str = json.dumps(_ANNOTATION_SCHEMA)

        try:
            proc = await asyncio.create_subprocess_exec(
                *_claude_cli_argv(claude_bin, schema_str, model_alias, prompt),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

            if proc.returncode != 0:
                logger.error(
                    "claude CLI failed for trial %d (%s): %s",
                    idx,
                    trial_dir,
                    stderr.decode()[:500],
                )
                return idx, AnnotationError(reason=f"claude CLI rc={proc.returncode}")

            envelope = json.loads(stdout.decode())
            categories = _parse_claude_response(envelope)
            if categories is None:
                return idx, AnnotationError(reason="Failed to parse claude response")
            validated = validate_categories(categories, trial_dir)
            return idx, _to_annotation_result(validated)

        except asyncio.TimeoutError:
            logger.error("claude CLI timed out for trial %d (%s)", idx, trial_dir)
            return idx, AnnotationError(reason="claude CLI timed out")
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error for trial %d (%s): %s", idx, trial_dir, exc)
            return idx, AnnotationError(reason=f"JSON decode error: {_short_exc(exc)}")
        except Exception as exc:
            logger.error("Error for trial %d (%s): %s", idx, trial_dir, exc)
            return idx, AnnotationError(reason=f"Unexpected error: {_short_exc(exc)}")


def annotate_batch_claude_code(
    trials: list[str | Path],
    signals_list: list[dict],
    model: str = "haiku",
    max_concurrent: int = 5,
) -> list[AnnotationResult]:
    """Annotate a batch of trials using parallel ``claude -p`` subprocesses.

    Returns a list of :class:`AnnotationResult` variants — one per input trial,
    preserving error reasons through the async boundary instead of collapsing
    failures to empty category lists.
    """
    if len(trials) != len(signals_list):
        raise ValueError(
            f"trials ({len(trials)}) and signals_list ({len(signals_list)}) "
            "must have the same length"
        )

    claude_bin = _find_claude_cli()
    taxonomy = _taxonomy_yaml()
    sem = asyncio.Semaphore(max_concurrent)

    async def _run_all() -> list[AnnotationResult]:
        # Initializer is an honest error — any slot surviving the loop
        # (task cancellation, scheduler bug) surfaces as an explicit error
        # instead of a silent empty-success.
        results: list[AnnotationResult] = [
            AnnotationError(reason="internal: trial not processed") for _ in trials
        ]
        tasks = [
            _annotate_one_claude_code(
                i,
                Path(t),
                s,
                model,
                sem,
                claude_bin,
                taxonomy,
            )
            for i, (t, s) in enumerate(zip(trials, signals_list))
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result
            done += 1
            logger.info("  [%d/%d] annotated", done, len(trials))
        return results

    return asyncio.run(_run_all())
