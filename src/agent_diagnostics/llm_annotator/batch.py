"""Anthropic Message Batches API backend (50% cheaper, no per-minute limits)."""

from __future__ import annotations

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


def annotate_batch_messages(
    trials: list[str | Path],
    signals_list: list[dict],
    model: str = "haiku",
    poll_interval: float = 10.0,
) -> list[AnnotationResult]:
    """Annotate trials using the Anthropic Message Batches API.

    Submits all uncached trials as a single server-side batch, polls
    until processing completes, then parses the results.  Cached trials
    are resolved locally without an API call.

    This is 50% cheaper than the standard Messages API and avoids
    per-minute rate limits.

    Parameters
    ----------
    trials : list of str or Path
        Trial directory paths.
    signals_list : list of dict
        Corresponding signal vectors (same length as *trials*).
    model : str
        Model alias or full model ID.
    poll_interval : float
        Seconds between status polls (default 10).

    Returns
    -------
    list[AnnotationResult]
        One variant per input trial.  Cached ``list[dict]`` entries are
        rewrapped as :class:`AnnotationOk` / :class:`AnnotationNoCategoriesFound`
        on read.  Batch failures surface as :class:`AnnotationError` with the
        API-supplied reason, replacing the initializer value.
    """
    import time

    # Import cache helpers via the package so test mocks on
    # ``agent_diagnostics.llm_annotator.get_cached`` / ``put_cached`` apply.
    from agent_diagnostics import llm_annotator as _pkg

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

    taxonomy = _taxonomy_yaml()
    model_id = _resolve_model_api(model)
    annotate_tool = {
        "name": "annotate",
        "description": "Return taxonomy category annotations for the trial.",
        "input_schema": _ANNOTATION_SCHEMA,
    }

    # --- Phase 1: build prompts and check cache ---
    # Initializer is an honest error — if a slot is not overwritten by a
    # cache hit, a success, or an API error (e.g. the batch API returns
    # results for only a subset of the requested custom_ids), the slot
    # surfaces as an explicit internal error instead of a false success.
    results: list[AnnotationResult] = [
        AnnotationError(reason="internal: trial not processed") for _ in trials
    ]
    batch_requests: list[dict[str, Any]] = []
    pending: dict[str, tuple[int, str, str]] = {}  # custom_id -> (idx, trial_dir, ckey)
    cached_count = 0

    for idx, (trial_path, signals) in enumerate(zip(trials, signals_list)):
        trial_dir = Path(trial_path)
        instruction = _read_text(
            trial_dir / "agent" / "instruction.txt", max_chars=4000
        )
        traj = _load_json(trial_dir / "agent" / "trajectory.json")
        truncated = truncate_trajectory(traj, first_n=30, last_n=10)
        prompt = build_prompt(instruction, truncated, signals, taxonomy)

        ckey = cache_key(prompt, model_id)
        cached = _pkg.get_cached(DEFAULT_CACHE_DIR, ckey)
        if cached is not None:
            # Cache never stores errors (put_cached skips them), so rewrapping
            # always yields AnnotationOk or AnnotationNoCategoriesFound.
            results[idx] = _to_annotation_result(cached)
            cached_count += 1
            continue

        custom_id = f"trial-{idx}"
        batch_requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model_id,
                    "max_tokens": 2048,
                    "tools": [annotate_tool],
                    "tool_choice": {"type": "tool", "name": "annotate"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
        pending[custom_id] = (idx, str(trial_dir), ckey)

    print(
        f"Batch: {cached_count} cached, {len(batch_requests)} to submit",
        file=sys.stderr,
    )

    if not batch_requests:
        print("All trials resolved from cache.", file=sys.stderr)
        return results

    # --- Phase 2: submit batch ---
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=batch_requests)
    print(
        f"Batch created: {batch.id} ({len(batch_requests)} requests)", file=sys.stderr
    )

    # --- Phase 3: poll until complete ---
    while batch.processing_status != "ended":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        total = (
            counts.processing
            + counts.succeeded
            + counts.errored
            + counts.canceled
            + counts.expired
        )
        print(
            f"  Batch {batch.id}: {counts.succeeded}/{total} succeeded, "
            f"{counts.errored} errored, {counts.processing} processing",
            file=sys.stderr,
        )

    # --- Phase 4: parse results ---
    succeeded = 0
    errored = 0
    for result in client.messages.batches.results(batch.id):
        custom_id = result.custom_id
        if custom_id not in pending:
            continue
        idx, trial_dir_str, ckey = pending[custom_id]

        if result.result.type == "succeeded":
            message = result.result.message
            categories: list[dict] = []
            for block in message.content:
                if (
                    getattr(block, "type", None) == "tool_use"
                    and block.name == "annotate"
                ):
                    tool_input = block.input
                    if isinstance(tool_input, dict):
                        categories = tool_input.get("categories", [])
                    break

            validated = validate_categories(categories, trial_dir_str)
            annotation = _to_annotation_result(validated)
            _pkg.put_cached(
                DEFAULT_CACHE_DIR, ckey, validated, is_error=_is_error(annotation)
            )
            results[idx] = annotation
            succeeded += 1
        else:
            error_msg = getattr(result.result, "error", None)
            logger.error(
                "Batch error for %s: %s",
                trial_dir_str,
                error_msg,
            )
            results[idx] = AnnotationError(
                reason=f"batch error: {error_msg}" if error_msg else "batch error"
            )
            errored += 1

    print(
        f"Batch complete: {succeeded} succeeded, {errored} errored, "
        f"{cached_count} from cache",
        file=sys.stderr,
    )
    return results
