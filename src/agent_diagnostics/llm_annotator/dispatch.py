"""Unified dispatch for the three LLM annotator backends.

Backend entry points are resolved via lazy attribute lookup on the package
namespace (``agent_diagnostics.llm_annotator``) rather than direct imports,
so test-time ``mock.patch("agent_diagnostics.llm_annotator.annotate_trial_claude_code", ...)``
takes effect on the binding this module actually calls.

The public return type is :class:`AnnotationResult` (single trial) or
``list[AnnotationResult]`` (batch).  Error reasons from subprocess / API
failures propagate through this layer instead of being silently unwrapped
to empty category lists.
"""

from __future__ import annotations

from pathlib import Path

from agent_diagnostics import llm_annotator as _pkg
from agent_diagnostics.types import AnnotationResult

_VALID_BACKENDS = ("claude-code", "api", "batch")


def _reject_unknown_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'claude-code', 'api', or 'batch'.")


def annotate_trial_llm(
    trial_dir: str | Path,
    signals: dict,
    model: str = "haiku",
    backend: str = "claude-code",
) -> AnnotationResult:
    """Annotate a single trial using an LLM.

    ``backend`` is one of ``'claude-code'`` (default, via the ``claude`` CLI),
    ``'api'`` (Anthropic Python SDK), or ``'batch'`` — which falls back to the
    API path because the Message Batches API has no meaningful single-trial
    flavour; use :func:`annotate_batch` to benefit from it.
    """
    _reject_unknown_backend(backend)
    if backend == "claude-code":
        return _pkg.annotate_trial_claude_code(trial_dir, signals, model)
    # Both 'api' and 'batch' route here for single-trial calls.
    return _pkg.annotate_trial_api(trial_dir, signals, model)


def annotate_batch(
    trials: list[str | Path],
    signals_list: list[dict],
    model: str = "haiku",
    max_concurrent: int = 5,
    backend: str = "claude-code",
) -> list[AnnotationResult]:
    """Annotate a batch of trials using the LLM, with concurrency control.

    ``backend`` is one of ``'claude-code'`` (default), ``'api'``, or
    ``'batch'`` (Anthropic Message Batches API, 50% cheaper).
    """
    _reject_unknown_backend(backend)
    if backend == "claude-code":
        return _pkg.annotate_batch_claude_code(trials, signals_list, model, max_concurrent)
    if backend == "api":
        return _pkg.annotate_batch_api(trials, signals_list, model, max_concurrent)
    return _pkg.annotate_batch_messages(trials, signals_list, model)
