"""Unified dispatch for the three LLM annotator backends.

Backend entry points are resolved via lazy attribute lookup on the package
namespace (``agent_diagnostics.llm_annotator``) rather than direct imports,
so test-time ``mock.patch("agent_diagnostics.llm_annotator.annotate_trial_claude_code", ...)``
takes effect on the binding this module actually calls.
"""

from __future__ import annotations

from pathlib import Path

from agent_diagnostics import llm_annotator as _pkg


def annotate_trial_llm(
    trial_dir: str | Path,
    signals: dict,
    model: str = "haiku",
    backend: str = "claude-code",
) -> list[dict]:
    """Annotate a single trial using an LLM.

    Parameters
    ----------
    trial_dir : str or Path
        Path to the trial directory.
    signals : dict
        Pre-extracted signal vector for this trial.
    model : str
        Model alias (``'haiku'``, ``'sonnet'``, ``'opus'``) or full model ID.
    backend : str
        ``'claude-code'`` (default) uses the ``claude`` CLI.
        ``'api'`` uses the Anthropic Python SDK.
        ``'batch'`` is not supported for single-trial calls; use
        :func:`annotate_batch` instead.
    """
    if backend == "claude-code":
        return _pkg.annotate_trial_claude_code(trial_dir, signals, model)
    elif backend == "api":
        return _pkg.annotate_trial_api(trial_dir, signals, model)
    elif backend == "batch":
        return _pkg.annotate_trial_api(trial_dir, signals, model)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'claude-code', 'api', or 'batch'."
        )


def annotate_batch(
    trials: list[str | Path],
    signals_list: list[dict],
    model: str = "haiku",
    max_concurrent: int = 5,
    backend: str = "claude-code",
) -> list[list[dict]]:
    """Annotate a batch of trials using the LLM, with concurrency control.

    Parameters
    ----------
    trials : list of str or Path
        Trial directory paths.
    signals_list : list of dict
        Corresponding signal vectors (same length as *trials*).
    model : str
        Model alias or full model ID.
    max_concurrent : int
        Maximum number of concurrent calls/subprocesses.
    backend : str
        ``'claude-code'`` (default), ``'api'``, or ``'batch'``.
    """
    if backend == "claude-code":
        return _pkg.annotate_batch_claude_code(trials, signals_list, model, max_concurrent)
    elif backend == "api":
        return _pkg.annotate_batch_api(trials, signals_list, model, max_concurrent)
    elif backend == "batch":
        return _pkg.annotate_batch_messages(trials, signals_list, model)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'claude-code', 'api', or 'batch'."
        )
