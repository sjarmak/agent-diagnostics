"""Heuristic annotator that maps TrialSignals to taxonomy categories."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from agent_diagnostics.taxonomy import valid_category_names
from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry
from agent_diagnostics.types import CategoryAssignment, TrialSignals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(signals: TrialSignals, key: str, default: Any = None) -> Any:
    """Safely access a TrialSignals key (total=False means any key may be absent)."""
    return signals.get(key, default)  # type: ignore[arg-type]


def _count_tools_in_sequence(
    sequence: Sequence[str],
    tool_set: frozenset[str],
) -> int:
    """Count how many entries in *sequence* belong to *tool_set*."""
    return sum(1 for t in sequence if t in tool_set)


def _has_bash_in_sequence(sequence: Sequence[str]) -> bool:
    """Return True if the sequence contains Bash tool calls."""
    return any(t == "Bash" for t in sequence)


def _assignment(name: str, confidence: float, evidence: str) -> CategoryAssignment:
    """Create a CategoryAssignment with clamped confidence."""
    return CategoryAssignment(
        name=name,
        confidence=max(0.0, min(1.0, confidence)),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Neutral / infrastructure checks (run first, can coexist with others)
# ---------------------------------------------------------------------------


def _check_rate_limited(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    if _get(signals, "rate_limited", False):
        return _assignment("rate_limited_run", 0.9, "rate_limited flag is True")
    return None


def _check_exception_crash(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    if _get(signals, "exception_crashed", False):
        return _assignment("exception_crash", 0.9, "exception_crashed flag is True")
    return None


def _check_task_ambiguity(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None:
        return None
    total = _get(signals, "tool_calls_total", 0)
    duration = _get(signals, "duration_seconds", 0.0)
    crashed = _get(signals, "exception_crashed", False)
    limited = _get(signals, "rate_limited", False)
    edits = _get(signals, "edit_tool_calls", 0)

    if (
        reward == 0.0
        and not crashed
        and not limited
        and total < 5
        and duration < 60.0
        and edits == 0
    ):
        return _assignment(
            "task_ambiguity",
            0.4,
            f"zero reward with very few tool calls ({total}) and short duration ({duration:.0f}s)",
        )
    return None


# ---------------------------------------------------------------------------
# Failure heuristics (generally reward < 1.0)
# ---------------------------------------------------------------------------


def _check_retrieval_failure(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", None)
    if reward is None or reward > 0.0:
        return None
    if passed:
        return None

    search = _get(signals, "search_tool_calls", 0)
    files_read = _get(signals, "unique_files_read", 0)

    if search <= 1 and files_read <= 1:
        confidence = 0.9 if (search == 0 and files_read == 0) else 0.7
        return _assignment(
            "retrieval_failure",
            confidence,
            f"reward=0 with search_tool_calls={search}, unique_files_read={files_read}",
        )
    return None


def _check_query_churn(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward >= 1.0:
        return None

    search = _get(signals, "search_tool_calls", 0)
    files_read = _get(signals, "unique_files_read", 0)

    if search > 5 and files_read < search // 2:
        return _assignment(
            "query_churn",
            min(0.9, 0.5 + (search - 5) * 0.05),
            f"high search calls ({search}) with few file reads ({files_read})",
        )
    return None


def _check_wrong_tool_choice(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward >= 1.0:
        return None

    search = _get(signals, "search_tool_calls", 0)
    code_nav = _get(signals, "code_nav_tool_calls", 0)
    files_read = _get(signals, "unique_files_read", 0)

    if search > 5 and code_nav == 0 and files_read > 3:
        return _assignment(
            "wrong_tool_choice",
            0.6,
            f"high search ({search}) with no code navigation, {files_read} files read manually",
        )
    return None


def _check_missing_code_navigation(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward >= 1.0:
        return None

    code_nav = _get(signals, "code_nav_tool_calls", 0)
    files_read = _get(signals, "unique_files_read", 0)
    edits = _get(signals, "unique_files_edited", 0)

    if code_nav == 0 and files_read >= 5 and edits >= 2:
        return _assignment(
            "missing_code_navigation",
            0.6,
            f"no code navigation calls despite reading {files_read} files and editing {edits}",
        )
    return None


def _check_decomposition_failure(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward >= 1.0:
        return None

    edits = _get(signals, "unique_files_edited", 0)
    total = _get(signals, "tool_calls_total", 0)

    if reward == 0.0 and edits >= 2 and total > 10:
        return _assignment(
            "decomposition_failure",
            0.5,
            f"reward=0 despite editing {edits} files across {total} tool calls",
        )
    return None


def _check_edit_verify_loop_failure(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward >= 1.0:
        return None

    retry = _get(signals, "retry_count", 0)
    errors = _get(signals, "error_count", 0)
    edits = _get(signals, "edit_tool_calls", 0)

    if retry > 3 and edits > 3:
        return _assignment(
            "edit_verify_loop_failure",
            min(0.9, 0.5 + retry * 0.05),
            f"edit-verify loop: {retry} retries, {errors} errors, {edits} edits",
        )
    if errors > 3 and edits > 3:
        return _assignment(
            "edit_verify_loop_failure",
            min(0.8, 0.4 + errors * 0.05),
            f"edit-verify loop: {errors} errors across {edits} edits",
        )
    return None


def _check_stale_context(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward >= 1.0:
        return None

    trajectory_len = _get(signals, "trajectory_length", 0)
    total = _get(signals, "tool_calls_total", 0)
    files_read = _get(signals, "unique_files_read", 0)
    edits = _get(signals, "unique_files_edited", 0)

    if trajectory_len > 50 and edits > 0 and files_read > 5 and total > 40:
        return _assignment(
            "stale_context",
            0.4,
            f"long trajectory ({trajectory_len} steps, {total} calls) — "
            f"stale context likely with {files_read} reads and {edits} edits",
        )
    return None


def _check_multi_repo_scope_failure(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward > 0.0:
        return None

    files_read = _get(signals, "unique_files_read", 0)
    edits = _get(signals, "unique_files_edited", 0)
    total = _get(signals, "tool_calls_total", 0)

    if edits <= 1 and files_read <= 3 and total > 10:
        return _assignment(
            "multi_repo_scope_failure",
            0.4,
            f"narrow scope: only {edits} files edited, {files_read} read in {total} calls",
        )
    return None


def _check_local_remote_mismatch(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward > 0.0:
        return None

    errors = _get(signals, "error_count", 0)
    edits = _get(signals, "edit_tool_calls", 0)
    total = _get(signals, "tool_calls_total", 0)

    if errors > 5 and edits <= 1 and total > 5:
        return _assignment(
            "local_remote_mismatch",
            0.5,
            f"many errors ({errors}) with few edits ({edits}) — possible environment mismatch",
        )
    return None


def _check_verifier_mismatch(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward > 0.0:
        return None

    edits = _get(signals, "edit_tool_calls", 0)
    patch = _get(signals, "patch_size_lines", 0)

    if edits > 0 and patch > 0:
        return _assignment(
            "verifier_mismatch",
            0.3,
            f"reward=0 despite {edits} edits producing {patch} lines of patch",
        )
    return None


def _check_over_exploration(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward > 0.0:
        return None

    total = _get(signals, "tool_calls_total", 0)
    edits = _get(signals, "edit_tool_calls", 0)
    files_read = _get(signals, "unique_files_read", 0)

    if total > 80 and edits <= 3:
        return _assignment(
            "over_exploration",
            min(0.9, 0.5 + (total - 80) * 0.005),
            f"excessive exploration: {total} tool calls, {files_read} files read, only {edits} edits",
        )
    return None


def _check_incomplete_solution(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None:
        return None
    if 0.0 < reward < 1.0:
        return _assignment(
            "incomplete_solution",
            0.8,
            f"partial reward ({reward}) indicates incomplete solution",
        )
    return None


def _check_near_miss(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None:
        return None
    if 0.5 <= reward < 1.0:
        return _assignment(
            "near_miss",
            0.7 + (reward - 0.5) * 0.4,
            f"near miss with reward={reward}",
        )
    return None


def _check_minimal_progress(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None:
        return None
    if 0.0 < reward < 0.5:
        return _assignment(
            "minimal_progress",
            0.7,
            f"minimal progress with reward={reward}",
        )
    return None


# ---------------------------------------------------------------------------
# Success heuristics (reward >= 1.0 or passed=True)
# ---------------------------------------------------------------------------


def _check_success_via_code_nav(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", False)
    if not (reward is not None and reward >= 1.0) and not passed:
        return None

    code_nav = _get(signals, "code_nav_tool_calls", 0)
    seq = _get(signals, "tool_call_sequence", [])
    nav_from_seq = _count_tools_in_sequence(seq, registry.code_nav_tools)
    effective = max(code_nav, nav_from_seq)

    if effective > 0:
        confidence = min(0.9, 0.5 + effective * 0.05)
        return _assignment(
            "success_via_code_nav",
            confidence,
            f"success with {effective} code navigation calls",
        )
    return None


def _check_success_via_semantic_search(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", False)
    if not (reward is not None and reward >= 1.0) and not passed:
        return None

    sem = _get(signals, "semantic_search_tool_calls", 0)
    seq = _get(signals, "tool_call_sequence", [])
    sem_from_seq = _count_tools_in_sequence(seq, registry.semantic_search_tools)
    effective = max(sem, sem_from_seq)

    if effective > 0:
        return _assignment(
            "success_via_semantic_search",
            min(0.9, 0.6 + effective * 0.1),
            f"success using {effective} semantic search calls",
        )
    return None


def _check_success_via_local_exec(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", False)
    if not (reward is not None and reward >= 1.0) and not passed:
        return None

    seq = _get(signals, "tool_call_sequence", [])
    edits = _get(signals, "edit_tool_calls", 0)

    if _has_bash_in_sequence(seq) and edits > 0:
        return _assignment(
            "success_via_local_exec",
            0.7,
            f"success with Bash execution and {edits} edits (edit-test loop)",
        )
    return None


def _check_success_via_commit_context(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", False)
    if not (reward is not None and reward >= 1.0) and not passed:
        return None

    seq = _get(signals, "tool_call_sequence", [])
    # Detect git-related tools: commit_search, compare_revisions, diff_search
    git_tools = frozenset(
        t
        for t in registry.all_tools
        if any(kw in t.lower() for kw in ("commit", "diff", "compare", "blame"))
    )
    git_count = _count_tools_in_sequence(seq, git_tools)

    if git_count > 0:
        return _assignment(
            "success_via_commit_context",
            min(0.8, 0.5 + git_count * 0.1),
            f"success leveraging {git_count} git/commit context calls",
        )
    return None


def _check_success_via_decomposition(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", False)
    if not (reward is not None and reward >= 1.0) and not passed:
        return None

    edits = _get(signals, "unique_files_edited", 0)

    if edits >= 3:
        return _assignment(
            "success_via_decomposition",
            min(0.8, 0.4 + edits * 0.1),
            f"success editing {edits} files — multi-step decomposition",
        )
    return None


def _check_insufficient_provenance(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is None or reward <= 0.0:
        return None

    total = _get(signals, "tool_calls_total", 0)
    search = _get(signals, "search_tool_calls", 0)
    code_nav = _get(signals, "code_nav_tool_calls", 0)

    if total <= 5 and search == 0 and code_nav == 0:
        return _assignment(
            "insufficient_provenance",
            0.6,
            f"reward={reward} with only {total} tool calls and no search/navigation — unclear provenance",
        )
    return None


# ---------------------------------------------------------------------------
# All heuristic checkers
# ---------------------------------------------------------------------------

_ALL_CHECKERS = (
    # Neutral / infrastructure
    _check_rate_limited,
    _check_exception_crash,
    _check_task_ambiguity,
    # Failure modes
    _check_retrieval_failure,
    _check_query_churn,
    _check_wrong_tool_choice,
    _check_missing_code_navigation,
    _check_decomposition_failure,
    _check_edit_verify_loop_failure,
    _check_stale_context,
    _check_multi_repo_scope_failure,
    _check_local_remote_mismatch,
    _check_verifier_mismatch,
    _check_over_exploration,
    _check_incomplete_solution,
    _check_near_miss,
    _check_minimal_progress,
    # Success modes
    _check_success_via_code_nav,
    _check_success_via_semantic_search,
    _check_success_via_local_exec,
    _check_success_via_commit_context,
    _check_success_via_decomposition,
    # Neutral (post-success)
    _check_insufficient_provenance,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_trial(
    signals: TrialSignals,
    *,
    tool_registry: ToolRegistry = DEFAULT_REGISTRY,
) -> list[CategoryAssignment]:
    """Apply heuristic rules to *signals* and return matching taxonomy categories.

    Each of the 23 taxonomy categories has a dedicated heuristic. Multiple
    categories can be returned when multiple patterns match. Results are sorted
    by confidence (descending).

    Args:
        signals: A :class:`TrialSignals` dict (partial is fine — ``total=False``).
        tool_registry: Tool name sets used to classify ``tool_call_sequence``
            entries. Defaults to :data:`DEFAULT_REGISTRY`.

    Returns:
        List of :class:`CategoryAssignment` instances with validated names.
    """
    valid_names = valid_category_names()
    results: list[CategoryAssignment] = []

    for checker in _ALL_CHECKERS:
        assignment = checker(signals, tool_registry)
        if assignment is not None and assignment.name in valid_names:
            results.append(assignment)

    results.sort(key=lambda a: a.confidence, reverse=True)
    return results
