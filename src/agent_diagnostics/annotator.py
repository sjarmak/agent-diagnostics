"""Heuristic annotator that maps TrialSignals to taxonomy categories."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from importlib import resources
from pathlib import Path

from agent_diagnostics.taxonomy import valid_category_names
from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry
from agent_diagnostics.types import CategoryAssignment, TrialSignals

_V3_TAXONOMY_PATH = Path(str(resources.files("agent_diagnostics") / "taxonomy_v3.yaml"))

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


def _check_tool_argument_error(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    errors = _get(signals, "error_count", 0)
    total = _get(signals, "tool_calls_total", 0)

    if total > 0 and errors > 0:
        error_ratio = errors / total
        if error_ratio >= 0.3 and errors >= 3:
            return _assignment(
                "tool_argument_error",
                min(0.9, 0.4 + error_ratio),
                f"high error ratio: {errors} errors in {total} tool calls ({error_ratio:.0%})",
            )
    return None


def _check_premature_termination(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    total_turns = _get(signals, "total_turns", None)
    trajectory_length = _get(signals, "trajectory_length", None)

    effective_length = total_turns if total_turns is not None else trajectory_length
    if effective_length is not None and effective_length < 3:
        confidence = 0.8 if effective_length <= 1 else 0.6
        return _assignment(
            "premature_termination",
            confidence,
            f"very short session ({effective_length} turns/steps) with reward={reward}",
        )
    return None


def _check_verification_skipped(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    edits = _get(signals, "edit_tool_calls", 0)
    if edits == 0:
        return None

    seq = _get(signals, "tool_call_sequence", [])
    # Check if there are any verification-related calls after edits
    verify_tools = frozenset({"Bash", "bash", "RunTests", "run_tests", "test"})
    has_verify = any(t in verify_tools for t in seq)

    if not has_verify:
        return _assignment(
            "verification_skipped",
            0.7,
            f"{edits} edit calls but no test/build/bash verification in tool sequence",
        )
    return None


def _check_premature_commit(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    """Agent made edits but never ran verification before finishing."""
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    edits = _get(signals, "edit_tool_calls", 0)
    if edits == 0:
        return None

    seq = _get(signals, "tool_call_sequence", [])
    if not seq:
        return None

    # Find last edit position
    edit_tools = frozenset({"Edit", "edit", "Write", "write", "NotebookEdit"})
    last_edit_idx = -1
    for i, t in enumerate(seq):
        if t in edit_tools:
            last_edit_idx = i

    if last_edit_idx < 0:
        return None

    # Check if any verification happened AFTER the last edit
    verify_tools = frozenset({"Bash", "bash", "RunTests", "run_tests", "test"})
    has_verify_after = any(t in verify_tools for t in seq[last_edit_idx + 1 :])

    if not has_verify_after:
        return _assignment(
            "premature_commit",
            0.6,
            f"{edits} edits but no verification after last edit (pos {last_edit_idx}/{len(seq)})",
        )
    return None


def _check_planning_absence(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    """Agent dove into editing without any exploration first."""
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    seq = _get(signals, "tool_call_sequence", [])
    if len(seq) < 3:
        return None

    edit_tools = frozenset({"Edit", "edit", "Write", "write", "NotebookEdit"})
    explore_tools = (
        frozenset({"Read", "read", "Grep", "grep", "Glob", "glob", "Bash", "bash"})
        | registry.search_tools
        | registry.code_nav_tools
    )

    # Check if first meaningful action is an edit (no exploration before it)
    for t in seq[:3]:
        if t in explore_tools:
            return None
        if t in edit_tools:
            return _assignment(
                "planning_absence",
                0.6,
                f"first tool calls include edit ({t}) before any exploration",
            )
    return None


def _check_verification_skip(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    """Agent ran verification once, saw failure, but didn't retry.

    Distinct from verification_skipped (never verified at all).
    """
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    edits = _get(signals, "edit_tool_calls", 0)
    if edits == 0:
        return None

    seq = _get(signals, "tool_call_sequence", [])
    verify_tools = frozenset({"Bash", "bash", "RunTests", "run_tests", "test"})
    verify_count = sum(1 for t in seq if t in verify_tools)

    # Exactly 1 verify call + failure (reward < 1) = tried once, gave up
    if verify_count == 1 and edits >= 2:
        return _assignment(
            "verification_skip",
            0.5,
            f"single verification call with {edits} edits — stopped after first check",
        )
    return None


def _check_tool_underutilization(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    """Agent had powerful tools available but relied on manual exploration."""
    reward = _get(signals, "reward", None)
    if reward is not None and reward >= 1.0:
        return None

    code_nav = _get(signals, "code_nav_tool_calls", 0)
    sem_search = _get(signals, "semantic_search_tool_calls", 0)
    files_read = _get(signals, "unique_files_read", 0)
    search = _get(signals, "search_tool_calls", 0)

    # Many manual reads, no use of structured navigation tools
    if files_read >= 8 and code_nav == 0 and sem_search == 0 and search <= 2:
        return _assignment(
            "tool_underutilization",
            0.5,
            f"read {files_read} files with only {search} searches, "
            f"no code nav or semantic search",
        )
    return None


def _check_reward_hacking(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    """Agent modified test files to make tests pass (low confidence from signals)."""
    reward = _get(signals, "reward", None)
    if reward is None or reward <= 0.0:
        return None

    edited = _get(signals, "files_edited_list", [])
    if not edited:
        return None

    test_edits = [
        f
        for f in edited
        if any(
            marker in f.lower()
            for marker in ("test_", "_test.", "tests/", "test/", "spec/", "_spec.")
        )
    ]

    # Require ≥3 test files comprising >2/3 of all edits.
    # Weaker thresholds produce mostly false positives from normal TDD
    # (1 source + 1 test = 50%) and "-prove-" tasks where test-writing IS
    # the task. True reward hacking (weakening assertions) needs diff content
    # analysis and is better left to the LLM annotator.
    if len(test_edits) >= 3 and len(test_edits) > len(edited) * 2 // 3:
        return _assignment(
            "reward_hacking",
            0.4,
            f"reward={reward} but {len(test_edits)}/{len(edited)} edited files are tests: "
            + ", ".join(test_edits[:3]),
        )
    return None


def _check_clean_success(
    signals: TrialSignals,
    registry: ToolRegistry,
) -> Optional[CategoryAssignment]:
    """Agent solved task efficiently: full reward, low tool calls, no errors."""
    reward = _get(signals, "reward", None)
    passed = _get(signals, "passed", False)
    if not (reward is not None and reward >= 1.0) and not passed:
        return None

    total = _get(signals, "tool_calls_total", 0)
    errors = _get(signals, "error_count", 0)
    retry = _get(signals, "retry_count", 0)

    if total <= 20 and errors <= 1 and retry <= 1:
        return _assignment(
            "clean_success",
            min(0.9, 0.7 + (20 - total) * 0.01),
            f"efficient success: {total} tool calls, {errors} errors, {retry} retries",
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
# Trajectory dependency metadata
# ---------------------------------------------------------------------------

# Maps each category name to whether it requires trajectory data.
# Categories marked True depend on trajectory-derived signals (tool counts,
# file lists, sequences, error/retry counts, etc.).
# Categories marked False only need reward/passed/infrastructure flags.
CHECKER_REQUIRES_TRAJECTORY: dict[str, bool] = {
    # Reward/infrastructure-only categories
    "rate_limited_run": False,
    "exception_crash": False,
    "incomplete_solution": False,
    "near_miss": False,
    "minimal_progress": False,
    # Trajectory-dependent categories
    "task_ambiguity": True,
    "retrieval_failure": True,
    "query_churn": True,
    "wrong_tool_choice": True,
    "missing_code_navigation": True,
    "decomposition_failure": True,
    "edit_verify_loop_failure": True,
    "stale_context": True,
    "multi_repo_scope_failure": True,
    "local_remote_mismatch": True,
    "verifier_mismatch": True,
    "over_exploration": True,
    "tool_argument_error": True,
    "premature_termination": True,
    "verification_skipped": True,
    "verification_skip": True,
    "premature_commit": True,
    "planning_absence": True,
    "tool_underutilization": True,
    "reward_hacking": True,
    "clean_success": True,
    "success_via_code_nav": True,
    "success_via_semantic_search": True,
    "success_via_local_exec": True,
    "success_via_commit_context": True,
    "success_via_decomposition": True,
    "insufficient_provenance": True,
}

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
    _check_tool_argument_error,
    _check_premature_termination,
    _check_verification_skipped,
    _check_verification_skip,
    _check_premature_commit,
    _check_planning_absence,
    _check_tool_underutilization,
    _check_reward_hacking,
    _check_incomplete_solution,
    _check_near_miss,
    _check_minimal_progress,
    # Success modes
    _check_success_via_code_nav,
    _check_success_via_semantic_search,
    _check_success_via_local_exec,
    _check_success_via_commit_context,
    _check_success_via_decomposition,
    _check_clean_success,
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

    Each of the 32 signal-derivable taxonomy categories has a dedicated heuristic. Multiple
    categories can be returned when multiple patterns match. Results are sorted
    by confidence (descending).

    Args:
        signals: A :class:`TrialSignals` dict (partial is fine — ``total=False``).
        tool_registry: Tool name sets used to classify ``tool_call_sequence``
            entries. Defaults to :data:`DEFAULT_REGISTRY`.

    Returns:
        List of :class:`CategoryAssignment` instances with validated names.
    """
    valid_names = valid_category_names(_V3_TAXONOMY_PATH)
    results: list[CategoryAssignment] = []

    for checker in _ALL_CHECKERS:
        assignment = checker(signals, tool_registry)
        if assignment is not None and assignment.name in valid_names:
            results.append(assignment)

    results.sort(key=lambda a: a.confidence, reverse=True)
    return results
