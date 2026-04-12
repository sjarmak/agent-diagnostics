"""Tests for the heuristic annotator."""

from __future__ import annotations

import pytest

from pathlib import Path

from agent_diagnostics.annotator import annotate_trial
from agent_diagnostics.taxonomy import valid_category_names
from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry
from agent_diagnostics.types import CategoryAssignment, TrialSignals

_V3_TAXONOMY = (
    Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v3.yaml"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _names(results: list[CategoryAssignment]) -> set[str]:
    """Extract category names from a list of assignments."""
    return {r.name for r in results}


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_annotate_trial(self) -> None:
        from agent_diagnostics.annotator import annotate_trial as fn

        assert callable(fn)


# ---------------------------------------------------------------------------
# Acceptance-criteria scenarios
# ---------------------------------------------------------------------------


class TestRetrievalFailure:
    """AC: reward=0.0, passed=False, search_tool_calls=0, unique_files_read=0."""

    def test_basic(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "search_tool_calls": 0,
            "unique_files_read": 0,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "retrieval_failure" in names

    def test_confidence_high_when_zero(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "search_tool_calls": 0,
            "unique_files_read": 0,
        }
        results = annotate_trial(signals)
        rf = next(r for r in results if r.name == "retrieval_failure")
        assert rf.confidence >= 0.8


class TestSuccessViaCodeNav:
    """AC: reward=1.0, passed=True, code_nav_tool_calls>5."""

    def test_basic(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "code_nav_tool_calls": 6,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "success_via_code_nav" in names

    def test_evidence_present(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "code_nav_tool_calls": 10,
        }
        results = annotate_trial(signals)
        nav = next(r for r in results if r.name == "success_via_code_nav")
        assert nav.evidence is not None
        assert "10" in nav.evidence


class TestRateLimited:
    """AC: rate_limited=True -> rate_limited_run."""

    def test_basic(self) -> None:
        signals: TrialSignals = {"rate_limited": True}
        results = annotate_trial(signals)
        names = _names(results)
        assert "rate_limited_run" in names

    def test_coexists_with_failure(self) -> None:
        signals: TrialSignals = {
            "rate_limited": True,
            "reward": 0.0,
            "passed": False,
            "search_tool_calls": 0,
            "unique_files_read": 0,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "rate_limited_run" in names
        assert "retrieval_failure" in names


class TestExceptionCrash:
    """AC: exception_crashed=True -> exception_crash."""

    def test_basic(self) -> None:
        signals: TrialSignals = {"exception_crashed": True}
        results = annotate_trial(signals)
        names = _names(results)
        assert "exception_crash" in names


class TestCustomToolRegistry:
    """AC: works with a custom ToolRegistry that defines different tool names."""

    def test_custom_registry_code_nav(self) -> None:
        custom = ToolRegistry(
            search_tools=frozenset({"my_search"}),
            edit_tools=frozenset({"my_edit"}),
            code_nav_tools=frozenset({"my_nav_tool"}),
            semantic_search_tools=frozenset({"my_semantic"}),
        )
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "code_nav_tool_calls": 0,
            "tool_call_sequence": ["my_nav_tool", "my_nav_tool", "my_nav_tool"],
        }
        results = annotate_trial(signals, tool_registry=custom)
        names = _names(results)
        assert "success_via_code_nav" in names

    def test_custom_registry_semantic(self) -> None:
        custom = ToolRegistry(
            search_tools=frozenset({"s"}),
            edit_tools=frozenset({"e"}),
            code_nav_tools=frozenset({"n"}),
            semantic_search_tools=frozenset({"my_deep_search"}),
        )
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "semantic_search_tool_calls": 0,
            "tool_call_sequence": ["my_deep_search"],
        }
        results = annotate_trial(signals, tool_registry=custom)
        names = _names(results)
        assert "success_via_semantic_search" in names


class TestAllNamesValid:
    """AC: all returned names are valid taxonomy category names."""

    @pytest.mark.parametrize(
        "signals",
        [
            {
                "reward": 0.0,
                "passed": False,
                "search_tool_calls": 0,
                "unique_files_read": 0,
            },
            {"reward": 1.0, "passed": True, "code_nav_tool_calls": 6},
            {"rate_limited": True},
            {"exception_crashed": True},
            {"reward": 0.5, "passed": False},
            {"reward": 0.2, "passed": False},
            {
                "reward": 0.0,
                "passed": False,
                "tool_calls_total": 120,
                "edit_tool_calls": 1,
                "unique_files_read": 30,
            },
        ],
    )
    def test_names_valid(self, signals: TrialSignals) -> None:
        valid = valid_category_names(_V3_TAXONOMY)
        results = annotate_trial(signals)
        for r in results:
            assert r.name in valid, f"{r.name} is not a valid taxonomy category"


class TestMultipleCategories:
    """AC: multiple categories can be returned for the same trial."""

    def test_rate_limited_plus_retrieval(self) -> None:
        signals: TrialSignals = {
            "rate_limited": True,
            "reward": 0.0,
            "passed": False,
            "search_tool_calls": 0,
            "unique_files_read": 0,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert len(names) >= 2

    def test_incomplete_and_near_miss(self) -> None:
        signals: TrialSignals = {"reward": 0.7, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "incomplete_solution" in names
        assert "near_miss" in names

    def test_incomplete_and_minimal_progress(self) -> None:
        signals: TrialSignals = {"reward": 0.3, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "incomplete_solution" in names
        assert "minimal_progress" in names


class TestIncompleteSolution:
    def test_partial_reward(self) -> None:
        signals: TrialSignals = {"reward": 0.6, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "incomplete_solution" in names


class TestNearMiss:
    def test_high_partial_reward(self) -> None:
        signals: TrialSignals = {"reward": 0.8, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "near_miss" in names


class TestMinimalProgress:
    def test_low_partial_reward(self) -> None:
        signals: TrialSignals = {"reward": 0.2, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "minimal_progress" in names


class TestOverExploration:
    def test_many_calls_no_reward(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "tool_calls_total": 150,
            "edit_tool_calls": 2,
            "unique_files_read": 40,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "over_exploration" in names


class TestSuccessViaDecomposition:
    def test_multiple_files_edited(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "unique_files_edited": 5,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "success_via_decomposition" in names


class TestInsufficientProvenance:
    def test_few_calls_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "tool_calls_total": 3,
            "search_tool_calls": 0,
            "code_nav_tool_calls": 0,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "insufficient_provenance" in names


class TestSortedByConfidence:
    def test_descending_confidence(self) -> None:
        signals: TrialSignals = {
            "rate_limited": True,
            "reward": 0.0,
            "passed": False,
            "search_tool_calls": 0,
            "unique_files_read": 0,
        }
        results = annotate_trial(signals)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)


class TestEmptySignals:
    def test_empty_dict_returns_list(self) -> None:
        results = annotate_trial({})  # type: ignore[typeddict-item]
        assert isinstance(results, list)

    def test_empty_returns_no_crash(self) -> None:
        results = annotate_trial({})  # type: ignore[typeddict-item]
        for r in results:
            assert isinstance(r, CategoryAssignment)


class TestRewardHacking:
    """AC: reward_hacking is a valid taxonomy category."""

    def test_reward_hacking_in_taxonomy(self) -> None:
        valid = valid_category_names(_V3_TAXONOMY)
        assert "reward_hacking" in valid

    def test_reward_hacking_not_detected_by_heuristic(self) -> None:
        """reward_hacking requires LLM judgment, not heuristic detection."""
        signals: TrialSignals = {"reward": 0.0, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "reward_hacking" not in names


class TestFabricatedSuccess:
    """AC: fabricated_success is a valid taxonomy category."""

    def test_fabricated_success_in_taxonomy(self) -> None:
        valid = valid_category_names(_V3_TAXONOMY)
        assert "fabricated_success" in valid

    def test_fabricated_success_not_detected_by_heuristic(self) -> None:
        """fabricated_success requires LLM judgment, not heuristic detection."""
        signals: TrialSignals = {"reward": 1.0, "passed": True}
        results = annotate_trial(signals)
        names = _names(results)
        assert "fabricated_success" not in names


class TestHallucinatedApi:
    """AC: hallucinated_api is a valid taxonomy category."""

    def test_hallucinated_api_in_taxonomy(self) -> None:
        valid = valid_category_names(_V3_TAXONOMY)
        assert "hallucinated_api" in valid

    def test_hallucinated_api_not_detected_by_heuristic(self) -> None:
        """hallucinated_api requires LLM judgment, not heuristic detection."""
        signals: TrialSignals = {"reward": 0.0, "passed": False}
        results = annotate_trial(signals)
        names = _names(results)
        assert "hallucinated_api" not in names


class TestToolArgumentError:
    """AC: tool_argument_error detected when error_count is high relative to tool_calls_total."""

    def test_tool_argument_error_high_error_ratio(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "error_count": 6,
            "tool_calls_total": 10,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "tool_argument_error" in names

    def test_tool_argument_error_not_triggered_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "error_count": 6,
            "tool_calls_total": 10,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "tool_argument_error" not in names

    def test_tool_argument_error_not_triggered_low_ratio(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "error_count": 1,
            "tool_calls_total": 20,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "tool_argument_error" not in names

    def test_tool_argument_error_evidence_present(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "error_count": 5,
            "tool_calls_total": 10,
        }
        results = annotate_trial(signals)
        tae = next(r for r in results if r.name == "tool_argument_error")
        assert "error" in tae.evidence.lower()


class TestPrematureTermination:
    """AC: premature_termination detected when total_turns or trajectory_length is very low."""

    def test_premature_termination_low_turns(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "total_turns": 1,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "premature_termination" in names

    def test_premature_termination_low_trajectory_length(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "trajectory_length": 2,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "premature_termination" in names

    def test_premature_termination_not_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "total_turns": 1,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "premature_termination" not in names

    def test_premature_termination_not_normal_length(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "total_turns": 10,
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "premature_termination" not in names

    def test_premature_termination_confidence_single_turn(self) -> None:
        signals_1: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "total_turns": 1,
        }
        signals_2: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "total_turns": 2,
        }
        r1 = annotate_trial(signals_1)
        r2 = annotate_trial(signals_2)
        pt1 = next(r for r in r1 if r.name == "premature_termination")
        pt2 = next(r for r in r2 if r.name == "premature_termination")
        assert pt1.confidence > pt2.confidence


class TestVerificationSkipped:
    """AC: verification_skipped detected when edits exist but no verify calls."""

    def test_verification_skipped_edits_no_verify(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 3,
            "tool_call_sequence": ["Read", "Edit", "Edit", "Edit"],
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "verification_skipped" in names

    def test_verification_skipped_not_with_bash(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 2,
            "tool_call_sequence": ["Read", "Edit", "Edit", "Bash"],
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "verification_skipped" not in names

    def test_verification_skipped_not_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "edit_tool_calls": 2,
            "tool_call_sequence": ["Read", "Edit", "Edit"],
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "verification_skipped" not in names

    def test_verification_skipped_not_no_edits(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 0,
            "tool_call_sequence": ["Read", "Grep"],
        }
        results = annotate_trial(signals)
        names = _names(results)
        assert "verification_skipped" not in names

    def test_verification_skipped_evidence_present(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 2,
            "tool_call_sequence": ["Read", "Edit", "Edit"],
        }
        results = annotate_trial(signals)
        vs = next(r for r in results if r.name == "verification_skipped")
        assert "edit" in vs.evidence.lower()


class TestCleanSuccess:
    """AC: clean_success when reward=1, low tool calls, few errors."""

    def test_clean_success_basic(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "tool_calls_total": 10,
            "error_count": 0,
            "retry_count": 0,
        }
        results = annotate_trial(signals)
        assert "clean_success" in _names(results)

    def test_clean_success_not_on_failure(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "tool_calls_total": 10,
            "error_count": 0,
            "retry_count": 0,
        }
        results = annotate_trial(signals)
        assert "clean_success" not in _names(results)

    def test_clean_success_not_high_tool_calls(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "tool_calls_total": 50,
            "error_count": 0,
            "retry_count": 0,
        }
        results = annotate_trial(signals)
        assert "clean_success" not in _names(results)

    def test_clean_success_not_many_errors(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "tool_calls_total": 15,
            "error_count": 5,
            "retry_count": 0,
        }
        results = annotate_trial(signals)
        assert "clean_success" not in _names(results)


class TestPlanningAbsence:
    """AC: planning_absence when first tool calls are edits with no exploration."""

    def test_planning_absence_edit_first(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "tool_call_sequence": ["Edit", "Edit", "Bash", "Read"],
        }
        results = annotate_trial(signals)
        assert "planning_absence" in _names(results)

    def test_planning_absence_not_when_read_first(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "tool_call_sequence": ["Read", "Edit", "Edit", "Bash"],
        }
        results = annotate_trial(signals)
        assert "planning_absence" not in _names(results)

    def test_planning_absence_not_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "tool_call_sequence": ["Edit", "Edit", "Bash", "Read"],
        }
        results = annotate_trial(signals)
        assert "planning_absence" not in _names(results)

    def test_planning_absence_not_short_sequence(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "tool_call_sequence": ["Edit"],
        }
        results = annotate_trial(signals)
        assert "planning_absence" not in _names(results)


class TestPrematureCommit:
    """AC: premature_commit when edits exist but no verification after last edit."""

    def test_premature_commit_no_verify_after_edit(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 2,
            "tool_call_sequence": ["Read", "Grep", "Edit", "Edit"],
        }
        results = annotate_trial(signals)
        assert "premature_commit" in _names(results)

    def test_premature_commit_not_when_bash_after(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 2,
            "tool_call_sequence": ["Read", "Edit", "Edit", "Bash"],
        }
        results = annotate_trial(signals)
        assert "premature_commit" not in _names(results)

    def test_premature_commit_not_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "edit_tool_calls": 2,
            "tool_call_sequence": ["Read", "Edit", "Edit"],
        }
        results = annotate_trial(signals)
        assert "premature_commit" not in _names(results)


class TestVerificationSkip:
    """AC: verification_skip when exactly 1 verify call but multiple edits and failure."""

    def test_verification_skip_single_bash(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 3,
            "tool_call_sequence": ["Read", "Edit", "Edit", "Bash", "Edit"],
        }
        results = annotate_trial(signals)
        assert "verification_skip" in _names(results)

    def test_verification_skip_not_multiple_bash(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "edit_tool_calls": 3,
            "tool_call_sequence": ["Read", "Edit", "Bash", "Edit", "Bash"],
        }
        results = annotate_trial(signals)
        assert "verification_skip" not in _names(results)

    def test_verification_skip_not_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "edit_tool_calls": 3,
            "tool_call_sequence": ["Read", "Edit", "Edit", "Bash", "Edit"],
        }
        results = annotate_trial(signals)
        assert "verification_skip" not in _names(results)


class TestToolUnderutilization:
    """AC: tool_underutilization when many file reads but no nav/semantic tools."""

    def test_tool_underutilization_basic(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "unique_files_read": 12,
            "code_nav_tool_calls": 0,
            "semantic_search_tool_calls": 0,
            "search_tool_calls": 1,
        }
        results = annotate_trial(signals)
        assert "tool_underutilization" in _names(results)

    def test_tool_underutilization_not_with_code_nav(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "unique_files_read": 12,
            "code_nav_tool_calls": 2,
            "semantic_search_tool_calls": 0,
            "search_tool_calls": 1,
        }
        results = annotate_trial(signals)
        assert "tool_underutilization" not in _names(results)

    def test_tool_underutilization_not_few_reads(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "unique_files_read": 3,
            "code_nav_tool_calls": 0,
            "semantic_search_tool_calls": 0,
            "search_tool_calls": 0,
        }
        results = annotate_trial(signals)
        assert "tool_underutilization" not in _names(results)

    def test_tool_underutilization_not_on_success(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "unique_files_read": 12,
            "code_nav_tool_calls": 0,
            "semantic_search_tool_calls": 0,
            "search_tool_calls": 1,
        }
        results = annotate_trial(signals)
        assert "tool_underutilization" not in _names(results)


class TestRewardHacking:
    """AC: reward_hacking when test files dominate the edited files list."""

    def test_reward_hacking_mostly_test_edits(self) -> None:
        signals: TrialSignals = {
            "reward": 0.8,
            "passed": True,
            "files_edited_list": ["tests/test_foo.py", "tests/test_bar.py"],
        }
        results = annotate_trial(signals)
        assert "reward_hacking" in _names(results)

    def test_reward_hacking_not_zero_reward(self) -> None:
        signals: TrialSignals = {
            "reward": 0.0,
            "passed": False,
            "files_edited_list": ["tests/test_foo.py"],
        }
        results = annotate_trial(signals)
        assert "reward_hacking" not in _names(results)

    def test_reward_hacking_not_when_mostly_source(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "files_edited_list": [
                "src/main.py",
                "src/utils.py",
                "src/api.py",
                "tests/test_main.py",
            ],
        }
        results = annotate_trial(signals)
        assert "reward_hacking" not in _names(results)

    def test_reward_hacking_evidence_lists_files(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "files_edited_list": ["tests/test_foo.py", "tests/test_bar.py"],
        }
        results = annotate_trial(signals)
        rh = next(r for r in results if r.name == "reward_hacking")
        assert "test_foo" in rh.evidence


class TestDefaultRegistryKwarg:
    def test_default_registry_used(self) -> None:
        signals: TrialSignals = {
            "reward": 1.0,
            "passed": True,
            "code_nav_tool_calls": 3,
        }
        r1 = annotate_trial(signals)
        r2 = annotate_trial(signals, tool_registry=DEFAULT_REGISTRY)
        assert _names(r1) == _names(r2)
