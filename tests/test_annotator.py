"""Tests for the heuristic annotator."""

from __future__ import annotations

import pytest

from agent_observatory.annotator import annotate_trial
from agent_observatory.taxonomy import valid_category_names
from agent_observatory.tool_registry import DEFAULT_REGISTRY, ToolRegistry
from agent_observatory.types import CategoryAssignment, TrialSignals

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
        from agent_observatory.annotator import annotate_trial as fn

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
        valid = valid_category_names()
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
