"""Tests for the two-tier ensemble annotator."""

from __future__ import annotations

import ast
import importlib
from unittest.mock import patch

import pytest

from agent_diagnostics.ensemble import (
    HEURISTIC_ONLY,
    ensemble_all,
    ensemble_annotate,
)
from agent_diagnostics.types import CategoryAssignment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model(classifiers: dict[str, dict] | None = None) -> dict:
    """Build a minimal mock model dict."""
    if classifiers is None:
        classifiers = {
            "retrieval_failure": {
                "weights": [0.0] * 21,
                "bias": 0.0,
                "eval_f1": 0.8,
                "train_accuracy": 0.85,
            },
            "query_churn": {
                "weights": [0.0] * 21,
                "bias": 0.0,
                "eval_f1": 0.5,  # below default min_f1
                "train_accuracy": 0.6,
            },
        }
    return {
        "classifiers": classifiers,
        "means": [0.0] * 21,
        "stds": [1.0] * 21,
        "training_samples": 100,
    }


def _make_signals(**overrides: object) -> dict:
    """Build a minimal signals dict with sensible defaults."""
    base: dict = {
        "task_id": "test-task-1",
        "trial_path": "/tmp/trial",
        "reward": 0.0,
        "passed": False,
        "exception_crashed": False,
        "rate_limited": False,
        "tool_calls_total": 10,
        "search_tool_calls": 0,
        "edit_tool_calls": 0,
        "code_nav_tool_calls": 0,
        "unique_files_read": 0,
        "unique_files_edited": 0,
        "error_count": 0,
        "retry_count": 0,
        "trajectory_length": 10,
        "duration_seconds": 120.0,
        "patch_size_lines": 0,
        "tool_call_sequence": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: ensemble_annotate
# ---------------------------------------------------------------------------


class TestEnsembleAnnotate:
    """Tests for ensemble_annotate."""

    def test_returns_list_of_dicts_with_required_keys(self) -> None:
        signals = _make_signals(exception_crashed=True)
        model = _make_model()
        result = ensemble_annotate(signals, model)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "name" in item
            assert "confidence" in item
            assert "evidence" in item
            assert "source" in item

    def test_heuristic_only_categories_come_from_heuristic(self) -> None:
        """HEURISTIC_ONLY categories must have source='heuristic'."""
        signals = _make_signals(exception_crashed=True)
        model = _make_model()
        result = ensemble_annotate(signals, model)

        heuristic_results = [r for r in result if r["name"] in HEURISTIC_ONLY]
        for r in heuristic_results:
            assert r["source"] == "heuristic"

    def test_exception_crash_detected_via_heuristic(self) -> None:
        signals = _make_signals(exception_crashed=True)
        model = _make_model()
        result = ensemble_annotate(signals, model)

        names = [r["name"] for r in result]
        assert "exception_crash" in names

        crash_result = next(r for r in result if r["name"] == "exception_crash")
        assert crash_result["source"] == "heuristic"

    def test_rate_limited_detected_via_heuristic(self) -> None:
        signals = _make_signals(rate_limited=True)
        model = _make_model()
        result = ensemble_annotate(signals, model)

        names = [r["name"] for r in result]
        assert "rate_limited_run" in names

        rate_result = next(r for r in result if r["name"] == "rate_limited_run")
        assert rate_result["source"] == "heuristic"

    def test_classifier_categories_filtered_by_min_f1(self) -> None:
        """Classifier results below min_f1 threshold should be excluded."""
        mock_clf_results = [
            {
                "name": "retrieval_failure",
                "confidence": 0.9,
                "evidence": "clf prob=0.9",
            },
            {"name": "query_churn", "confidence": 0.8, "evidence": "clf prob=0.8"},
        ]
        mock_heur_results: list[CategoryAssignment] = []

        model = _make_model()  # retrieval_failure has eval_f1=0.8, query_churn has 0.5

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=mock_heur_results,
            ),
            patch(
                "agent_diagnostics.ensemble.predict_trial",
                return_value=mock_clf_results,
            ),
        ):
            result = ensemble_annotate(_make_signals(), model, classifier_min_f1=0.7)

        names = [r["name"] for r in result]
        assert "retrieval_failure" in names  # eval_f1=0.8 >= 0.7
        assert "query_churn" not in names  # eval_f1=0.5 < 0.7

    def test_classifier_skips_heuristic_only_categories(self) -> None:
        """Even if the classifier predicts a HEURISTIC_ONLY category, it should be ignored."""
        mock_clf_results = [
            {
                "name": "exception_crash",
                "confidence": 0.99,
                "evidence": "clf prob=0.99",
            },
        ]
        mock_heur_results: list[CategoryAssignment] = []

        model = _make_model(
            classifiers={
                "exception_crash": {
                    "weights": [0.0] * 21,
                    "bias": 0.0,
                    "eval_f1": 0.95,
                    "train_accuracy": 0.95,
                },
            }
        )

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=mock_heur_results,
            ),
            patch(
                "agent_diagnostics.ensemble.predict_trial",
                return_value=mock_clf_results,
            ),
        ):
            result = ensemble_annotate(_make_signals(), model)

        # exception_crash from classifier should be skipped (it's HEURISTIC_ONLY)
        names = [r["name"] for r in result]
        assert "exception_crash" not in names

    def test_classifier_result_includes_eval_f1_in_evidence(self) -> None:
        """Classifier evidence should include eval_f1 annotation."""
        mock_clf_results = [
            {
                "name": "retrieval_failure",
                "confidence": 0.85,
                "evidence": "clf prob=0.850",
            },
        ]
        mock_heur_results: list[CategoryAssignment] = []
        model = _make_model()

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=mock_heur_results,
            ),
            patch(
                "agent_diagnostics.ensemble.predict_trial",
                return_value=mock_clf_results,
            ),
        ):
            result = ensemble_annotate(_make_signals(), model)

        clf_result = next(r for r in result if r["name"] == "retrieval_failure")
        assert "[eval_f1=0.80]" in clf_result["evidence"]
        assert clf_result["source"] == "classifier"

    def test_empty_result_when_no_matches(self) -> None:
        """No heuristic or classifier matches should return empty list."""
        mock_heur_results: list[CategoryAssignment] = []
        mock_clf_results: list[dict] = []

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=mock_heur_results,
            ),
            patch(
                "agent_diagnostics.ensemble.predict_trial",
                return_value=mock_clf_results,
            ),
        ):
            result = ensemble_annotate(_make_signals(), _make_model())

        assert result == []


# ---------------------------------------------------------------------------
# Tests: HEURISTIC_ONLY
# ---------------------------------------------------------------------------


class TestHeuristicOnly:
    """Tests for the HEURISTIC_ONLY frozenset."""

    def test_is_frozenset(self) -> None:
        assert isinstance(HEURISTIC_ONLY, frozenset)

    def test_contains_structural_categories(self) -> None:
        expected = {
            "exception_crash",
            "rate_limited_run",
            "edit_verify_loop_failure",
        }
        assert HEURISTIC_ONLY == expected

    def test_excludes_derived_signal_categories(self) -> None:
        """near_miss and minimal_progress are derived_from_signal and must not be in HEURISTIC_ONLY."""
        assert "near_miss" not in HEURISTIC_ONLY
        assert "minimal_progress" not in HEURISTIC_ONLY


# ---------------------------------------------------------------------------
# Tests: ensemble_all
# ---------------------------------------------------------------------------


class TestEnsembleAll:
    """Tests for ensemble_all."""

    def test_produces_annotation_document(self) -> None:
        signals = _make_signals(exception_crashed=True)
        model = _make_model()

        result = ensemble_all([signals], model)

        assert isinstance(result, dict)
        assert result["schema_version"] == "observatory-annotation-v1"
        assert "taxonomy_version" in result
        assert "generated_at" in result
        assert "annotator" in result
        assert result["annotator"]["type"] == "ensemble"
        assert "annotations" in result
        assert isinstance(result["annotations"], list)

    def test_annotations_have_correct_structure(self) -> None:
        signals = _make_signals(exception_crashed=True)
        model = _make_model()

        result = ensemble_all([signals], model)
        annotations = result["annotations"]

        assert len(annotations) >= 1
        ann = annotations[0]
        assert "task_id" in ann
        assert "trial_path" in ann
        assert "reward" in ann
        assert "passed" in ann
        assert "categories" in ann
        assert "annotated_at" in ann

    def test_source_key_stripped_from_categories(self) -> None:
        """The 'source' key should not appear in output categories."""
        signals = _make_signals(exception_crashed=True)
        model = _make_model()

        result = ensemble_all([signals], model)
        for ann in result["annotations"]:
            for cat in ann["categories"]:
                assert "source" not in cat

    def test_empty_signals_list_returns_empty_annotations(self) -> None:
        model = _make_model()
        result = ensemble_all([], model)

        assert result["annotations"] == []

    def test_none_reward_preserved_in_output(self) -> None:
        """When reward is None, ensemble_all should not crash and handle it."""
        signals = _make_signals(reward=None, passed=False)
        model = _make_model()

        result = ensemble_all([signals], model)
        # Should produce annotations without error
        assert isinstance(result, dict)
        for ann in result["annotations"]:
            # reward should be 0.0 (coerced for output) since None is not JSON-friendly
            assert "reward" in ann

    def test_annotator_identity_includes_tier_info(self) -> None:
        signals = _make_signals(exception_crashed=True)
        model = _make_model()

        result = ensemble_all([signals], model)
        identity = result["annotator"]["identity"]

        assert "heuristic(" in identity
        assert "classifier(" in identity


# ---------------------------------------------------------------------------
# Tests: No CSB imports
# ---------------------------------------------------------------------------


class TestNoCsbImports:
    """Verify ensemble.py does not import from observatory (CSB)."""

    def test_no_observatory_imports(self) -> None:
        """Parse ensemble.py AST and verify no imports from 'observatory'."""
        import agent_diagnostics.ensemble as mod

        source_path = mod.__file__
        assert source_path is not None

        with open(source_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(
                    "observatory."
                ), f"Found CSB import: from {module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "observatory."
                    ), f"Found CSB import: import {alias.name}"
