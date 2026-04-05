"""Tests for agent_diagnostics.classifier module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_diagnostics.classifier import (
    FEATURE_NAMES,
    evaluate,
    load_model,
    predict_all,
    predict_trial,
    save_model,
    signals_to_features,
    train,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic data generation
# ---------------------------------------------------------------------------


def _make_signals(trial_path: str, reward: float, passed: bool) -> dict:
    """Create a minimal signals dict with a trial_path for alignment."""
    return {
        "trial_path": trial_path,
        "task_id": "task-1",
        "reward": reward,
        "passed": passed,
        "exception_crashed": not passed,
        "tool_calls_total": 10,
        "trajectory_length": 5,
        "duration_seconds": 30.0,
        "has_trajectory": True,
    }


def _make_llm_file(tmp_path: Path, annotations: list[dict]) -> Path:
    """Write a synthetic LLM annotations JSON file."""
    path = tmp_path / "llm_labels.json"
    path.write_text(json.dumps({"annotations": annotations}))
    return path


def _make_signals_file(tmp_path: Path, signals: list[dict]) -> Path:
    """Write a synthetic signals JSON file."""
    path = tmp_path / "signals.json"
    path.write_text(json.dumps(signals))
    return path


def _build_synthetic_dataset(tmp_path: Path, n: int = 20) -> tuple[Path, Path]:
    """Build synthetic LLM labels and signals files with enough data to train.

    Creates two categories: 'retrieval_failure' (for failed trials) and
    'success_via_code_nav' (for passed trials), each with >= 3 positive
    examples to meet min_positive default.
    """
    signals_list: list[dict] = []
    annotations: list[dict] = []

    for i in range(n):
        passed = i % 2 == 0
        trial_path = f"/trials/trial_{i}"
        sig = _make_signals(trial_path, reward=1.0 if passed else 0.0, passed=passed)
        signals_list.append(sig)

        cats: list[dict] = []
        if not passed:
            cats.append({"name": "retrieval_failure", "confidence": 0.9})
        else:
            cats.append({"name": "success_via_code_nav", "confidence": 0.85})

        annotations.append(
            {
                "trial_path": trial_path,
                "task_id": f"task-{i}",
                "categories": cats,
            }
        )

    llm_path = _make_llm_file(tmp_path, annotations)
    sig_path = _make_signals_file(tmp_path, signals_list)
    return llm_path, sig_path


# ---------------------------------------------------------------------------
# Tests — signals_to_features
# ---------------------------------------------------------------------------


class TestSignalsToFeatures:
    def test_returns_list_of_floats(self) -> None:
        signals = {"reward": 1.0, "passed": True, "exception_crashed": False}
        result = signals_to_features(signals)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_length_matches_feature_names(self) -> None:
        result = signals_to_features({})
        assert len(result) == len(FEATURE_NAMES)

    def test_missing_keys_default_to_zero(self) -> None:
        result = signals_to_features({})
        assert all(v == 0.0 for v in result)

    def test_bool_converted_to_float(self) -> None:
        signals = {"passed": True, "exception_crashed": False}
        result = signals_to_features(signals)
        passed_idx = FEATURE_NAMES.index("passed")
        exc_idx = FEATURE_NAMES.index("exception_crashed")
        assert result[passed_idx] == 1.0
        assert result[exc_idx] == 0.0

    def test_none_values_become_zero(self) -> None:
        signals = {"reward": None, "passed": None}
        result = signals_to_features(signals)
        assert result[FEATURE_NAMES.index("reward")] == 0.0


# ---------------------------------------------------------------------------
# Tests — FEATURE_NAMES
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_is_list_of_strings(self) -> None:
        assert isinstance(FEATURE_NAMES, list)
        assert all(isinstance(name, str) for name in FEATURE_NAMES)

    def test_contains_mapped_keys(self) -> None:
        assert "exception_crashed" in FEATURE_NAMES
        assert "duration_seconds" in FEATURE_NAMES
        assert "trajectory_length" in FEATURE_NAMES

    def test_does_not_contain_old_csb_keys(self) -> None:
        assert "has_exception" not in FEATURE_NAMES
        assert "wall_clock_seconds" not in FEATURE_NAMES
        assert "trajectory_steps" not in FEATURE_NAMES


# ---------------------------------------------------------------------------
# Tests — train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_returns_model_dict(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        assert model["schema_version"] == "observatory-classifier-v1"
        assert model["feature_names"] == FEATURE_NAMES
        assert isinstance(model["means"], list)
        assert isinstance(model["stds"], list)
        assert isinstance(model["classifiers"], dict)
        assert isinstance(model["skipped_categories"], list)

    def test_train_has_expected_categories(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)
        assert "retrieval_failure" in model["classifiers"]
        assert "success_via_code_nav" in model["classifiers"]

    def test_train_no_match_raises(self, tmp_path: Path) -> None:
        llm_path = _make_llm_file(
            tmp_path,
            [{"trial_path": "/no/match", "categories": [{"name": "x"}]}],
        )
        sig_path = _make_signals_file(
            tmp_path, [{"trial_path": "/other/path", "reward": 0.5}]
        )
        with pytest.raises(ValueError, match="No trials matched"):
            train(llm_path, sig_path)


# ---------------------------------------------------------------------------
# Tests — save_model / load_model round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadModel:
    def test_round_trip(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        model_path = tmp_path / "model.json"
        save_model(model, model_path)
        loaded = load_model(model_path)

        assert loaded["schema_version"] == model["schema_version"]
        assert loaded["feature_names"] == model["feature_names"]
        assert loaded["means"] == model["means"]
        assert loaded["stds"] == model["stds"]
        assert set(loaded["classifiers"].keys()) == set(model["classifiers"].keys())
        assert loaded["skipped_categories"] == model["skipped_categories"]


# ---------------------------------------------------------------------------
# Tests — predict_trial
# ---------------------------------------------------------------------------


class TestPredictTrial:
    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        signals = _make_signals("/test", reward=0.0, passed=False)
        results = predict_trial(signals, model, threshold=0.3)

        assert isinstance(results, list)
        for item in results:
            assert "name" in item
            assert "confidence" in item
            assert "evidence" in item
            assert isinstance(item["name"], str)
            assert isinstance(item["confidence"], float)
            assert isinstance(item["evidence"], str)

    def test_threshold_filters(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        signals = _make_signals("/test", reward=0.0, passed=False)
        high_threshold = predict_trial(signals, model, threshold=0.99)
        low_threshold = predict_trial(signals, model, threshold=0.01)

        assert len(low_threshold) >= len(high_threshold)


# ---------------------------------------------------------------------------
# Tests — evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_per_category_metrics(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        results = evaluate(model, llm_path, sig_path)

        assert isinstance(results, dict)
        for cat, metrics in results.items():
            if metrics.get("status") == "no_classifier":
                assert "positive_in_eval" in metrics
            else:
                assert "precision" in metrics
                assert "recall" in metrics
                assert "f1" in metrics
                assert "tp" in metrics
                assert "fp" in metrics
                assert "fn" in metrics
                assert "tn" in metrics


# ---------------------------------------------------------------------------
# Tests — no forbidden imports
# ---------------------------------------------------------------------------


class TestNoForbiddenImports:
    def test_no_numpy_import(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "agent_diagnostics"
            / "classifier.py"
        )
        content = source.read_text()
        assert "import numpy" not in content
        assert "from numpy" not in content

    def test_no_sklearn_import(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "agent_diagnostics"
            / "classifier.py"
        )
        content = source.read_text()
        assert "import sklearn" not in content
        assert "from sklearn" not in content

    def test_no_csb_imports(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "agent_diagnostics"
            / "classifier.py"
        )
        content = source.read_text()
        assert "from observatory." not in content
        assert "from csb" not in content


# ---------------------------------------------------------------------------
# Tests — import check (acceptance criteria)
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_public_names_importable(self) -> None:
        from agent_diagnostics.classifier import (  # noqa: F401
            FEATURE_NAMES,
            evaluate,
            load_model,
            predict_all,
            predict_trial,
            save_model,
            signals_to_features,
            train,
        )
