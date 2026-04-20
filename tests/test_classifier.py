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

    def test_none_reward_converted_to_zero_in_features(self) -> None:
        """Nullable reward from extract_signals should become 0.0 in feature vector."""
        signals = {"reward": None, "passed": False, "has_verifier_result": False}
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
        sig_path = _make_signals_file(tmp_path, [{"trial_path": "/other/path", "reward": 0.5}])
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
            Path(__file__).resolve().parent.parent / "src" / "agent_diagnostics" / "classifier.py"
        )
        content = source.read_text()
        assert "import numpy" not in content
        assert "from numpy" not in content

    def test_no_sklearn_import(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent / "src" / "agent_diagnostics" / "classifier.py"
        )
        content = source.read_text()
        assert "import sklearn" not in content
        assert "from sklearn" not in content

    def test_no_csb_imports(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent / "src" / "agent_diagnostics" / "classifier.py"
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


# ---------------------------------------------------------------------------
# Tests — _to_float edge cases
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_unconvertible_string_returns_zero(self) -> None:
        from agent_diagnostics.classifier import _to_float

        assert _to_float("not_a_number") == 0.0

    def test_none_returns_zero(self) -> None:
        from agent_diagnostics.classifier import _to_float

        assert _to_float(None) == 0.0

    def test_bool_true(self) -> None:
        from agent_diagnostics.classifier import _to_float

        assert _to_float(True) == 1.0

    def test_bool_false(self) -> None:
        from agent_diagnostics.classifier import _to_float

        assert _to_float(False) == 0.0

    def test_numeric_string(self) -> None:
        from agent_diagnostics.classifier import _to_float

        assert _to_float("3.14") == pytest.approx(3.14)

    def test_unconvertible_object_returns_zero(self) -> None:
        from agent_diagnostics.classifier import _to_float

        assert _to_float(object()) == 0.0


# ---------------------------------------------------------------------------
# Tests — _scale edge cases
# ---------------------------------------------------------------------------


class TestScale:
    def test_empty_matrix_returns_defaults(self) -> None:
        from agent_diagnostics.classifier import FEATURE_NAMES, _scale

        means, stds = _scale([])
        assert len(means) == len(FEATURE_NAMES)
        assert len(stds) == len(FEATURE_NAMES)
        assert all(m == 0.0 for m in means)
        assert all(s == 1.0 for s in stds)


# ---------------------------------------------------------------------------
# Tests — _train_binary_lr degenerate inputs
# ---------------------------------------------------------------------------


class TestTrainBinaryLR:
    def test_all_same_labels_positive(self) -> None:
        from agent_diagnostics.classifier import _scale, _train_binary_lr

        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [1, 1, 1]  # all positive
        means, stds = _scale(X)
        w, b = _train_binary_lr(X, y, means, stds, epochs=10)
        assert isinstance(w, list)
        assert isinstance(b, float)
        assert len(w) == 2

    def test_all_same_labels_negative(self) -> None:
        from agent_diagnostics.classifier import _scale, _train_binary_lr

        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [0, 0, 0]  # all negative
        means, stds = _scale(X)
        w, b = _train_binary_lr(X, y, means, stds, epochs=10)
        assert isinstance(w, list)
        assert isinstance(b, float)

    def test_zero_variance_features(self) -> None:
        from agent_diagnostics.classifier import _scale, _train_binary_lr

        # All rows identical — zero variance features
        X = [[1.0, 1.0]] * 6
        y = [1, 0, 1, 0, 1, 0]
        means, stds = _scale(X)
        w, b = _train_binary_lr(X, y, means, stds, epochs=10)
        assert isinstance(w, list)
        assert isinstance(b, float)


# ---------------------------------------------------------------------------
# Tests — predict_all with trained model
# ---------------------------------------------------------------------------


class TestPredictAll:
    def test_returns_annotation_document(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        signals_list = [
            _make_signals("/test/0", reward=0.0, passed=False),
            _make_signals("/test/1", reward=1.0, passed=True),
        ]

        fake_taxonomy = {"version": "2.0", "dimensions": []}
        with patch("agent_diagnostics.taxonomy.load_taxonomy", return_value=fake_taxonomy):
            result = predict_all(signals_list, model, threshold=0.01)

        assert result["schema_version"] == "observatory-annotation-v1"
        assert "taxonomy_version" in result
        assert "generated_at" in result
        assert result["annotator"]["type"] == "classifier"
        assert isinstance(result["annotations"], list)

    def test_annotations_have_expected_fields(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        signals_list = [_make_signals("/test/0", reward=0.0, passed=False)]

        fake_taxonomy = {"version": "2.0", "dimensions": []}
        with patch("agent_diagnostics.taxonomy.load_taxonomy", return_value=fake_taxonomy):
            result = predict_all(signals_list, model, threshold=0.01)

        for ann in result["annotations"]:
            assert "task_id" in ann
            assert "trial_path" in ann
            assert "categories" in ann
            assert "annotated_at" in ann
            assert "reward" in ann
            assert "passed" in ann


# ---------------------------------------------------------------------------
# Tests — format_eval_markdown
# ---------------------------------------------------------------------------


class TestFormatEvalMarkdown:
    def test_output_contains_header_and_table(self, tmp_path: Path) -> None:
        from agent_diagnostics.classifier import format_eval_markdown

        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)
        eval_results = evaluate(model, llm_path, sig_path)

        md = format_eval_markdown(eval_results, model)
        assert "## Classifier Evaluation" in md
        assert "| Category |" in md
        assert f"Training samples: {model['training_samples']}" in md
        assert f"Categories trained: {len(model['classifiers'])}" in md

    def test_no_classifier_row(self) -> None:
        from agent_diagnostics.classifier import format_eval_markdown

        model = {
            "training_samples": 10,
            "classifiers": {},
            "skipped_categories": ["cat_a"],
            "min_positive": 3,
        }
        eval_results = {
            "missing_cat": {
                "status": "no_classifier",
                "positive_in_eval": 5,
            }
        }
        md = format_eval_markdown(eval_results, model)
        assert "no clf" in md
        assert "missing_cat" in md

    def test_skipped_categories_displayed(self) -> None:
        from agent_diagnostics.classifier import format_eval_markdown

        model = {
            "training_samples": 10,
            "classifiers": {"cat_b": {"train_accuracy": 0.95}},
            "skipped_categories": ["cat_a", "cat_c"],
            "min_positive": 3,
        }
        eval_results = {
            "cat_b": {
                "tp": 5,
                "fp": 1,
                "fn": 1,
                "tn": 3,
                "precision": 0.83,
                "recall": 0.83,
                "f1": 0.83,
            },
        }
        md = format_eval_markdown(eval_results, model)
        assert "cat_a, cat_c" in md


# ---------------------------------------------------------------------------
# Tests — evaluate with no_classifier path
# ---------------------------------------------------------------------------


class TestEvaluateNoClassifier:
    def test_category_not_in_model_returns_no_classifier(self, tmp_path: Path) -> None:
        llm_path, sig_path = _build_synthetic_dataset(tmp_path)
        model = train(llm_path, sig_path, min_positive=3)

        # Remove one classifier from the model to trigger no_classifier path
        removed_cat = list(model["classifiers"].keys())[0]
        del model["classifiers"][removed_cat]

        results = evaluate(model, llm_path, sig_path)
        assert results[removed_cat]["status"] == "no_classifier"
        assert "positive_in_eval" in results[removed_cat]


# ---------------------------------------------------------------------------
# Tests — train with invalid LLM file
# ---------------------------------------------------------------------------


class TestTrainValidation:
    def test_non_list_annotations_raises(self, tmp_path: Path) -> None:
        llm_path = tmp_path / "bad_llm.json"
        llm_path.write_text(json.dumps({"annotations": "not_a_list"}))
        sig_path = _make_signals_file(tmp_path, [{"trial_path": "/t", "reward": 1.0}])

        with pytest.raises(ValueError, match="Expected annotations list"):
            train(llm_path, sig_path)


# ---------------------------------------------------------------------------
# Tests — derived categories excluded from training
# ---------------------------------------------------------------------------


class TestDerivedCategoriesExcluded:
    """train() must skip categories where taxonomy marks derived_from_signal: true."""

    def test_derived_categories_excluded(self, tmp_path: Path) -> None:
        """incomplete_solution, near_miss, minimal_progress must not appear in classifiers."""
        derived_cats = {"incomplete_solution", "near_miss", "minimal_progress"}
        n = 20

        signals_list: list[dict] = []
        annotations: list[dict] = []

        for i in range(n):
            trial_path = f"/trials/trial_{i}"
            reward = (i % 5) * 0.25  # 0.0, 0.25, 0.5, 0.75, 1.0
            passed = reward >= 1.0
            sig = _make_signals(trial_path, reward=reward, passed=passed)
            signals_list.append(sig)

            cats: list[dict] = []
            # Always assign a non-derived category so training data exists
            cats.append({"name": "retrieval_failure", "confidence": 0.9})
            # Also assign derived categories to many trials
            if 0 < reward < 1.0:
                cats.append({"name": "incomplete_solution", "confidence": 0.8})
            if reward >= 0.5 and reward < 1.0:
                cats.append({"name": "near_miss", "confidence": 0.85})
            if 0 < reward < 0.5:
                cats.append({"name": "minimal_progress", "confidence": 0.7})

            annotations.append(
                {"trial_path": trial_path, "task_id": f"task-{i}", "categories": cats}
            )

        llm_path = _make_llm_file(tmp_path, annotations)
        sig_path = _make_signals_file(tmp_path, signals_list)

        model = train(llm_path, sig_path, min_positive=2)

        # None of the derived categories should have classifiers
        for cat in derived_cats:
            assert cat not in model["classifiers"], (
                f"Derived category {cat!r} should be excluded from training"
            )

        # They should appear in skipped_categories
        for cat in derived_cats:
            assert cat in model["skipped_categories"], (
                f"Derived category {cat!r} should be in skipped_categories"
            )

        # Non-derived category should still be trained
        assert "retrieval_failure" in model["classifiers"]
