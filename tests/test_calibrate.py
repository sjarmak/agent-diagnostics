"""Tests for agent_observatory.calibrate — agreement and calibration analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_observatory.calibrate import (
    cohen_kappa,
    compare_annotations,
    compare_cross_model,
    format_cross_model_markdown,
    format_markdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_annotations(path: Path, annotations: list[dict]) -> Path:
    """Write an annotation JSON file (raw list format)."""
    path.write_text(json.dumps(annotations))
    return path


def _write_annotation_doc(path: Path, annotations: list[dict]) -> Path:
    """Write an annotation JSON file (document format with 'annotations' key)."""
    path.write_text(json.dumps({"annotations": annotations}))
    return path


# ---------------------------------------------------------------------------
# compare_annotations
# ---------------------------------------------------------------------------


class TestCompareAnnotations:
    """Tests for compare_annotations with known TP/FP/FN scenarios."""

    def test_perfect_agreement(self, tmp_path: Path) -> None:
        """Both annotators assign the same categories to the same trials."""
        anns = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}, {"name": "cat_b"}]},
            {"trial_path": "t2", "categories": [{"name": "cat_a"}]},
        ]
        file_a = _write_annotations(tmp_path / "a.json", anns)
        file_b = _write_annotations(tmp_path / "b.json", anns)

        result = compare_annotations(file_a, file_b)
        assert result["shared_trials"] == 2
        for cat_metrics in result["categories"].values():
            assert cat_metrics["true_positive"] > 0
            assert cat_metrics["false_positive"] == 0
            assert cat_metrics["false_negative"] == 0
            assert cat_metrics["precision"] == 1.0
            assert cat_metrics["recall"] == 1.0
            assert cat_metrics["f1"] == 1.0

    def test_known_tp_fp_fn(self, tmp_path: Path) -> None:
        """Explicit TP/FP/FN scenario for a single category."""
        # t1: both have cat_a -> TP
        # t2: only heuristic has cat_a -> FP
        # t3: only llm has cat_a -> FN
        heuristic = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t3", "categories": []},
        ]
        llm = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": []},
            {"trial_path": "t3", "categories": [{"name": "cat_a"}]},
        ]
        file_h = _write_annotations(tmp_path / "h.json", heuristic)
        file_l = _write_annotations(tmp_path / "l.json", llm)

        result = compare_annotations(file_h, file_l)
        cat_a = result["categories"]["cat_a"]
        assert cat_a["true_positive"] == 1
        assert cat_a["false_positive"] == 1
        assert cat_a["false_negative"] == 1
        assert cat_a["precision"] == 0.5
        assert cat_a["recall"] == 0.5

    def test_no_shared_trials(self, tmp_path: Path) -> None:
        """Disjoint trial paths yield zero shared trials."""
        file_a = _write_annotations(
            tmp_path / "a.json",
            [{"trial_path": "t1", "categories": [{"name": "c"}]}],
        )
        file_b = _write_annotations(
            tmp_path / "b.json",
            [{"trial_path": "t2", "categories": [{"name": "c"}]}],
        )
        result = compare_annotations(file_a, file_b)
        assert result["shared_trials"] == 0
        assert result["categories"] == {}
        assert result["macro_avg"]["f1"] == 0.0

    def test_empty_annotations(self, tmp_path: Path) -> None:
        """Empty annotation lists produce zero shared trials."""
        file_a = _write_annotations(tmp_path / "a.json", [])
        file_b = _write_annotations(tmp_path / "b.json", [])
        result = compare_annotations(file_a, file_b)
        assert result["shared_trials"] == 0
        assert result["categories"] == {}

    def test_document_format(self, tmp_path: Path) -> None:
        """Annotation document format with top-level 'annotations' key."""
        anns = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
        ]
        file_a = _write_annotation_doc(tmp_path / "a.json", anns)
        file_b = _write_annotation_doc(tmp_path / "b.json", anns)

        result = compare_annotations(file_a, file_b)
        assert result["shared_trials"] == 1
        assert "cat_a" in result["categories"]

    def test_macro_avg_multiple_categories(self, tmp_path: Path) -> None:
        """Macro average computed across multiple categories."""
        heuristic = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}, {"name": "cat_b"}]},
        ]
        llm = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
        ]
        file_h = _write_annotations(tmp_path / "h.json", heuristic)
        file_l = _write_annotations(tmp_path / "l.json", llm)

        result = compare_annotations(file_h, file_l)
        assert "macro_avg" in result
        assert "precision" in result["macro_avg"]
        assert "recall" in result["macro_avg"]
        assert "f1" in result["macro_avg"]


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------


class TestCohenKappa:
    """Tests for cohen_kappa with known agreement scenarios."""

    def test_perfect_agreement(self) -> None:
        """Identical non-trivial labels yield kappa = 1.0."""
        a = [1, 0, 1, 0, 1, 0, 1, 0]
        b = [1, 0, 1, 0, 1, 0, 1, 0]
        assert cohen_kappa(a, b) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        """Completely opposite labels yield negative kappa."""
        a = [1, 0, 1, 0]
        b = [0, 1, 0, 1]
        kappa = cohen_kappa(a, b)
        assert kappa < 0.0

    def test_random_agreement(self) -> None:
        """Random-ish agreement yields kappa near 0."""
        # 50/50 split with half agreement -> kappa ~ 0
        a = [1, 1, 0, 0]
        b = [1, 0, 1, 0]
        kappa = cohen_kappa(a, b)
        assert kappa == pytest.approx(0.0, abs=0.01)

    def test_all_same_label(self) -> None:
        """Both raters assign same label to all items -> kappa 0.0 (degenerate)."""
        a = [1, 1, 1, 1]
        b = [1, 1, 1, 1]
        assert cohen_kappa(a, b) == 0.0

    def test_length_mismatch_raises(self) -> None:
        """Mismatched vector lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            cohen_kappa([1, 0], [1])

    def test_empty_raises(self) -> None:
        """Empty vectors raise ValueError."""
        with pytest.raises(ValueError, match="not be empty"):
            cohen_kappa([], [])


# ---------------------------------------------------------------------------
# compare_cross_model
# ---------------------------------------------------------------------------


class TestCompareCrossModel:
    """Tests for compare_cross_model with known categories."""

    def test_basic_cross_model(self, tmp_path: Path) -> None:
        """Cross-model comparison with known agreement/disagreement."""
        model_a = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": [{"name": "cat_a"}, {"name": "cat_b"}]},
            {"trial_path": "t3", "categories": [{"name": "cat_b"}]},
        ]
        model_b = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t3", "categories": [{"name": "cat_b"}]},
        ]
        file_a = _write_annotations(tmp_path / "a.json", model_a)
        file_b = _write_annotations(tmp_path / "b.json", model_b)

        result = compare_cross_model(file_a, file_b)
        assert result["shared_trials"] == 3
        assert "cat_a" in result["categories"]
        assert "cat_b" in result["categories"]
        assert "kappa" in result["categories"]["cat_a"]
        assert "calibrated" in result["categories"]["cat_a"]
        assert isinstance(result["uncalibrated_categories"], list)
        assert "macro_kappa" in result

    def test_no_shared_trials(self, tmp_path: Path) -> None:
        """Disjoint trials yield empty result."""
        file_a = _write_annotations(
            tmp_path / "a.json",
            [{"trial_path": "t1", "categories": [{"name": "c"}]}],
        )
        file_b = _write_annotations(
            tmp_path / "b.json",
            [{"trial_path": "t2", "categories": [{"name": "c"}]}],
        )
        result = compare_cross_model(file_a, file_b)
        assert result["shared_trials"] == 0
        assert result["categories"] == {}
        assert result["uncalibrated_categories"] == []
        assert result["macro_kappa"] == 0.0

    def test_uncalibrated_detection(self, tmp_path: Path) -> None:
        """Categories with low kappa are flagged as uncalibrated."""
        # cat_a: complete disagreement across many trials
        model_a = [
            {
                "trial_path": f"t{i}",
                "categories": [{"name": "cat_a"}] if i % 2 == 0 else [],
            }
            for i in range(10)
        ]
        model_b = [
            {
                "trial_path": f"t{i}",
                "categories": [{"name": "cat_a"}] if i % 2 == 1 else [],
            }
            for i in range(10)
        ]
        file_a = _write_annotations(tmp_path / "a.json", model_a)
        file_b = _write_annotations(tmp_path / "b.json", model_b)

        result = compare_cross_model(file_a, file_b)
        assert "cat_a" in result["uncalibrated_categories"]
        assert result["categories"]["cat_a"]["calibrated"] is False

    def test_empty_files(self, tmp_path: Path) -> None:
        """Empty annotation files produce zero shared trials."""
        file_a = _write_annotations(tmp_path / "a.json", [])
        file_b = _write_annotations(tmp_path / "b.json", [])
        result = compare_cross_model(file_a, file_b)
        assert result["shared_trials"] == 0


# ---------------------------------------------------------------------------
# format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    """Tests for format_markdown producing valid markdown strings."""

    def test_produces_markdown_string(self, tmp_path: Path) -> None:
        """format_markdown returns a non-empty string with markdown table markers."""
        anns = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
        ]
        file_a = _write_annotations(tmp_path / "a.json", anns)
        file_b = _write_annotations(tmp_path / "b.json", anns)
        summary = compare_annotations(file_a, file_b)

        md = format_markdown(summary)
        assert isinstance(md, str)
        assert len(md) > 0
        assert "## Heuristic vs LLM Agreement" in md
        assert "| Category |" in md
        assert "Shared trials:" in md
        assert "Macro avg" in md

    def test_empty_summary(self) -> None:
        """format_markdown handles empty categories."""
        summary = {
            "shared_trials": 0,
            "categories": {},
            "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
        md = format_markdown(summary)
        assert isinstance(md, str)
        assert "Shared trials: 0" in md


# ---------------------------------------------------------------------------
# format_cross_model_markdown
# ---------------------------------------------------------------------------


class TestFormatCrossModelMarkdown:
    """Tests for format_cross_model_markdown producing valid markdown strings."""

    def test_produces_markdown_string(self, tmp_path: Path) -> None:
        """format_cross_model_markdown returns valid markdown."""
        model_a = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": [{"name": "cat_a"}]},
        ]
        model_b = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": []},
        ]
        file_a = _write_annotations(tmp_path / "a.json", model_a)
        file_b = _write_annotations(tmp_path / "b.json", model_b)

        summary = compare_cross_model(file_a, file_b)
        md = format_cross_model_markdown(summary, "GPT-4", "Claude")

        assert isinstance(md, str)
        assert "## Cross-Model Calibration: GPT-4 vs Claude" in md
        assert "Shared trials:" in md
        assert "Macro kappa:" in md
        assert "| Category |" in md

    def test_empty_summary(self) -> None:
        """format_cross_model_markdown handles empty result."""
        summary = {
            "shared_trials": 0,
            "categories": {},
            "uncalibrated_categories": [],
            "macro_kappa": 0.0,
        }
        md = format_cross_model_markdown(summary)
        assert isinstance(md, str)
        assert "Shared trials: 0" in md

    def test_uncalibrated_listed(self, tmp_path: Path) -> None:
        """Uncalibrated categories appear in the markdown output."""
        # Construct a summary with uncalibrated categories directly
        summary = {
            "shared_trials": 5,
            "categories": {
                "bad_cat": {
                    "kappa": 0.1,
                    "agreement": 0.5,
                    "a_count": 3,
                    "b_count": 2,
                    "calibrated": False,
                },
            },
            "uncalibrated_categories": ["bad_cat"],
            "macro_kappa": 0.1,
        }
        md = format_cross_model_markdown(summary)
        assert "Uncalibrated categories" in md
        assert "bad_cat" in md
