"""Synthetic unit tests for the calibration scoring rules in ``calibrate.py``.

Covers:

- :func:`compute_ece` on perfectly-calibrated, overconfident, and underconfident
  predictions, plus n_bins behaviour.
- :func:`compute_brier` on perfectly-confident correct/incorrect predictions.
- :func:`reliability_diagram` structure, bin counts, and JSON-serializability.
- Degenerate inputs (empty, single-class, out-of-range confidences).
- The per-category ``ece``/``brier``/``reliability_bins`` fields added to
  :func:`compare_annotations` output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_diagnostics.calibrate import (
    compare_annotations,
    compute_brier,
    compute_ece,
    format_markdown,
    reliability_diagram,
)

# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


class TestComputeECE:
    """Synthetic ECE fixtures with known closed-form expected values."""

    def test_perfectly_calibrated_bins_yield_zero(self) -> None:
        """Confidence 0.1 correct 10% of the time, 0.9 correct 90% of the time, etc.

        Each bin's mean confidence equals its observed accuracy -> ECE == 0.0.
        """
        # For each of conf ∈ {0.1, 0.2, …, 0.9}, place 100 samples at that
        # exact confidence with round(conf * 100) of them observed correct,
        # so the bin's mean_confidence equals its accuracy and ECE vanishes.
        pairs: list[tuple[float, int]] = []
        for center_tenths in range(1, 10):  # 0.1 through 0.9
            conf = center_tenths / 10.0
            n = 100
            n_correct = round(conf * n)
            pairs.extend((conf, 1) for _ in range(n_correct))
            pairs.extend((conf, 0) for _ in range(n - n_correct))
        ece = compute_ece(pairs, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_overconfident_all_99_half_correct(self) -> None:
        """All predictions at 0.99, exactly half observed correct -> ECE ≈ 0.49."""
        n = 200
        pairs = [(0.99, 1 if i % 2 == 0 else 0) for i in range(n)]
        ece = compute_ece(pairs, n_bins=10)
        # Single populated bin: mean_conf = 0.99, acc = 0.5, |diff| = 0.49,
        # weight = 1.0 -> ECE = 0.49
        assert ece == pytest.approx(0.49, abs=1e-6)

    def test_underconfident_positive_ece(self) -> None:
        """All predictions at 0.1 but all correct -> ECE = 0.9 (positive)."""
        pairs = [(0.1, 1) for _ in range(50)]
        ece = compute_ece(pairs, n_bins=10)
        assert ece == pytest.approx(0.9, abs=1e-6)

    def test_bin_count_affects_granularity(self) -> None:
        """Finer binning can only reveal finer miscalibration; ECE is non-negative."""
        pairs = [(0.4, 1), (0.4, 0), (0.6, 1), (0.6, 1)]
        ece_10 = compute_ece(pairs, n_bins=10)
        ece_2 = compute_ece(pairs, n_bins=2)
        assert ece_10 >= 0.0
        assert ece_2 >= 0.0

    def test_empty_input_raises(self) -> None:
        """Empty iterable raises ValueError."""
        with pytest.raises(ValueError, match="empty|no predictions"):
            compute_ece([], n_bins=10)

    def test_out_of_range_confidence_raises(self) -> None:
        """Confidences outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match=r"confidence"):
            compute_ece([(1.5, 1)], n_bins=10)
        with pytest.raises(ValueError, match=r"confidence"):
            compute_ece([(-0.1, 0)], n_bins=10)

    def test_non_binary_label_raises(self) -> None:
        """Labels outside {0, 1} raise ValueError."""
        with pytest.raises(ValueError, match=r"observed|label"):
            compute_ece([(0.5, 2)], n_bins=10)

    def test_zero_bins_raises(self) -> None:
        """n_bins <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_bins"):
            compute_ece([(0.5, 1)], n_bins=0)

    def test_single_class_all_correct(self) -> None:
        """All labels are 1 and confidence is 1.0 -> ECE = 0."""
        pairs = [(1.0, 1) for _ in range(10)]
        ece = compute_ece(pairs, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_single_class_all_wrong(self) -> None:
        """All labels are 0 and confidence is 1.0 -> ECE = 1.0."""
        pairs = [(1.0, 0) for _ in range(10)]
        ece = compute_ece(pairs, n_bins=10)
        assert ece == pytest.approx(1.0, abs=1e-9)

    def test_accepts_generator(self) -> None:
        """compute_ece accepts any iterable, not just list."""
        gen = ((0.99, i % 2) for i in range(4))
        ece = compute_ece(gen, n_bins=10)
        assert ece == pytest.approx(0.49, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_brier
# ---------------------------------------------------------------------------


class TestComputeBrier:
    """Brier score = mean((conf - label)^2)."""

    def test_perfect_prediction_zero(self) -> None:
        """Confidence matches label exactly -> Brier = 0."""
        pairs = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
        assert compute_brier(pairs) == pytest.approx(0.0, abs=1e-9)

    def test_worst_case_one(self) -> None:
        """Confidence 1.0 but label 0 -> Brier = 1.0."""
        pairs = [(1.0, 0), (0.0, 1)]
        assert compute_brier(pairs) == pytest.approx(1.0, abs=1e-9)

    def test_uniform_05_with_mixed_labels(self) -> None:
        """All predictions 0.5 against 50/50 labels -> Brier = 0.25."""
        pairs = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
        assert compute_brier(pairs) == pytest.approx(0.25, abs=1e-9)

    def test_empty_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty|no predictions"):
            compute_brier([])

    def test_non_binary_label_raises(self) -> None:
        """Non-binary labels raise ValueError."""
        with pytest.raises(ValueError, match=r"observed|label"):
            compute_brier([(0.5, -1)])


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------


class TestReliabilityDiagram:
    """Structural tests for the reliability diagram output."""

    def test_returns_expected_keys(self) -> None:
        """Output dict has the documented keys."""
        pairs = [(0.2, 0), (0.8, 1)]
        rd = reliability_diagram(pairs, n_bins=10)
        assert set(rd.keys()) == {
            "bin_edges",
            "bin_centers",
            "mean_confidence",
            "accuracy",
            "count",
            "n_bins",
            "total",
        }

    def test_bin_count_matches_n_bins(self) -> None:
        """Length of per-bin arrays equals n_bins."""
        pairs = [(i / 100.0, i % 2) for i in range(1, 100)]
        rd = reliability_diagram(pairs, n_bins=5)
        for key in ("bin_centers", "mean_confidence", "accuracy", "count"):
            assert len(rd[key]) == 5
        assert len(rd["bin_edges"]) == 6

    def test_bin_counts_sum_to_total_input(self) -> None:
        """Each sample lands in exactly one bin."""
        pairs = [(0.05, 0), (0.5, 1), (0.95, 1), (0.5, 0), (0.5, 1)]
        rd = reliability_diagram(pairs, n_bins=10)
        assert sum(rd["count"]) == len(pairs)
        assert rd["total"] == len(pairs)

    def test_empty_bins_report_zero_values(self) -> None:
        """Bins with no samples have count 0, mean_conf/accuracy = 0."""
        pairs = [(0.05, 0), (0.95, 1)]
        rd = reliability_diagram(pairs, n_bins=10)
        # Middle bins should be empty.
        zero_bins = [i for i, c in enumerate(rd["count"]) if c == 0]
        assert len(zero_bins) >= 2
        for i in zero_bins:
            assert rd["mean_confidence"][i] == 0.0
            assert rd["accuracy"][i] == 0.0

    def test_json_serializable(self) -> None:
        """Output round-trips through JSON serialization."""
        pairs = [(0.1, 0), (0.9, 1)]
        rd = reliability_diagram(pairs, n_bins=4)
        blob = json.dumps(rd)
        restored = json.loads(blob)
        assert restored["n_bins"] == 4
        assert restored["total"] == 2

    def test_empty_input_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty|no predictions"):
            reliability_diagram([], n_bins=10)

    def test_confidence_1_lands_in_last_bin(self) -> None:
        """Exact upper bound (1.0) must not get dropped (right-edge inclusive)."""
        pairs = [(1.0, 1), (1.0, 1)]
        rd = reliability_diagram(pairs, n_bins=10)
        assert rd["count"][-1] == 2


# ---------------------------------------------------------------------------
# compare_annotations integration
# ---------------------------------------------------------------------------


def _write_annotations(path: Path, annotations: list[dict]) -> Path:
    path.write_text(json.dumps({"annotations": annotations}))
    return path


class TestCompareAnnotationsCalibration:
    """compare_annotations exposes per-category ece/brier/reliability_bins."""

    def test_calibration_fields_present_per_category(self, tmp_path: Path) -> None:
        """Each category metrics dict gains ece/brier/reliability_bins."""
        heuristic = [
            {
                "trial_path": "t1",
                "categories": [{"name": "cat_a", "confidence": 0.8}],
            },
            {
                "trial_path": "t2",
                "categories": [{"name": "cat_a", "confidence": 0.8}],
            },
        ]
        llm = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
            {"trial_path": "t2", "categories": []},
        ]
        file_h = _write_annotations(tmp_path / "h.json", heuristic)
        file_l = _write_annotations(tmp_path / "l.json", llm)

        result = compare_annotations(file_h, file_l)
        cat_a = result["categories"]["cat_a"]
        assert "ece" in cat_a
        assert "brier" in cat_a
        assert "reliability_bins" in cat_a
        # Structural: reliability_bins has the standard keys.
        assert "n_bins" in cat_a["reliability_bins"]
        assert "count" in cat_a["reliability_bins"]

    def test_calibration_support_counts_samples(self, tmp_path: Path) -> None:
        """Support == number of (confidence, observed) pairs used for the metric.

        Pairs are built over the union of category names on the non-error
        shared subset, so support equals shared_trials.
        """
        heuristic = [
            {
                "trial_path": "t1",
                "categories": [{"name": "cat_a", "confidence": 0.7}],
            },
            {
                "trial_path": "t2",
                "categories": [{"name": "cat_a", "confidence": 0.7}],
            },
            {
                "trial_path": "t3",
                "categories": [],
            },
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
        assert cat_a["support"] == 3  # 3 shared trials
        assert cat_a["reliability_bins"]["total"] == 3

    def test_backwards_compatible_fields_preserved(self, tmp_path: Path) -> None:
        """Existing TP/FP/FN/precision/recall/F1 still returned alongside calibration."""
        heuristic = [
            {
                "trial_path": "t1",
                "categories": [{"name": "cat_a", "confidence": 0.9}],
            },
        ]
        llm = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
        ]
        file_h = _write_annotations(tmp_path / "h.json", heuristic)
        file_l = _write_annotations(tmp_path / "l.json", llm)

        result = compare_annotations(file_h, file_l)
        cat_a = result["categories"]["cat_a"]
        # Legacy keys must still exist.
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ):
            assert key in cat_a

    def test_markdown_includes_calibration_table(self, tmp_path: Path) -> None:
        """format_markdown renders a section with ECE/Brier columns."""
        heuristic = [
            {
                "trial_path": "t1",
                "categories": [{"name": "cat_a", "confidence": 0.9}],
            },
        ]
        llm = [
            {"trial_path": "t1", "categories": [{"name": "cat_a"}]},
        ]
        file_h = _write_annotations(tmp_path / "h.json", heuristic)
        file_l = _write_annotations(tmp_path / "l.json", llm)

        summary = compare_annotations(file_h, file_l)
        md = format_markdown(summary)
        assert "ECE" in md
        assert "Brier" in md

    def test_calibration_is_zero_when_predictions_are_binary_and_correct(
        self, tmp_path: Path
    ) -> None:
        """Confidence = 1.0 exactly when LLM agrees -> ECE == 0, Brier == 0.

        Reliability is a degenerate "perfect" case: predictions concentrate in
        the top bin and always match labels.
        """
        heuristic = [
            {
                "trial_path": f"t{i}",
                "categories": [{"name": "cat_a", "confidence": 1.0}],
            }
            for i in range(5)
        ]
        llm = [{"trial_path": f"t{i}", "categories": [{"name": "cat_a"}]} for i in range(5)]
        file_h = _write_annotations(tmp_path / "h.json", heuristic)
        file_l = _write_annotations(tmp_path / "l.json", llm)

        summary = compare_annotations(file_h, file_l)
        cat_a = summary["categories"]["cat_a"]
        assert cat_a["ece"] == pytest.approx(0.0, abs=1e-6)
        assert cat_a["brier"] == pytest.approx(0.0, abs=1e-6)
