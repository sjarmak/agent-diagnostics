"""Tests for agent_diagnostics.blend_labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_diagnostics.blend_labels import blend


def _write_annotation_file(tmp_path: Path, name: str, annotations: list[dict]) -> Path:
    """Write an annotation document JSON file and return its path."""
    doc = {
        "schema_version": "observatory-annotation-v1",
        "annotations": annotations,
    }
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def _write_calibration_file(tmp_path: Path, categories: dict[str, dict]) -> Path:
    """Write a calibration JSON file and return its path."""
    doc = {"categories": categories}
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(doc))
    return p


def _ann(
    trial_path: str,
    task_id: str = "task-1",
    categories: list[dict] | None = None,
    reward: float = 0.0,
    passed: bool = False,
) -> dict:
    """Create a minimal annotation dict."""
    return {
        "task_id": task_id,
        "trial_path": trial_path,
        "reward": reward,
        "passed": passed,
        "categories": categories or [],
    }


class TestBasicBlend:
    def test_returns_required_keys(self, tmp_path: Path) -> None:
        heur = _write_annotation_file(tmp_path, "heur.json", [])
        llm = _write_annotation_file(tmp_path, "llm.json", [])

        result = blend(heur, llm)

        assert result["schema_version"] == "observatory-annotation-v1"
        assert "annotations" in result
        assert "blend_metadata" in result

    def test_basic_blend_with_data(self, tmp_path: Path) -> None:
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    "trial/a",
                    categories=[{"name": "exception_crash", "confidence": 0.9}],
                )
            ],
        )
        llm = _write_annotation_file(
            tmp_path,
            "llm.json",
            [_ann("trial/b", categories=[{"name": "query_churn", "confidence": 0.8}])],
        )

        result = blend(heur, llm)

        assert len(result["annotations"]) == 2


class TestLLMPriority:
    def test_llm_categories_take_priority(self, tmp_path: Path) -> None:
        """When both LLM and heuristic annotate the same trial, LLM categories win."""
        shared_path = "trial/shared"
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    shared_path,
                    categories=[
                        {"name": "exception_crash", "confidence": 0.5},
                        {"name": "query_churn", "confidence": 0.7},
                    ],
                )
            ],
        )
        llm = _write_annotation_file(
            tmp_path,
            "llm.json",
            [
                _ann(
                    shared_path,
                    categories=[
                        {"name": "exception_crash", "confidence": 0.95},
                    ],
                )
            ],
        )

        result = blend(heur, llm)

        annotations = result["annotations"]
        assert len(annotations) == 1
        cats = {c["name"]: c for c in annotations[0]["categories"]}
        # LLM version of exception_crash wins (source=llm, confidence=0.95)
        assert cats["exception_crash"]["source"] == "llm"
        assert cats["exception_crash"]["confidence"] == 0.95
        # query_churn is NOT a default trusted category, so not supplemented
        assert "query_churn" not in cats


class TestHeuristicOnlyTrusted:
    def test_only_trusted_categories_included(self, tmp_path: Path) -> None:
        """Heuristic-only trials only include trusted categories."""
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    "trial/heur_only",
                    categories=[
                        {"name": "exception_crash", "confidence": 0.9},
                        {"name": "query_churn", "confidence": 0.8},
                        {"name": "retrieval_failure", "confidence": 0.7},
                    ],
                )
            ],
        )
        llm = _write_annotation_file(tmp_path, "llm.json", [])

        result = blend(heur, llm)

        assert len(result["annotations"]) == 1
        cat_names = {c["name"] for c in result["annotations"][0]["categories"]}
        # Only exception_crash is in the default trusted set
        assert cat_names == {"exception_crash"}

    def test_heuristic_only_skipped_if_no_trusted(self, tmp_path: Path) -> None:
        """Heuristic-only trial with no trusted categories is skipped."""
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    "trial/untrusted",
                    categories=[{"name": "query_churn", "confidence": 0.9}],
                )
            ],
        )
        llm = _write_annotation_file(tmp_path, "llm.json", [])

        result = blend(heur, llm)

        assert len(result["annotations"]) == 0


class TestMaxHeuristicSamples:
    def test_cap_limits_heuristic_only_count(self, tmp_path: Path) -> None:
        heur_anns = [
            _ann(
                f"trial/h{i}",
                categories=[{"name": "exception_crash", "confidence": 0.9}],
            )
            for i in range(10)
        ]
        heur = _write_annotation_file(tmp_path, "heur.json", heur_anns)
        llm = _write_annotation_file(tmp_path, "llm.json", [])

        result = blend(heur, llm, max_heuristic_samples=3)

        assert result["blend_metadata"]["heuristic_only_trials"] == 3
        assert len(result["annotations"]) == 3


class TestCalibrationFile:
    def test_calibration_selects_trusted_categories(self, tmp_path: Path) -> None:
        """With calibration, only categories meeting F1 threshold are trusted."""
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    "trial/cal",
                    categories=[
                        {"name": "cat_high_f1", "confidence": 0.8},
                        {"name": "cat_low_f1", "confidence": 0.8},
                    ],
                )
            ],
        )
        llm = _write_annotation_file(tmp_path, "llm.json", [])
        cal = _write_calibration_file(
            tmp_path,
            {
                "cat_high_f1": {"f1": 0.85},
                "cat_low_f1": {"f1": 0.4},
            },
        )

        result = blend(heur, llm, calibration_file=cal, heuristic_trust_threshold=0.7)

        assert len(result["annotations"]) == 1
        cat_names = {c["name"] for c in result["annotations"][0]["categories"]}
        assert "cat_high_f1" in cat_names
        assert "cat_low_f1" not in cat_names

    def test_calibration_threshold_exact_boundary(self, tmp_path: Path) -> None:
        """Category with F1 exactly at threshold is trusted."""
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    "trial/boundary",
                    categories=[{"name": "boundary_cat", "confidence": 0.7}],
                )
            ],
        )
        llm = _write_annotation_file(tmp_path, "llm.json", [])
        cal = _write_calibration_file(tmp_path, {"boundary_cat": {"f1": 0.7}})

        result = blend(heur, llm, calibration_file=cal, heuristic_trust_threshold=0.7)

        assert len(result["annotations"]) == 1


class TestDefaultTrustedCategories:
    def test_default_categories_used_without_calibration(self, tmp_path: Path) -> None:
        expected_defaults = {
            "rate_limited_run",
            "exception_crash",
            "near_miss",
            "over_exploration",
            "edit_verify_loop_failure",
        }
        heur_anns = [
            _ann(f"trial/{cat}", categories=[{"name": cat, "confidence": 0.9}])
            for cat in expected_defaults
        ]
        heur = _write_annotation_file(tmp_path, "heur.json", heur_anns)
        llm = _write_annotation_file(tmp_path, "llm.json", [])

        result = blend(heur, llm)

        assert len(result["annotations"]) == len(expected_defaults)
        assert (
            set(result["blend_metadata"]["trusted_heuristic_categories"])
            == expected_defaults
        )


class TestBlendMetadata:
    def test_metadata_counts_correct(self, tmp_path: Path) -> None:
        heur = _write_annotation_file(
            tmp_path,
            "heur.json",
            [
                _ann(
                    "trial/shared",
                    categories=[{"name": "exception_crash", "confidence": 0.9}],
                ),
                _ann(
                    "trial/heur_only",
                    categories=[{"name": "near_miss", "confidence": 0.8}],
                ),
            ],
        )
        llm = _write_annotation_file(
            tmp_path,
            "llm.json",
            [
                _ann(
                    "trial/shared",
                    categories=[{"name": "query_churn", "confidence": 0.9}],
                ),
                _ann(
                    "trial/llm_only",
                    categories=[{"name": "retrieval_failure", "confidence": 0.7}],
                ),
            ],
        )

        result = blend(heur, llm)
        meta = result["blend_metadata"]

        assert meta["llm_trials"] == 2
        assert meta["heuristic_only_trials"] == 1  # trial/heur_only
        assert meta["total_blended"] == 3  # shared + heur_only + llm_only
        assert meta["trust_threshold"] == 0.7
