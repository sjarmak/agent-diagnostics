"""End-to-end golden path test: extract -> annotate -> report."""

from __future__ import annotations

import warnings
from pathlib import Path

from agent_diagnostics.annotator import annotate_trial
from agent_diagnostics.report import generate_report
from agent_diagnostics.signals import extract_all
from agent_diagnostics.taxonomy import valid_category_names

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden_trial"


def test_e2e_golden_path(tmp_path: Path) -> None:
    """Run the full pipeline against the golden fixture and verify the report."""
    # --- Step 1: Extract signals ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        signals_list = extract_all(GOLDEN_DIR)

    assert len(signals_list) == 1, f"Expected 1 trial, got {len(signals_list)}"
    signals = signals_list[0]

    assert signals["task_id"] == "golden_test_task"
    assert signals["has_result_json"] is True
    assert signals["has_trajectory"] is True

    # --- Step 2: Annotate ---
    annotations_raw = annotate_trial(signals)
    assert len(annotations_raw) > 0, "Expected at least one category assignment"

    # All assigned names must be valid taxonomy categories
    from importlib import resources

    _v3_path = Path(str(resources.files("agent_diagnostics") / "taxonomy_v3.yaml"))
    valid_names = valid_category_names(_v3_path)
    for cat in annotations_raw:
        assert cat.name in valid_names, f"Unknown category: {cat.name}"

    # --- Step 3: Build annotation document and generate report ---
    annotation_doc = {
        "annotations": [
            {
                "task_id": signals["task_id"],
                "config_name": signals["config_name"],
                "benchmark": signals["benchmark"],
                "passed": signals["passed"],
                "reward": signals["reward"],
                "categories": [
                    {
                        "name": c.name,
                        "confidence": c.confidence,
                        "evidence": c.evidence,
                    }
                    for c in annotations_raw
                ],
            }
        ]
    }

    md_path, json_path = generate_report(annotation_doc, tmp_path)

    # --- Step 4: Verify report ---
    assert md_path.exists()
    assert json_path.exists()

    report_text = md_path.read_text()
    assert len(report_text) > 0, "Report markdown is empty"
    assert "Agent Reliability Observatory" in report_text

    # At least one known category name appears in the report
    assigned_names = [c.name for c in annotations_raw]
    found_in_report = [name for name in assigned_names if name in report_text]
    assert (
        len(found_in_report) > 0
    ), f"None of the assigned categories {assigned_names} found in report"
