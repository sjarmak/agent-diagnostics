"""Regression guard: verify JSONL pipeline data contract after PRD changes.

Ensures that:
1. trial_id and trial_id_full are the ONLY new TrialSignals fields vs the
   pre-PRD baseline (29 original fields).
2. Trajectory fields (tool_call_sequence, files_read_list, files_edited_list)
   remain inline in TrialSignals — not split into a separate table.
3. annotation_schema.json is still valid and validates a conforming document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_type_hints

import jsonschema
import pytest

from agent_diagnostics.signals import extract_signals
from agent_diagnostics.types import TrialSignals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 29 fields that existed before the PRD added trial_id / trial_id_full.
PRE_PRD_FIELDS: frozenset[str] = frozenset(
    {
        "task_id",
        "model",
        "agent_name",
        "config_name",
        "benchmark",
        "reward",
        "passed",
        "has_verifier_result",
        "total_turns",
        "tool_calls_total",
        "search_tool_calls",
        "edit_tool_calls",
        "code_nav_tool_calls",
        "semantic_search_tool_calls",
        "unique_files_read",
        "unique_files_edited",
        "files_read_list",
        "files_edited_list",
        "error_count",
        "retry_count",
        "trajectory_length",
        "has_result_json",
        "has_trajectory",
        "duration_seconds",
        "rate_limited",
        "exception_crashed",
        "patch_size_lines",
        "tool_call_sequence",
        "benchmark_source",
    }
)

TRAJECTORY_INLINE_FIELDS: frozenset[str] = frozenset(
    {
        "tool_call_sequence",
        "files_read_list",
        "files_edited_list",
    }
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "agent_diagnostics"
    / "annotation_schema.json"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_result(trial_dir: Path, data: dict) -> None:
    (trial_dir / "result.json").write_text(json.dumps(data))


def _write_trajectory(trial_dir: Path, data: dict) -> None:
    (trial_dir / "trajectory.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Test 1: trial_id and trial_id_full are the ONLY new fields
# ---------------------------------------------------------------------------


class TestTrialSignalsOnlyNewFieldsAreTrialId:
    """Confirm that no unexpected fields crept into TrialSignals."""

    def test_new_fields_exactly_trial_id_pair(self) -> None:
        current_fields = set(get_type_hints(TrialSignals).keys())
        new_fields = current_fields - PRE_PRD_FIELDS
        assert new_fields == {
            "trial_id",
            "trial_id_full",
        }, f"Unexpected new fields in TrialSignals: {new_fields - {'trial_id', 'trial_id_full'}}"

    def test_pre_prd_fields_still_present(self) -> None:
        """All 29 original fields must still exist."""
        current_fields = set(get_type_hints(TrialSignals).keys())
        missing = PRE_PRD_FIELDS - current_fields
        assert missing == set(), f"Pre-PRD fields missing from TrialSignals: {missing}"

    def test_total_field_count_is_31(self) -> None:
        current_fields = get_type_hints(TrialSignals)
        assert (
            len(current_fields) == 31
        ), f"Expected 31 fields (29 original + 2 new), got {len(current_fields)}"


# ---------------------------------------------------------------------------
# Test 2: trajectory fields remain inline in signals
# ---------------------------------------------------------------------------


class TestTrajectoryFieldsInlineInSignals:
    """Trajectory data must be stored inline in TrialSignals, not factored out."""

    def test_trajectory_fields_in_type_hints(self) -> None:
        hints = get_type_hints(TrialSignals)
        for field in TRAJECTORY_INLINE_FIELDS:
            assert (
                field in hints
            ), f"Trajectory field '{field}' missing from TrialSignals type hints"

    def test_trajectory_fields_are_lists(self) -> None:
        hints = get_type_hints(TrialSignals)
        for field in TRAJECTORY_INLINE_FIELDS:
            assert (
                hints[field] == list[str]
            ), f"Expected list[str] for '{field}', got {hints[field]}"

    def test_extract_signals_populates_trajectory_fields(self, tmp_path: Path) -> None:
        """A synthetic trial produces signals with inline trajectory fields."""
        trial = tmp_path / "trial_inline"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "regression_task",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:05:00Z",
                "agent_info": {
                    "model_info": {"name": "anthropic/claude-sonnet-4-6"},
                },
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "function_name": "Read",
                                "arguments": {"file_path": "/src/app.py"},
                            },
                            {
                                "function_name": "Edit",
                                "arguments": {
                                    "file_path": "/src/app.py",
                                    "new_string": "fixed\n",
                                },
                            },
                        ],
                        "observation": {},
                    },
                ],
            },
        )

        signals = extract_signals(
            trial, suite_mapping={"regression_task": "test_bench"}
        )

        # All three trajectory fields must be present and populated
        assert "tool_call_sequence" in signals
        assert "files_read_list" in signals
        assert "files_edited_list" in signals

        assert isinstance(signals["tool_call_sequence"], list)
        assert isinstance(signals["files_read_list"], list)
        assert isinstance(signals["files_edited_list"], list)

        # Verify the tool call sequence captured the two calls
        assert len(signals["tool_call_sequence"]) == 2
        assert "Read" in signals["tool_call_sequence"]
        assert "Edit" in signals["tool_call_sequence"]

        # Verify file lists are populated
        assert "/src/app.py" in signals["files_read_list"]
        assert "/src/app.py" in signals["files_edited_list"]


# ---------------------------------------------------------------------------
# Test 3: annotation_schema.json still valid and validates a document
# ---------------------------------------------------------------------------


class TestAnnotationSchemaStillValid:
    """annotation_schema.json must load as valid JSON and validate conforming docs."""

    def test_schema_file_exists(self) -> None:
        assert (
            _SCHEMA_PATH.is_file()
        ), f"annotation_schema.json not found at {_SCHEMA_PATH}"

    def test_schema_loads_as_valid_json(self) -> None:
        schema = json.loads(_SCHEMA_PATH.read_text())
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert schema.get("$id") == "observatory-annotation-v1"

    def test_schema_validates_minimal_document(self) -> None:
        schema = json.loads(_SCHEMA_PATH.read_text())
        minimal_doc = {
            "schema_version": "observatory-annotation-v1",
            "annotations": [
                {
                    "task_id": "test_task_001",
                    "trial_path": "runs/trial_01",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [
                        {"name": "retrieval_failure", "confidence": 0.85},
                    ],
                },
            ],
        }
        # Should not raise
        jsonschema.validate(instance=minimal_doc, schema=schema)

    def test_schema_rejects_invalid_document(self) -> None:
        schema = json.loads(_SCHEMA_PATH.read_text())
        invalid_doc = {
            "schema_version": "observatory-annotation-v1",
            # missing required "annotations"
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=invalid_doc, schema=schema)

    def test_schema_rejects_extra_top_level_properties(self) -> None:
        schema = json.loads(_SCHEMA_PATH.read_text())
        doc_with_extra = {
            "schema_version": "observatory-annotation-v1",
            "annotations": [],
            "unknown_field": "should_fail",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=doc_with_extra, schema=schema)
