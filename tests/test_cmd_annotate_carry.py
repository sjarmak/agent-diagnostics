"""Tests for the signal carry-over fix in ``cmd_annotate`` (bead ifp).

Verifies that ``cmd_annotate`` propagates analytic fields from ``TrialSignals``
into each annotation record so downstream consumers (``report.py``) can slice
by benchmark, agent, trajectory availability, model, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import jsonschema

from agent_diagnostics.cli import cmd_annotate
from agent_diagnostics.signals import write_jsonl
from agent_diagnostics.types import CategoryAssignment


def _signal(
    task_id: str,
    *,
    benchmark: str = "swebench_lite",
    agent_name: str = "claude-code",
    has_trajectory: bool = True,
    model: str = "claude-sonnet-4-6",
    config_name: str = "baseline",
    benchmark_source: str = "manifest",
    trial_id: str = "0" * 32,
    trial_id_full: str = "0" * 64,
    trajectory_length: int | None = 42,
    total_turns: int | None = 7,
) -> dict:
    signal: dict = {
        "task_id": task_id,
        "trial_path": f"runs/{task_id}",
        "reward": 1.0,
        "passed": True,
        "benchmark": benchmark,
        "agent_name": agent_name,
        "has_trajectory": has_trajectory,
        "model": model,
        "config_name": config_name,
        "benchmark_source": benchmark_source,
        "trial_id": trial_id,
        "trial_id_full": trial_id_full,
    }
    if trajectory_length is not None:
        signal["trajectory_length"] = trajectory_length
    if total_turns is not None:
        signal["total_turns"] = total_turns
    return signal


@patch("agent_diagnostics.annotator.annotate_trial")
@patch("agent_diagnostics.taxonomy.load_taxonomy")
def test_carries_analytic_fields_into_annotation_records(
    mock_taxonomy, mock_annotate_trial, tmp_path: Path
) -> None:
    mock_taxonomy.return_value = {"version": "3.0"}
    mock_annotate_trial.return_value = [
        CategoryAssignment(name="cat1", confidence=0.9, evidence="ev")
    ]

    signals = [
        _signal(
            "t1",
            benchmark="swebench_lite",
            agent_name="claude-code",
            trajectory_length=180,
            total_turns=12,
        ),
        _signal(
            "t2",
            benchmark="locobench",
            agent_name="codex",
            has_trajectory=False,
            trajectory_length=None,
            total_turns=None,
        ),
    ]
    signals_file = tmp_path / "signals.jsonl"
    write_jsonl(signals, signals_file)

    output = tmp_path / "annotations.json"
    args = argparse.Namespace(signals=str(signals_file), output=str(output))
    cmd_annotate(args)

    data = json.loads(output.read_text())
    annotations = data["annotations"]
    assert len(annotations) == 2

    record_by_task = {a["task_id"]: a for a in annotations}

    a1 = record_by_task["t1"]
    assert a1["benchmark"] == "swebench_lite"
    assert a1["agent_name"] == "claude-code"
    assert a1["has_trajectory"] is True
    assert a1["model"] == "claude-sonnet-4-6"
    assert a1["config_name"] == "baseline"
    assert a1["benchmark_source"] == "manifest"
    assert a1["trial_id"] == "0" * 32
    assert a1["trial_id_full"] == "0" * 64
    assert a1["trajectory_length"] == 180
    assert a1["total_turns"] == 12

    a2 = record_by_task["t2"]
    assert a2["benchmark"] == "locobench"
    assert a2["agent_name"] == "codex"
    assert a2["has_trajectory"] is False
    # trajectory_length / total_turns absent when signal omits them (no trajectory file).
    assert "trajectory_length" not in a2
    assert "total_turns" not in a2


@patch("agent_diagnostics.annotator.annotate_trial")
@patch("agent_diagnostics.taxonomy.load_taxonomy")
def test_missing_signal_fields_are_not_emitted(
    mock_taxonomy, mock_annotate_trial, tmp_path: Path
) -> None:
    """When a signal lacks analytic fields, the annotation record omits them.

    Keeps schema-compliant output without forcing empty strings or ``None``
    into optional fields.
    """
    mock_taxonomy.return_value = {"version": "3.0"}
    mock_annotate_trial.return_value = []

    signals = [{"task_id": "bare", "reward": 0.0, "passed": False}]
    signals_file = tmp_path / "signals.jsonl"
    write_jsonl(signals, signals_file)

    output = tmp_path / "annotations.json"
    args = argparse.Namespace(signals=str(signals_file), output=str(output))
    cmd_annotate(args)

    record = json.loads(output.read_text())["annotations"][0]
    for absent in (
        "trial_id",
        "trial_id_full",
        "agent_name",
        "model",
        "config_name",
        "benchmark",
        "benchmark_source",
        "has_trajectory",
        "trajectory_length",
        "total_turns",
    ):
        assert absent not in record, f"{absent!r} should be omitted when signal lacks it"


@patch("agent_diagnostics.annotator.annotate_trial")
@patch("agent_diagnostics.taxonomy.load_taxonomy")
def test_zero_valued_trajectory_length_is_preserved(
    mock_taxonomy, mock_annotate_trial, tmp_path: Path
) -> None:
    """Integer 0 is a legitimate trajectory_length (crashed before first turn).

    The carry guard must distinguish absent (None) from the real value 0.
    Regression test: ensures the guard does not accidentally drop falsy ints.
    """
    mock_taxonomy.return_value = {"version": "3.0"}
    mock_annotate_trial.return_value = []

    signals = [_signal("zero_turn", trajectory_length=0, total_turns=0)]
    signals_file = tmp_path / "signals.jsonl"
    write_jsonl(signals, signals_file)

    output = tmp_path / "annotations.json"
    args = argparse.Namespace(signals=str(signals_file), output=str(output))
    cmd_annotate(args)

    record = json.loads(output.read_text())["annotations"][0]
    assert record["trajectory_length"] == 0
    assert record["total_turns"] == 0


@patch("agent_diagnostics.annotator.annotate_trial")
@patch("agent_diagnostics.taxonomy.load_taxonomy")
def test_output_validates_against_annotation_schema(
    mock_taxonomy, mock_annotate_trial, tmp_path: Path
) -> None:
    """The enriched annotation output must still validate against the JSON Schema.

    ``annotation_schema.json`` uses ``additionalProperties: false``, so any
    new field carried over must be declared as a property.
    """
    mock_taxonomy.return_value = {"version": "3.0"}
    mock_annotate_trial.return_value = [
        CategoryAssignment(name="cat1", confidence=0.9, evidence="ev")
    ]

    signals = [_signal("t1")]
    signals_file = tmp_path / "signals.jsonl"
    write_jsonl(signals, signals_file)

    output = tmp_path / "annotations.json"
    args = argparse.Namespace(signals=str(signals_file), output=str(output))
    cmd_annotate(args)

    data = json.loads(output.read_text())
    schema_path = (
        Path(__file__).parent.parent
        / "src"
        / "agent_diagnostics"
        / "annotation_schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(instance=data, schema=schema)
