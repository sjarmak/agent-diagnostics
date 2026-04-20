"""Tests for the 5 target SQL queries and the db schema introspection command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

QUERIES_DIR = Path(__file__).resolve().parent.parent / "docs" / "queries"


# ---------------------------------------------------------------------------
# Fixture: create a tmp data directory with rich JSONL test data
# ---------------------------------------------------------------------------


@pytest.fixture()
def rich_data_dir(tmp_path: Path) -> Path:
    """Create JSONL fixtures with all fields needed by the 5 target queries."""
    # signals.jsonl — includes trial_id, tool_call_sequence, and all columns
    signals = [
        {
            "trial_id": "trial-001",
            "task_id": "task-alpha",
            "model": "claude-sonnet-4-6",
            "benchmark": "swe_bench",
            "reward": 1.0,
            "passed": True,
            "total_turns": 12,
            "tool_calls_total": 8,
            "duration_seconds": 120.5,
            "tool_call_sequence": ["read", "edit", "bash", "read"],
        },
        {
            "trial_id": "trial-002",
            "task_id": "task-beta",
            "model": "claude-sonnet-4-6",
            "benchmark": "swe_bench",
            "reward": 0.0,
            "passed": False,
            "total_turns": 5,
            "tool_calls_total": 2,
            "duration_seconds": 45.0,
            "tool_call_sequence": ["read", "bash"],
        },
        {
            "trial_id": "trial-003",
            "task_id": "task-gamma",
            "model": "claude-opus-4-5",
            "benchmark": "bigcode_mcp",
            "reward": 0.5,
            "passed": True,
            "total_turns": 20,
            "tool_calls_total": 15,
            "duration_seconds": 300.0,
            "tool_call_sequence": ["bash", "edit", "edit", "bash"],
        },
        {
            "trial_id": "trial-004",
            "task_id": "task-delta",
            "model": "claude-opus-4-5",
            "benchmark": "bigcode_mcp",
            "reward": 0.0,
            "passed": False,
            "total_turns": 3,
            "tool_calls_total": 1,
            "duration_seconds": 60.0,
            "tool_call_sequence": ["read"],
        },
    ]
    signals_path = tmp_path / "signals.jsonl"
    signals_path.write_text("\n".join(json.dumps(row) for row in signals) + "\n")

    # annotations.jsonl — uses trial_id + category_name for co-occurrence query
    annotations = [
        {
            "trial_id": "trial-001",
            "category_name": "clean_success",
            "confidence": 0.95,
            "annotator": "heuristic",
        },
        {
            "trial_id": "trial-001",
            "category_name": "efficient_tooling",
            "confidence": 0.80,
            "annotator": "heuristic",
        },
        {
            "trial_id": "trial-002",
            "category_name": "premature_termination",
            "confidence": 0.88,
            "annotator": "heuristic",
        },
        {
            "trial_id": "trial-002",
            "category_name": "incomplete_verification",
            "confidence": 0.72,
            "annotator": "llm",
        },
        {
            "trial_id": "trial-003",
            "category_name": "clean_success",
            "confidence": 0.90,
            "annotator": "heuristic",
        },
    ]
    annotations_path = tmp_path / "annotations.jsonl"
    annotations_path.write_text("\n".join(json.dumps(row) for row in annotations) + "\n")

    # manifests.jsonl
    manifests = [
        {
            "manifest_id": "m-001",
            "benchmark": "swe_bench",
            "task_count": 100,
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "manifest_id": "m-002",
            "benchmark": "bigcode_mcp",
            "task_count": 50,
            "created_at": "2026-02-01T00:00:00Z",
        },
    ]
    manifests_path = tmp_path / "manifests.jsonl"
    manifests_path.write_text("\n".join(json.dumps(row) for row in manifests) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Helper: load a SQL file from docs/queries/
# ---------------------------------------------------------------------------


def _load_query(name: str) -> str:
    """Read a .sql file from docs/queries/ and return its contents."""
    path = QUERIES_DIR / name
    assert path.is_file(), f"Query file not found: {path}"
    return path.read_text()


# ---------------------------------------------------------------------------
# Test: per_model_outcomes.sql
# ---------------------------------------------------------------------------


class TestPerModelOutcomes:
    """Tests for per_model_outcomes.sql."""

    def test_runs_and_returns_expected_columns(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("per_model_outcomes.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        assert len(rows) > 0
        expected_columns = {"model", "trials", "passed", "pass_rate"}
        assert set(rows[0].keys()) == expected_columns

    def test_correct_aggregation(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("per_model_outcomes.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        # 2 models: claude-sonnet-4-6, claude-opus-4-5
        assert len(rows) == 2
        by_model = {r["model"]: r for r in rows}

        sonnet = by_model["claude-sonnet-4-6"]
        assert sonnet["trials"] == 2
        assert sonnet["passed"] == 1  # 1 passed, 1 failed

        opus = by_model["claude-opus-4-5"]
        assert opus["trials"] == 2
        assert opus["passed"] == 1


# ---------------------------------------------------------------------------
# Test: benchmark_model_matrix.sql
# ---------------------------------------------------------------------------


class TestBenchmarkModelMatrix:
    """Tests for benchmark_model_matrix.sql."""

    def test_runs_and_returns_expected_columns(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("benchmark_model_matrix.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        assert len(rows) > 0
        expected_columns = {"benchmark", "model", "trials", "pass_rate"}
        assert set(rows[0].keys()) == expected_columns

    def test_correct_grouping(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("benchmark_model_matrix.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        # 3 combos: (bigcode_mcp, opus), (swe_bench, sonnet), (bigcode_mcp not present for sonnet)
        # Actually: swe_bench+sonnet, bigcode_mcp+opus = 2 combos
        assert len(rows) == 2
        # Ordered by benchmark, model
        assert rows[0]["benchmark"] == "bigcode_mcp"
        assert rows[1]["benchmark"] == "swe_bench"


# ---------------------------------------------------------------------------
# Test: annotation_cooccurrence.sql
# ---------------------------------------------------------------------------


class TestAnnotationCooccurrence:
    """Tests for annotation_cooccurrence.sql."""

    def test_runs_and_returns_expected_columns(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("annotation_cooccurrence.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        assert len(rows) > 0
        expected_columns = {"cat_a", "cat_b", "co_occurrences"}
        assert set(rows[0].keys()) == expected_columns

    def test_finds_cooccurrences(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("annotation_cooccurrence.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        # trial-001 has clean_success + efficient_tooling (1 pair)
        # trial-002 has incomplete_verification + premature_termination (1 pair)
        assert len(rows) == 2
        pairs = {(r["cat_a"], r["cat_b"]) for r in rows}
        assert ("clean_success", "efficient_tooling") in pairs
        assert ("incomplete_verification", "premature_termination") in pairs


# ---------------------------------------------------------------------------
# Test: tool_sequence_patterns.sql
# ---------------------------------------------------------------------------


class TestToolSequencePatterns:
    """Tests for tool_sequence_patterns.sql."""

    def test_runs_and_returns_expected_columns(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("tool_sequence_patterns.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        assert len(rows) > 0
        expected_columns = {"tool_name", "usage_count"}
        assert set(rows[0].keys()) == expected_columns

    def test_counts_tool_usage(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("tool_sequence_patterns.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        by_tool = {r["tool_name"]: r["usage_count"] for r in rows}
        # read: trial-001 has 2, trial-002 has 1, trial-004 has 1 = 4
        assert by_tool["read"] == 4
        # edit: trial-001 has 1, trial-003 has 2 = 3
        assert by_tool["edit"] == 3
        # bash: trial-001 has 1, trial-002 has 1, trial-003 has 2 = 4
        assert by_tool["bash"] == 4


# ---------------------------------------------------------------------------
# Test: eval_subset_export.sql
# ---------------------------------------------------------------------------


class TestEvalSubsetExport:
    """Tests for eval_subset_export.sql."""

    def test_runs_and_returns_expected_columns(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("eval_subset_export.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        assert len(rows) > 0
        expected_columns = {
            "trial_id",
            "task_id",
            "model",
            "reward",
            "passed",
            "total_turns",
            "tool_calls_total",
            "duration_seconds",
        }
        assert set(rows[0].keys()) == expected_columns

    def test_only_failing_trials(self, rich_data_dir: Path):
        from agent_diagnostics.query import run_query

        sql = _load_query("eval_subset_export.sql")
        rows = run_query(sql, data_dir=rich_data_dir)

        # 2 failed trials: task-beta (trial-002) and task-delta (trial-004)
        assert len(rows) == 2
        task_ids = [r["task_id"] for r in rows]
        assert task_ids == ["task-beta", "task-delta"]  # ordered by task_id
        for row in rows:
            assert row["passed"] is False


# ---------------------------------------------------------------------------
# Test: get_schema() function
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Tests for the get_schema() function."""

    def test_json_format(self, rich_data_dir: Path):
        from agent_diagnostics.query import get_schema

        output = get_schema(rich_data_dir, fmt="json")
        schema = json.loads(output)

        assert "signals" in schema
        assert "annotations" in schema
        assert "manifests" in schema

        # Check signals columns include expected fields
        signal_cols = {c["column_name"] for c in schema["signals"]}
        assert "trial_id" in signal_cols
        assert "model" in signal_cols
        assert "passed" in signal_cols

        # Each column entry has column_name and column_type
        for col in schema["signals"]:
            assert "column_name" in col
            assert "column_type" in col

    def test_markdown_format(self, rich_data_dir: Path):
        from agent_diagnostics.query import get_schema

        output = get_schema(rich_data_dir, fmt="markdown")

        assert "## signals" in output
        assert "## annotations" in output
        assert "## manifests" in output
        assert "| Column | Type |" in output
        assert "|--------|------|" in output
        assert "trial_id" in output

    def test_empty_data_dir(self, tmp_path: Path):
        from agent_diagnostics.query import get_schema

        output = get_schema(tmp_path, fmt="markdown")
        assert output == "(no tables found)"

    def test_empty_data_dir_json(self, tmp_path: Path):
        from agent_diagnostics.query import get_schema

        output = get_schema(tmp_path, fmt="json")
        assert json.loads(output) == {}


# ---------------------------------------------------------------------------
# Test: cmd_db_schema CLI handler
# ---------------------------------------------------------------------------


class TestCmdDbSchema:
    """Tests for the cmd_db_schema CLI handler."""

    def test_json_output(self, rich_data_dir: Path, capsys):
        from agent_diagnostics.cli import cmd_db_schema

        args = SimpleNamespace(
            data_dir=str(rich_data_dir),
            format="json",
        )
        cmd_db_schema(args)

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert "signals" in schema
        assert "annotations" in schema
        assert "manifests" in schema

    def test_markdown_output(self, rich_data_dir: Path, capsys):
        from agent_diagnostics.cli import cmd_db_schema

        args = SimpleNamespace(
            data_dir=str(rich_data_dir),
            format="markdown",
        )
        cmd_db_schema(args)

        captured = capsys.readouterr()
        assert "## signals" in captured.out
        assert "| Column | Type |" in captured.out


# ---------------------------------------------------------------------------
# Test: db subcommand appears in CLI help
# ---------------------------------------------------------------------------


def test_help_includes_db_subcommand():
    """observatory --help lists the db subcommand."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "agent_diagnostics", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "db" in result.stdout
