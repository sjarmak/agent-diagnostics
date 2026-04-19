"""Tests for the DuckDB query engine and CLI subcommand."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "query"


# ---------------------------------------------------------------------------
# run_query — basic operations
# ---------------------------------------------------------------------------


def test_run_query_count_signals():
    """Basic SELECT count(*) against the signals fixture."""
    from agent_diagnostics.query import run_query

    rows = run_query("SELECT count(*) AS n FROM signals", data_dir=FIXTURES_DIR)
    assert rows == [{"n": 5}]


def test_run_query_select_columns():
    """SELECT specific columns returns expected values."""
    from agent_diagnostics.query import run_query

    rows = run_query(
        "SELECT task_id, reward FROM signals WHERE passed = true ORDER BY reward",
        data_dir=FIXTURES_DIR,
    )
    assert len(rows) == 3
    assert rows[0]["task_id"] == "task-alpha-001"
    assert rows[0]["reward"] == pytest.approx(0.6)
    assert rows[-1]["task_id"] == "task-beta-002"
    assert rows[-1]["reward"] == pytest.approx(1.0)


def test_run_query_annotations_table():
    """Annotations JSONL is auto-registered and queryable."""
    from agent_diagnostics.query import run_query

    rows = run_query("SELECT count(*) AS n FROM annotations", data_dir=FIXTURES_DIR)
    assert rows == [{"n": 5}]


def test_run_query_manifests_table():
    """Manifests JSONL is auto-registered and queryable."""
    from agent_diagnostics.query import run_query

    rows = run_query("SELECT count(*) AS n FROM manifests", data_dir=FIXTURES_DIR)
    assert rows == [{"n": 2}]


# ---------------------------------------------------------------------------
# Parquet registration and preference
# ---------------------------------------------------------------------------


def test_parquet_auto_registered(tmp_path: Path):
    """Parquet files in export/ are auto-registered."""
    import duckdb

    # Write a small parquet file using duckdb itself
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    parquet_path = export_dir / "signals.parquet"

    con = duckdb.connect(":memory:")
    con.execute(
        "COPY (SELECT 'pq-task' AS task_id, 0.99 AS reward) "
        f"TO '{parquet_path}' (FORMAT PARQUET)"
    )
    con.close()

    from agent_diagnostics.query import run_query

    rows = run_query("SELECT task_id, reward FROM signals", data_dir=tmp_path)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "pq-task"
    assert float(rows[0]["reward"]) == pytest.approx(0.99)


def test_parquet_preferred_over_jsonl(tmp_path: Path):
    """When both Parquet and JSONL exist, Parquet wins."""
    import duckdb

    # Write JSONL with 10 rows
    jsonl_path = tmp_path / "signals.jsonl"
    lines = [f'{{"task_id": "jsonl-{i}", "reward": {i}}}' for i in range(10)]
    jsonl_path.write_text("\n".join(lines) + "\n")

    # Write Parquet with 1 row
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    parquet_path = export_dir / "signals.parquet"

    con = duckdb.connect(":memory:")
    con.execute(
        "COPY (SELECT 'parquet-only' AS task_id, 42.0 AS reward) "
        f"TO '{parquet_path}' (FORMAT PARQUET)"
    )
    con.close()

    from agent_diagnostics.query import run_query

    rows = run_query("SELECT count(*) AS n FROM signals", data_dir=tmp_path)
    # Should be 1 (from parquet), not 10 (from JSONL)
    assert rows == [{"n": 1}]


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def test_join_signals_annotations():
    """JOIN across signals and annotations works."""
    from agent_diagnostics.query import run_query

    sql = (
        "SELECT s.task_id, s.reward, a.category "
        "FROM signals s "
        "JOIN annotations a ON s.task_id = a.task_id "
        "WHERE a.category = 'clean_success'"
    )
    rows = run_query(sql, data_dir=FIXTURES_DIR)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-beta-002"
    assert rows[0]["category"] == "clean_success"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_sql_raises():
    """Invalid SQL raises a duckdb error."""
    import duckdb

    from agent_diagnostics.query import run_query

    with pytest.raises(duckdb.Error):
        run_query("SELECT * FROM nonexistent_table_xyz", data_dir=FIXTURES_DIR)


def test_missing_table_skipped(tmp_path: Path):
    """Tables with no corresponding file are silently skipped."""
    from agent_diagnostics.query import run_query

    # tmp_path has no data files, but a query that doesn't reference
    # any table should still work
    rows = run_query("SELECT 1 AS one", data_dir=tmp_path)
    assert rows == [{"one": 1}]


def test_unreadable_parquet_skipped(tmp_path: Path, caplog):
    """An unreadable Parquet file must not break unrelated queries.

    Regression for the shipped stub bug: ``data/export/annotations.parquet``
    was a 318-byte zero-column Parquet file.  Eagerly registering a view
    over it caused ``SELECT count(*) FROM signals`` to fail with a cryptic
    ``Need at least one non-root column`` error.  The query engine must
    skip unreadable Parquet files and log a warning, so other tables
    remain queryable.
    """
    import logging

    import duckdb

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    # Write a valid signals.parquet.
    signals_path = export_dir / "signals.parquet"
    con = duckdb.connect(":memory:")
    con.execute(
        "COPY (SELECT 'pq-task' AS task_id, 0.99 AS reward) "
        f"TO '{signals_path}' (FORMAT PARQUET)"
    )
    con.close()

    # Write a zero-column annotations.parquet -- this is the exact shape
    # the buggy export produced: ``pa.table({})`` has no schema, and
    # ``read_parquet`` rejects it with "Need at least one non-root column".
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({}), export_dir / "annotations.parquet")

    from agent_diagnostics.query import run_query

    with caplog.at_level(logging.WARNING, logger="agent_diagnostics.query"):
        rows = run_query("SELECT count(*) AS n FROM signals", data_dir=tmp_path)

    assert rows == [{"n": 1}]
    # The zero-column stub case should log at WARNING (known issue path),
    # not ERROR — and the log must identify the skipped table.
    skip_records = [
        r for r in caplog.records if "annotations" in r.getMessage()
    ]
    assert skip_records, "expected a log entry mentioning the skipped annotations table"
    assert any(r.levelno == logging.WARNING for r in skip_records), (
        "zero-column stub should log at WARNING, not ERROR"
    )


def test_corrupt_parquet_logs_error_and_falls_through_to_jsonl(
    tmp_path: Path, caplog
):
    """Corrupt (non-stub) Parquet must log ERROR and fall through to JSONL.

    The zero-column-stub case is an expected degraded mode (WARNING).
    Any other unreadable-Parquet case is unexpected data loss and must
    surface loudly so it can't be missed in log review.
    """
    import logging

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    # Write a corrupt annotations.parquet (truncated garbage).
    corrupt_path = export_dir / "annotations.parquet"
    corrupt_path.write_bytes(b"not-a-parquet-file")

    # And a JSONL fallback with real data.
    jsonl_path = tmp_path / "annotations.jsonl"
    jsonl_path.write_text(
        '{"trial_id": "t1", "category_name": "exception_crash"}\n'
    )

    from agent_diagnostics.query import run_query

    with caplog.at_level(logging.ERROR, logger="agent_diagnostics.query"):
        rows = run_query(
            "SELECT count(*) AS n FROM annotations", data_dir=tmp_path
        )

    # Fallback to JSONL must have happened.
    assert rows == [{"n": 1}]
    # And the unexpected-corruption case should have logged at ERROR.
    error_records = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR and "annotations" in r.getMessage()
    ]
    assert error_records, (
        "expected ERROR-level log for unexpected corrupt Parquet"
    )


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def test_format_table_with_data():
    """format_table produces readable ASCII output."""
    from agent_diagnostics.query import format_table

    rows = [
        {"name": "Alice", "score": 95},
        {"name": "Bob", "score": 87},
    ]
    output = format_table(rows)
    assert "name" in output
    assert "score" in output
    assert "Alice" in output
    assert "Bob" in output
    assert "(2 rows)" in output


def test_format_table_empty():
    """format_table with no rows returns a helpful message."""
    from agent_diagnostics.query import format_table

    assert format_table([]) == "(0 rows)"


def test_format_table_single_row():
    """format_table correctly pluralizes for 1 row."""
    from agent_diagnostics.query import format_table

    output = format_table([{"x": 1}])
    assert "(1 row)" in output


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cmd_query_integration(capsys):
    """cmd_query prints formatted table output."""
    from agent_diagnostics.cli import cmd_query

    args = SimpleNamespace(
        sql="SELECT count(*) AS n FROM signals",
        data_dir=str(FIXTURES_DIR),
    )
    cmd_query(args)
    captured = capsys.readouterr()
    assert "n" in captured.out
    assert "5" in captured.out


def test_cmd_query_error_exits(capsys):
    """cmd_query exits with code 1 on invalid SQL."""
    from agent_diagnostics.cli import cmd_query

    args = SimpleNamespace(
        sql="SELECT * FROM nonexistent_xyz",
        data_dir=str(FIXTURES_DIR),
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_query(args)
    assert exc_info.value.code == 1


def test_help_includes_query_subcommand():
    """observatory --help lists the query subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_diagnostics", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "query" in result.stdout
