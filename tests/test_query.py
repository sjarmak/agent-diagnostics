"""Tests for the DuckDB query engine and CLI subcommand."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "query"


# Layout contract for the fixtures below: Parquet files live under
# ``data_dir/export/<name>.parquet`` (matching how export.py writes them),
# JSONL files live at ``data_dir/<name>.jsonl``. The resolution order in
# ``query.py`` prefers Parquet, falling back to JSONL on failure.


@pytest.fixture
def zero_col_annotations_dir(tmp_path: Path) -> Path:
    """tmp_path seeded with a valid ``signals.parquet`` and a zero-column
    ``annotations.parquet`` stub — the exact shape older ``export.py``
    produced (``pa.table({})`` has no schema and ``read_parquet`` rejects
    it with "Need at least one non-root column").

    Deliberately provides NO ``annotations.jsonl`` fallback so the
    "unrelated queries survive a skipped zero-column table" scenario
    stays under test. Callers that need a fallback write one explicitly.
    """
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    signals_path = export_dir / "signals.parquet"
    con = duckdb.connect(":memory:")
    con.execute(
        f"COPY (SELECT 'pq-task' AS task_id, 0.99 AS reward) TO '{signals_path}' (FORMAT PARQUET)"
    )
    con.close()

    pq.write_table(pa.table({}), export_dir / "annotations.parquet")
    return tmp_path


@pytest.fixture
def corrupt_annotations_dir(tmp_path: Path) -> Path:
    """tmp_path seeded with a corrupt ``annotations.parquet`` (truncated
    garbage) and a valid ``annotations.jsonl`` fallback.

    The corrupt-bytes case is unexpected data loss: callers of ``run_query``
    and ``get_schema`` are expected to log at ERROR and fall through to
    JSONL, so the fallback is part of the contract under test.
    """
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    (export_dir / "annotations.parquet").write_bytes(b"not-a-parquet-file")
    (tmp_path / "annotations.jsonl").write_text(
        '{"trial_id": "t1", "category_name": "exception_crash"}\n'
    )
    return tmp_path


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
        f"COPY (SELECT 'pq-task' AS task_id, 0.99 AS reward) TO '{parquet_path}' (FORMAT PARQUET)"
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


# ---------------------------------------------------------------------------
# run_query — friendly error when a known table is referenced but unregistered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["signals", "annotations", "manifests"])
def test_run_query_missing_known_table_raises(tmp_path: Path, missing: str):
    """Querying any unregistered TABLE_NAMES entry raises MissingTableError
    (a LookupError) whose message names both expected backing paths."""
    from agent_diagnostics.query import MissingTableError, run_query

    with pytest.raises(MissingTableError) as exc_info:
        run_query(f"SELECT * FROM {missing}", data_dir=tmp_path)

    # LookupError subclass contract lets broad-catch callers handle this
    # without importing MissingTableError.
    assert isinstance(exc_info.value, LookupError)
    msg = str(exc_info.value)
    assert missing in msg
    assert str(tmp_path / f"{missing}.jsonl") in msg
    assert str(tmp_path / "export" / f"{missing}.parquet") in msg


def test_run_query_missing_annotations_ok_if_unreferenced(tmp_path: Path):
    """A query that touches only existing tables succeeds when other known
    tables are absent — the friendly error must not become a false positive
    for queries that never reach the missing table."""
    from agent_diagnostics.query import run_query

    # Only signals.jsonl exists; annotations and manifests are unregistered.
    (tmp_path / "signals.jsonl").write_text(json.dumps({"task_id": "t1", "reward": 1.0}) + "\n")

    rows = run_query("SELECT task_id FROM signals", data_dir=tmp_path)
    assert rows == [{"task_id": "t1"}]


def test_run_query_cte_shadows_missing_table(tmp_path: Path):
    """A CTE named after a missing TABLE_NAMES entry must not be preempted.

    DuckDB resolves the CTE internally and never raises CatalogException, so
    the catch-and-rewrite path cannot trigger here — this test is a forward-
    looking regression guard against a future naive switch to regex preflight.
    """
    from agent_diagnostics.query import run_query

    rows = run_query(
        "WITH annotations AS (SELECT 42 AS v) SELECT v FROM annotations",
        data_dir=tmp_path,
    )
    assert rows == [{"v": 42}]


def test_run_query_missing_table_case_insensitive(tmp_path: Path):
    """DuckDB is case-insensitive for identifiers; the friendly rewrite
    must still fire for uppercase references."""
    from agent_diagnostics.query import MissingTableError, run_query

    with pytest.raises(MissingTableError) as exc_info:
        run_query("SELECT * FROM ANNOTATIONS", data_dir=tmp_path)
    assert "annotations" in str(exc_info.value).lower()


def test_run_query_string_literal_not_misparsed(tmp_path: Path):
    """A string literal containing a missing-table name must not trigger
    the friendly rewrite — the query itself never references the table."""
    from agent_diagnostics.query import run_query

    (tmp_path / "signals.jsonl").write_text(
        json.dumps({"task_id": "annotations", "reward": 0.5}) + "\n"
    )

    rows = run_query(
        "SELECT task_id FROM signals WHERE task_id = 'annotations'",
        data_dir=tmp_path,
    )
    assert rows == [{"task_id": "annotations"}]


def test_run_query_unknown_table_passthrough(tmp_path: Path):
    """Catalog errors for non-TABLE_NAMES tables must pass through with
    DuckDB's original error — we only rewrite for our known aliases.

    Registers a real signals.jsonl so a too-broad catch that re-raises
    MissingTableError on *any* catalog failure would be caught here.
    """
    import duckdb

    from agent_diagnostics.query import MissingTableError, run_query

    (tmp_path / "signals.jsonl").write_text(json.dumps({"task_id": "t1", "reward": 1.0}) + "\n")

    with pytest.raises(duckdb.Error) as exc_info:
        run_query("SELECT * FROM some_custom_table", data_dir=tmp_path)
    assert not isinstance(exc_info.value, MissingTableError)


def test_unreadable_parquet_skipped(zero_col_annotations_dir: Path, caplog):
    """An unreadable Parquet file must not break unrelated queries.

    Regression for the shipped stub bug: ``data/export/annotations.parquet``
    was a 318-byte zero-column Parquet file. Eagerly registering a view
    over it caused ``SELECT count(*) FROM signals`` to fail with a cryptic
    ``Need at least one non-root column`` error. The query engine must
    skip unreadable Parquet files and log a warning, so other tables
    remain queryable.
    """
    from agent_diagnostics.query import run_query

    with caplog.at_level(logging.WARNING, logger="agent_diagnostics.query"):
        rows = run_query(
            "SELECT count(*) AS n FROM signals",
            data_dir=zero_col_annotations_dir,
        )

    assert rows == [{"n": 1}]
    # The zero-column stub case should log at WARNING (known issue path),
    # not ERROR — and the log must identify the skipped table.
    skip_records = [r for r in caplog.records if "annotations" in r.getMessage()]
    assert skip_records, "expected a log entry mentioning the skipped annotations table"
    assert any(r.levelno == logging.WARNING for r in skip_records), (
        "zero-column stub should log at WARNING, not ERROR"
    )


def test_corrupt_parquet_logs_error_and_falls_through_to_jsonl(
    corrupt_annotations_dir: Path, caplog
):
    """Corrupt (non-stub) Parquet must log ERROR and fall through to JSONL.

    The zero-column-stub case is an expected degraded mode (WARNING).
    Any other unreadable-Parquet case is unexpected data loss and must
    surface loudly so it can't be missed in log review.
    """
    from agent_diagnostics.query import run_query

    with caplog.at_level(logging.ERROR, logger="agent_diagnostics.query"):
        rows = run_query(
            "SELECT count(*) AS n FROM annotations",
            data_dir=corrupt_annotations_dir,
        )

    # Fallback to JSONL must have happened.
    assert rows == [{"n": 1}]
    # And the unexpected-corruption case should have logged at ERROR.
    error_records = [
        r for r in caplog.records if r.levelno >= logging.ERROR and "annotations" in r.getMessage()
    ]
    assert error_records, "expected ERROR-level log for unexpected corrupt Parquet"


# ---------------------------------------------------------------------------
# get_schema — Parquet fallback coverage (mirrors run_query tests above)
# ---------------------------------------------------------------------------


def test_get_schema_unreadable_parquet_falls_through_to_jsonl(
    zero_col_annotations_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """get_schema must skip a zero-column Parquet stub and fall through to JSONL.

    ``get_schema`` uses the same ``_register_parquet_view`` helper as
    ``run_query`` via a parallel loop. The run_query path has regression
    coverage (``test_unreadable_parquet_skipped``); this test mirrors it
    for get_schema so the schema-introspection path cannot silently lose
    its fallback behaviour.
    """
    # The fixture deliberately omits annotations.jsonl; get_schema's fallback
    # is exactly what this test exercises, so we add one explicitly here.
    (zero_col_annotations_dir / "annotations.jsonl").write_text(
        '{"trial_id": "t1", "category_name": "exception_crash"}\n'
    )

    from agent_diagnostics.query import get_schema

    with caplog.at_level(logging.WARNING, logger="agent_diagnostics.query"):
        output = get_schema(zero_col_annotations_dir, fmt="json")

    schema = json.loads(output)

    # Signals described from the valid parquet, annotations from JSONL.
    assert "signals" in schema
    assert "annotations" in schema
    signals_cols = {c["column_name"] for c in schema["signals"]}
    assert "task_id" in signals_cols
    assert "reward" in signals_cols
    annotation_cols = {c["column_name"] for c in schema["annotations"]}
    assert "trial_id" in annotation_cols
    assert "category_name" in annotation_cols

    # The zero-column stub case should log at WARNING (known issue path).
    skip_records = [r for r in caplog.records if "annotations" in r.getMessage()]
    assert skip_records, "expected a log entry mentioning the skipped annotations table"
    assert any(r.levelno == logging.WARNING for r in skip_records), (
        "zero-column stub should log at WARNING, not ERROR"
    )


def test_get_schema_corrupt_parquet_logs_error_and_falls_through_to_jsonl(
    corrupt_annotations_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt Parquet in the get_schema path must log ERROR and fall through.

    Mirrors ``test_corrupt_parquet_logs_error_and_falls_through_to_jsonl``
    but exercises ``get_schema`` instead of ``run_query``.
    """
    from agent_diagnostics.query import get_schema

    with caplog.at_level(logging.ERROR, logger="agent_diagnostics.query"):
        output = get_schema(corrupt_annotations_dir, fmt="json")

    schema = json.loads(output)

    # Fallback to JSONL must have happened — annotations columns visible.
    assert "annotations" in schema
    annotation_cols = {c["column_name"] for c in schema["annotations"]}
    assert "trial_id" in annotation_cols
    assert "category_name" in annotation_cols

    # Unexpected-corruption case must log at ERROR.
    error_records = [
        r for r in caplog.records if r.levelno >= logging.ERROR and "annotations" in r.getMessage()
    ]
    assert error_records, "expected ERROR-level log for unexpected corrupt Parquet in get_schema"


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
