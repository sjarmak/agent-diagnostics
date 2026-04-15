"""DuckDB-backed SQL query engine for observatory data.

Auto-registers ``signals``, ``annotations``, and ``manifests`` table aliases
from JSONL or Parquet files found in the data directory. Parquet files in
``data_dir/export/`` take precedence over JSONL files in ``data_dir/``.

Requires the ``query`` extra: ``pip install agent-diagnostics[query]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TABLE_NAMES: tuple[str, ...] = ("signals", "annotations", "manifests")


def run_query(sql: str, data_dir: str | Path = "data/") -> list[dict[str, Any]]:
    """Execute *sql* against auto-registered tables and return rows as dicts.

    For each table name in :data:`TABLE_NAMES`:

    1. If ``data_dir/export/<name>.parquet`` exists, register it via
       ``read_parquet()``.
    2. Otherwise, if ``data_dir/<name>.jsonl`` exists, register it via
       ``read_json_auto()``.
    3. If neither file is found, the table is silently skipped.

    Parameters
    ----------
    sql:
        A SQL query string. Use ``signals``, ``annotations``, or
        ``manifests`` as table names.
    data_dir:
        Root data directory. Defaults to ``"data/"``.

    Returns
    -------
    list[dict[str, Any]]
        Each row as a dictionary keyed by column name.

    Raises
    ------
    duckdb.Error
        On invalid SQL or query execution failure.
    """
    import duckdb

    data_path = Path(data_dir)
    con = duckdb.connect(":memory:")

    for name in TABLE_NAMES:
        parquet_path = data_path / "export" / f"{name}.parquet"
        jsonl_path = data_path / f"{name}.jsonl"

        if parquet_path.is_file():
            escaped = str(parquet_path).replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')"
            )
        elif jsonl_path.is_file():
            escaped = str(jsonl_path).replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_json_auto('{escaped}')"
            )

    result = con.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    con.close()

    return [dict(zip(columns, row)) for row in rows]


def get_schema(
    data_dir: str | Path = "data/",
    fmt: str = "markdown",
) -> str:
    """Return column-level schema for all registered tables.

    For each table in :data:`TABLE_NAMES` that has a backing file, runs
    ``DESCRIBE SELECT * FROM <table>`` and collects column names and types.

    Parameters
    ----------
    data_dir:
        Root data directory. Defaults to ``"data/"``.
    fmt:
        Output format — ``"json"`` or ``"markdown"`` (default).

    Returns
    -------
    str
        Schema information formatted as JSON or a Markdown table.
    """
    import json

    import duckdb

    data_path = Path(data_dir)
    con = duckdb.connect(":memory:")

    schema: dict[str, list[dict[str, str]]] = {}

    for name in TABLE_NAMES:
        parquet_path = data_path / "export" / f"{name}.parquet"
        jsonl_path = data_path / f"{name}.jsonl"

        if parquet_path.is_file():
            escaped = str(parquet_path).replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')"
            )
        elif jsonl_path.is_file():
            escaped = str(jsonl_path).replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_json_auto('{escaped}')"
            )
        else:
            continue

        result = con.execute(f"DESCRIBE SELECT * FROM {name}")
        columns = []
        for row in result.fetchall():
            columns.append({"column_name": row[0], "column_type": row[1]})
        schema[name] = columns

    con.close()

    if fmt == "json":
        return json.dumps(schema, indent=2)

    # Markdown format
    lines: list[str] = []
    for table_name, columns in schema.items():
        lines.append(f"## {table_name}")
        lines.append("")
        lines.append("| Column | Type |")
        lines.append("|--------|------|")
        for col in columns:
            lines.append(f"| {col['column_name']} | {col['column_type']} |")
        lines.append("")

    return "\n".join(lines) if lines else "(no tables found)"


def format_table(rows: list[dict[str, Any]]) -> str:
    """Format *rows* as a simple ASCII table.

    Returns an empty string when *rows* is empty.
    """
    if not rows:
        return "(0 rows)"

    columns = list(rows[0].keys())

    # Compute column widths (header vs data)
    col_widths: list[int] = []
    for col in columns:
        max_data = max((len(str(row.get(col, ""))) for row in rows), default=0)
        col_widths.append(max(len(col), max_data))

    # Build header
    header = " | ".join(col.ljust(width) for col, width in zip(columns, col_widths))
    separator = "-+-".join("-" * width for width in col_widths)

    # Build data rows
    data_lines: list[str] = []
    for row in rows:
        line = " | ".join(
            str(row.get(col, "")).ljust(width)
            for col, width in zip(columns, col_widths)
        )
        data_lines.append(line)

    parts = [header, separator] + data_lines
    parts.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(parts)
