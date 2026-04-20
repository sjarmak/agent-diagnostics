"""DuckDB-backed SQL query engine for observatory data.

Auto-registers ``signals``, ``annotations``, and ``manifests`` table aliases
from JSONL or Parquet files found in the data directory. Parquet files in
``data_dir/export/`` take precedence over JSONL files in ``data_dir/``.

Requires the ``query`` extra: ``pip install agent-diagnostics[query]``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TABLE_NAMES: tuple[str, ...] = ("signals", "annotations", "manifests")

# Valid values for the ``--format`` flag on ``observatory query``.
OUTPUT_FORMATS: tuple[str, ...] = ("table", "json", "jsonl", "csv")


_ZERO_COLUMN_PARQUET_MARKER = "at least one non-root column"

# DuckDB surfaces "missing table" through two different error shapes, and we
# must recognize both:
#
# 1. CatalogException (clean case, no Python global shadows the name):
#      Catalog Error: Table with name annotations does not exist!
#
# 2. InvalidInputException (replacement-scan case — the reason this module
#    exists). `from __future__ import annotations` binds a `_Feature` global
#    that DuckDB's replacement scan picks up, producing:
#      Invalid Input Error: Python Object "annotations" of type "_Feature"
#      ... not suitable for replacement scans.
#
# Case-insensitive: DuckDB echoes identifiers in their original case today,
# but that is not a documented contract.
_MISSING_TABLE_RE = re.compile(
    r"Table with name (?P<catalog>\w+) does not exist"
    r'|Python Object "(?P<replacement>\w+)".*not suitable for replacement scans',
    re.IGNORECASE | re.DOTALL,
)


class MissingTableError(LookupError):
    """A known table alias was referenced but has no backing file.

    Raised by :func:`run_query` when a SQL query references one of
    :data:`TABLE_NAMES` for which neither ``data_dir/<name>.jsonl`` nor
    ``data_dir/export/<name>.parquet`` is present.

    Subclasses :class:`LookupError` so broad-catch callers can handle the
    "table not found" category without importing this symbol.
    """


def _register_parquet_view(con: Any, name: str, parquet_path: Path) -> bool:
    """Try to register *parquet_path* as DuckDB view *name*.

    Returns ``True`` on success, ``False`` if the file cannot be read.

    Failures are split by severity so we don't silently swallow data loss:

    * Known zero-column stubs (shipped by older versions of ``export.py``)
      log at WARNING — this is an expected degraded mode, fall through to
      the JSONL source if present.
    * Any other ``duckdb.Error`` (permissions, corruption, unsupported
      format) logs at ERROR so the skip is loud. Caller still receives
      ``False`` so the query can degrade rather than crash mid-pipeline.
    """
    import duckdb

    escaped = str(parquet_path).replace("'", "''")
    try:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')")
        return True
    except duckdb.Error as exc:
        if _ZERO_COLUMN_PARQUET_MARKER in str(exc):
            logger.warning(
                "Skipping zero-column Parquet stub for table %r at %s "
                "(known issue, falling through to JSONL if present)",
                name,
                parquet_path,
            )
        else:
            logger.error(
                "Unreadable Parquet for table %r at %s: %s — skipping; "
                "downstream queries may return stale or empty results",
                name,
                parquet_path,
                exc,
            )
        return False


def run_query(sql: str, data_dir: str | Path = "data/") -> list[dict[str, Any]]:
    """Execute *sql* against auto-registered tables and return rows as dicts.

    For each table name in :data:`TABLE_NAMES`:

    1. If ``data_dir/export/<name>.parquet`` exists and is readable, register
       it via ``read_parquet()``.
    2. If the Parquet file is absent or unreadable (e.g. a zero-column stub
       or a corrupted file), fall through to ``data_dir/<name>.jsonl`` via
       ``read_json_auto()`` if present.
    3. If neither file is available, the table is skipped.

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
    MissingTableError
        When the SQL references a known table alias
        (``signals``/``annotations``/``manifests``) that has no backing
        JSONL or Parquet file. The message names both expected paths.
    duckdb.Error
        On invalid SQL or query execution failure (including catalog errors
        for table names outside :data:`TABLE_NAMES`).
    """
    import duckdb

    data_path = Path(data_dir)
    con = duckdb.connect(":memory:")

    registered: set[str] = set()
    for name in TABLE_NAMES:
        parquet_path = data_path / "export" / f"{name}.parquet"
        jsonl_path = data_path / f"{name}.jsonl"

        if parquet_path.is_file():
            if _register_parquet_view(con, name, parquet_path):
                registered.add(name)
                continue
            # Parquet unreadable -- fall through to JSONL if present.
        if jsonl_path.is_file():
            escaped = str(jsonl_path).replace("'", "''")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_json_auto('{escaped}')")
            registered.add(name)

    try:
        try:
            result = con.execute(sql)
        except duckdb.Error as exc:
            missing = _match_unregistered_known_table(str(exc), registered)
            if missing is not None:
                raise MissingTableError(_missing_table_message(missing, data_path)) from exc
            raise

        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
    finally:
        con.close()

    return [dict(zip(columns, row)) for row in rows]


def _match_unregistered_known_table(error_message: str, registered: set[str]) -> str | None:
    """Return the unregistered TABLE_NAMES entry named in the error, or None.

    DuckDB's error message is the sole source of truth for which identifier
    it could not resolve — delegating to DuckDB avoids SQL-parsing pitfalls
    (comments, string literals, quoted identifiers, CTEs) that a naive
    preflight regex would mishandle.
    """
    match = _MISSING_TABLE_RE.search(error_message)
    if match is None:
        return None
    captured = match.group("catalog") or match.group("replacement")
    name = captured.lower()
    if name in TABLE_NAMES and name not in registered:
        return name
    return None


def _missing_table_message(name: str, data_path: Path) -> str:
    jsonl_path = data_path / f"{name}.jsonl"
    parquet_path = data_path / "export" / f"{name}.parquet"
    return (
        f"No data found for table '{name}'. Expected one of:\n"
        f"  - {jsonl_path}\n"
        f"  - {parquet_path}\n"
        f"Run the ingest/annotate/export pipeline to populate it."
    )


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
    import duckdb

    data_path = Path(data_dir)
    con = duckdb.connect(":memory:")

    schema: dict[str, list[dict[str, str]]] = {}

    for name in TABLE_NAMES:
        parquet_path = data_path / "export" / f"{name}.parquet"
        jsonl_path = data_path / f"{name}.jsonl"

        registered = False
        if parquet_path.is_file():
            registered = _register_parquet_view(con, name, parquet_path)
        if not registered:
            if jsonl_path.is_file():
                escaped = str(jsonl_path).replace("'", "''")
                con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_json_auto('{escaped}')")
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
            str(row.get(col, "")).ljust(width) for col, width in zip(columns, col_widths)
        )
        data_lines.append(line)

    parts = [header, separator] + data_lines
    parts.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(parts)


def format_rows(rows: list[dict[str, Any]], fmt: str) -> str:
    """Render *rows* in the requested output *fmt*.

    Supported formats (see :data:`OUTPUT_FORMATS`):

    * ``"table"`` — human-readable ASCII table via :func:`format_table`
      (default, preserves the pre-flag behaviour of the CLI).
    * ``"json"`` — JSON list-of-objects, two-space indented. ``[]`` when
      *rows* is empty. ``default=str`` handles DuckDB types (``datetime``,
      ``Decimal``, etc.) that are not natively JSON-serialisable.
    * ``"jsonl"`` — one JSON object per line, no trailing newline. Empty
      string when *rows* is empty. Stream-friendly for piping to
      line-oriented consumers (``jq``, ``awk``).
    * ``"csv"`` — RFC 4180 output with a header row. Empty string when
      *rows* is empty (no columns known). Uses ``csv.QUOTE_MINIMAL`` so
      fields containing commas, quotes, or newlines are properly quoted.

    Non-string, non-numeric cell values are coerced with ``str()`` for CSV
    so the output round-trips through :class:`csv.DictReader` without
    provoking a ``_csv.Error``. JSON/JSONL retain native types where
    possible and fall back to ``str()`` for exotic scalars.

    Parameters
    ----------
    rows:
        Result rows as returned by :func:`run_query`.
    fmt:
        Output format name. Must be one of :data:`OUTPUT_FORMATS`.

    Raises
    ------
    ValueError
        If *fmt* is not a recognised output format. Callers wiring this
        into argparse should restrict ``choices`` to :data:`OUTPUT_FORMATS`
        so this path is only reachable from programmatic callers.
    """
    if fmt == "table":
        return format_table(rows)
    if fmt == "json":
        return json.dumps(rows, indent=2, default=str)
    if fmt == "jsonl":
        if not rows:
            return ""
        return "\n".join(json.dumps(row, default=str) for row in rows)
    if fmt == "csv":
        if not rows:
            return ""
        buf = io.StringIO()
        columns = list(rows[0].keys())
        writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        # csv.writer stringifies scalars itself, but coerce None -> "" so
        # NULL cells render as empty fields instead of the literal "None".
        for row in rows:
            writer.writerow(_csv_cell(row.get(col)) for col in columns)
        return buf.getvalue().rstrip("\r\n")
    raise ValueError(
        f"unknown query output format: {fmt!r}. Expected one of: {', '.join(OUTPUT_FORMATS)}"
    )


def _csv_cell(value: Any) -> str:
    """Coerce a DuckDB cell value into a CSV-safe string.

    ``None`` becomes the empty field (``""``), matching typical CSV
    conventions for NULL. Scalars are passed through ``str()``.
    """
    if value is None:
        return ""
    return str(value)
