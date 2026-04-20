#!/usr/bin/env python3
"""Generate canonical Markdown schema docs from in-code authoritative sources.

Writes ``docs/schemas/{signals,annotations,manifests}.md`` — one table per
file — from the following authoritative code sources:

- ``signals`` → :class:`agent_diagnostics.types.TrialSignals` TypedDict
- ``annotations`` → :data:`agent_diagnostics.annotation_store.ROW_FIELDS`
- ``manifests`` → column set emitted by
  :func:`agent_diagnostics.export.refresh_manifests`

The generator is the single source of truth for the docs.  The accompanying
test (``tests/test_schema_docs.py``) regenerates into a temp directory and
diffs against the checked-in files, failing CI on drift.

Usage::

    python scripts/gen_schema_docs.py               # writes docs/schemas/
    python scripts/gen_schema_docs.py --check       # fails if out of sync
    python scripts/gen_schema_docs.py --out <dir>   # write elsewhere
"""

from __future__ import annotations

import argparse
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "schemas"

# Ensure the src/ layout is importable when invoked as a standalone script
# (without requiring the package to be installed).
_SRC_PATH = str(REPO_ROOT / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)


# ---------------------------------------------------------------------------
# Column descriptions — maintained alongside the authoritative sources.
# ---------------------------------------------------------------------------
#
# Descriptions live here (not inline on the TypedDict) because TypedDict
# does not support per-field docstrings in a way the runtime can read.
# When a new field is added to TrialSignals / ROW_FIELDS / refresh_manifests,
# the drift test fails loudly and points the contributor here.

_SIGNALS_DESCRIPTIONS: dict[str, str] = {
    "trial_id": (
        "Stable 32-hex primary key: ``sha256(task_id||config_name||started_at||model)[:32]``."
    ),
    "trial_id_full": "Full 64-hex SHA-256 of the trial identity key.",
    "task_id": "Benchmark task identifier (e.g. ``swe-bench-lite-001``).",
    "model": "Model name that produced the trial (e.g. ``claude-sonnet-4-6``).",
    "agent_name": "Agent harness name (e.g. ``claude-code``, ``openhands``).",
    "config_name": "Harness configuration name (e.g. ``baseline``).",
    "benchmark": "Benchmark suite name (e.g. ``swebench_lite``).",
    "reward": "Final verifier reward in [0, 1], or NULL if no verifier result.",
    "passed": "True when ``reward > 0``.",
    "has_verifier_result": "True when a verifier reward was present.",
    "total_turns": "Total conversational turns recorded in the trajectory.",
    "tool_calls_total": "Total number of tool calls across all turns.",
    "search_tool_calls": "Tool calls classified as search (per tool_registry).",
    "edit_tool_calls": "Tool calls classified as edits (per tool_registry).",
    "code_nav_tool_calls": "Tool calls classified as code navigation.",
    "semantic_search_tool_calls": "Tool calls classified as semantic search.",
    "unique_files_read": "Count of distinct files opened for reading.",
    "unique_files_edited": "Count of distinct files edited or written.",
    "files_read_list": "Sorted list of file paths read during the trial.",
    "files_edited_list": "Sorted list of file paths edited during the trial.",
    "error_count": "Turns where the tool observation contained an error keyword.",
    "retry_count": "Consecutive tool calls with identical function + args.",
    "trajectory_length": "Length of the tool-call sequence.",
    "has_result_json": "True when ``result.json`` was present for this trial.",
    "has_trajectory": "True when ``trajectory.json`` was present for this trial.",
    "duration_seconds": (
        "Wall-clock seconds derived from ``started_at``/``finished_at`` (0.0 when unknown)."
    ),
    "rate_limited": "True when the harness exception indicated rate limiting.",
    "exception_crashed": "True when the trial crashed with an exception.",
    "patch_size_lines": "Total lines written across Edit/Write tool calls.",
    "tool_call_sequence": "Ordered list of tool function names across the trial.",
    "benchmark_source": (
        "How the benchmark field was resolved — ``manifest``, ``directory``, "
        "or empty string when unresolved."
    ),
}

_ANNOTATIONS_DESCRIPTIONS: dict[str, str] = {
    "trial_id": "Foreign key to ``signals.trial_id``.",
    "category_name": "Taxonomy category short name (e.g. ``context_loss``).",
    "confidence": "Assignment confidence in [0, 1].",
    "evidence": "Pointer to trace step or transcript line supporting the label.",
    "annotator_type": (
        "Annotator kind — one of ``manual``, ``heuristic``, ``llm``, "
        "``classifier``, ``ensemble``, ``blended``."
    ),
    "annotator_identity": (
        "Canonical annotator identity (e.g. ``llm:haiku-4``, ``heuristic:rule-engine``)."
    ),
    "taxonomy_version": "Version string of the taxonomy used (e.g. ``3.0``).",
    "annotated_at": "ISO-8601 timestamp when the annotation was created.",
}

_MANIFESTS_DESCRIPTIONS: dict[str, str] = {
    "manifest_id": "Sequential manifest identifier (e.g. ``manifest-001``).",
    "benchmark": "Benchmark suite name aggregated in this manifest row.",
    "task_count": "Distinct task count observed in ``signals`` for this benchmark.",
    "created_at": "ISO-8601 timestamp when the manifest row was generated.",
}

# Annotations schema is explicitly typed in annotation_store (as JSONL); the
# narrow-tall store stores primitives, so we map ROW_FIELDS onto DuckDB /
# pyarrow-friendly types here. Kept alongside _ANNOTATIONS_DESCRIPTIONS so
# adding a row field fails both lookups in one place.
_ANNOTATIONS_TYPES: dict[str, str] = {
    "trial_id": "VARCHAR",
    "category_name": "VARCHAR",
    "confidence": "DOUBLE",
    "evidence": "VARCHAR",
    "annotator_type": "VARCHAR",
    "annotator_identity": "VARCHAR",
    "taxonomy_version": "VARCHAR",
    "annotated_at": "VARCHAR",
}

# Manifest column types mirror the keys written by export.refresh_manifests.
_MANIFESTS_TYPES: dict[str, str] = {
    "manifest_id": "VARCHAR",
    "benchmark": "VARCHAR",
    "task_count": "BIGINT",
    "created_at": "VARCHAR",
}


# ---------------------------------------------------------------------------
# Type rendering for TrialSignals
# ---------------------------------------------------------------------------


def _render_type(tp: object) -> tuple[str, bool]:
    """Render a type annotation as a ``(display, nullable)`` pair.

    Maps Python/typing annotations to DuckDB-style column types that match
    how the Parquet export stores them:

    - ``str`` → VARCHAR
    - ``bool`` → BOOLEAN
    - ``int`` → BIGINT
    - ``float`` → DOUBLE
    - ``list[str]`` → VARCHAR[]
    - ``X | None`` → nullable flag set True, inner type rendered

    Unknown types fall back to ``repr(tp)`` so additions are visible
    rather than silently mislabeled.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    # Optional / Union-with-None — recognize both `typing.Union[X, None]` and
    # PEP 604 `X | None` (which reports origin as ``types.UnionType``).
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) != len(args)
        if len(non_none) == 1:
            inner, _ = _render_type(non_none[0])
            return inner, has_none
        # Union of multiple non-None types — render as a human-readable union.
        rendered = " or ".join(_render_type(a)[0] for a in non_none)
        return rendered, has_none

    if origin is list:
        inner_type = args[0] if args else str
        inner, _ = _render_type(inner_type)
        return f"{inner}[]", False

    if tp is str:
        return "VARCHAR", False
    if tp is bool:
        return "BOOLEAN", False
    if tp is int:
        return "BIGINT", False
    if tp is float:
        return "DOUBLE", False

    return repr(tp), False


# ---------------------------------------------------------------------------
# Per-table column extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Column:
    name: str
    type: str
    nullable: bool
    description: str


def _signals_columns() -> list[_Column]:
    """Extract signals columns from the TrialSignals TypedDict."""
    from agent_diagnostics.types import TrialSignals

    hints = get_type_hints(TrialSignals)
    columns: list[_Column] = []
    for name, tp in hints.items():
        type_str, nullable = _render_type(tp)
        description = _SIGNALS_DESCRIPTIONS.get(name)
        if description is None:
            raise KeyError(
                f"Missing description for TrialSignals field {name!r} in "
                f"scripts/gen_schema_docs.py — add it to _SIGNALS_DESCRIPTIONS."
            )
        columns.append(
            _Column(name=name, type=type_str, nullable=nullable, description=description)
        )
    return columns


def _annotations_columns() -> list[_Column]:
    """Extract annotation columns from AnnotationStore.ROW_FIELDS."""
    from agent_diagnostics.annotation_store import ROW_FIELDS

    columns: list[_Column] = []
    for name in ROW_FIELDS:
        type_str = _ANNOTATIONS_TYPES.get(name)
        description = _ANNOTATIONS_DESCRIPTIONS.get(name)
        if type_str is None or description is None:
            raise KeyError(
                f"Missing type or description for annotation field {name!r} in "
                f"scripts/gen_schema_docs.py."
            )
        columns.append(_Column(name=name, type=type_str, nullable=False, description=description))
    return columns


def _manifests_columns() -> list[_Column]:
    """Return manifest columns (keys written by export.refresh_manifests).

    Column order is driven by :data:`_MANIFESTS_DESCRIPTIONS` insertion order,
    which mirrors the key order emitted by ``export.refresh_manifests``.
    """
    columns: list[_Column] = []
    for name, description in _MANIFESTS_DESCRIPTIONS.items():
        type_str = _MANIFESTS_TYPES[name]
        columns.append(_Column(name=name, type=type_str, nullable=False, description=description))
    return columns


# ---------------------------------------------------------------------------
# Table metadata — source pointer printed in each doc's banner.
# ---------------------------------------------------------------------------


_TABLE_META: dict[str, dict[str, str]] = {
    "signals": {
        "title": "signals",
        "summary": "One row per benchmark trial. Produced by `agent-diagnostics ingest`.",
        "source_path": "src/agent_diagnostics/types.py",
        "source_symbol": "TrialSignals",
    },
    "annotations": {
        "title": "annotations",
        "summary": (
            "Narrow-tall annotation store: one row per (trial, category, annotator) triple."
        ),
        "source_path": "src/agent_diagnostics/annotation_store.py",
        "source_symbol": "ROW_FIELDS",
    },
    "manifests": {
        "title": "manifests",
        "summary": (
            "Per-benchmark task-count aggregates. Produced by "
            "`agent-diagnostics manifest refresh`."
        ),
        "source_path": "src/agent_diagnostics/export.py",
        "source_symbol": "refresh_manifests",
    },
}

_COLUMN_EXTRACTORS: dict[str, typing.Callable[[], list[_Column]]] = {
    "signals": _signals_columns,
    "annotations": _annotations_columns,
    "manifests": _manifests_columns,
}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_banner(table: str) -> str:
    meta = _TABLE_META[table]
    return (
        "<!-- AUTOGENERATED by scripts/gen_schema_docs.py — do not edit by hand.\n"
        f"     Source of truth: {meta['source_path']} ({meta['source_symbol']}).\n"
        "     Regenerate with: `python scripts/gen_schema_docs.py`. -->\n"
    )


def render_table_doc(table: str) -> str:
    """Return the Markdown document body for one table.

    The output is deterministic: identical inputs always produce byte-identical
    output, so the drift test can diff against the checked-in file.
    """
    if table not in _TABLE_META:
        raise ValueError(f"Unknown table {table!r}; expected one of {list(_TABLE_META)}")

    meta = _TABLE_META[table]
    columns = _COLUMN_EXTRACTORS[table]()

    lines: list[str] = []
    lines.append(_render_banner(table))
    lines.append(f"# `{meta['title']}` table schema")
    lines.append("")
    lines.append(meta["summary"])
    lines.append("")
    lines.append(
        f"Source of truth: [`{meta['source_path']}`](../../{meta['source_path']}) "
        f"(`{meta['source_symbol']}`)."
    )
    lines.append("")
    lines.append("| Column | Type | Nullable | Description |")
    lines.append("|--------|------|----------|-------------|")
    for col in columns:
        nullable = "yes" if col.nullable else "no"
        # Escape pipe characters in every cell so they don't break the
        # Markdown table layout (e.g. ``||`` separators in the trial_id
        # description, or union types like ``X or Y``).
        type_cell = col.type.replace("|", "\\|")
        description = col.description.replace("|", "\\|")
        lines.append(f"| {col.name} | {type_cell} | {nullable} | {description} |")
    lines.append("")

    return "\n".join(lines)


def write_all_schema_docs(out_dir: str | Path) -> list[Path]:
    """Write Markdown docs for all known tables into *out_dir*.

    Returns the list of written file paths.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for table in _TABLE_META:
        doc_path = out_path / f"{table}.md"
        doc_path.write_text(render_table_doc(table), encoding="utf-8")
        written.append(doc_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical Markdown schema docs from code."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any existing doc differs from the generated output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out

    if args.check:
        drift: list[str] = []
        for table in _TABLE_META:
            expected = render_table_doc(table)
            doc_path = out_dir / f"{table}.md"
            actual = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else ""
            if actual != expected:
                drift.append(table)
        if drift:
            print(
                "Schema docs out of sync with code: "
                + ", ".join(f"{t}.md" for t in drift)
                + "\nRun `python scripts/gen_schema_docs.py` to refresh.",
                file=sys.stderr,
            )
            return 1
        return 0

    written = write_all_schema_docs(out_dir)
    for path in written:
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
