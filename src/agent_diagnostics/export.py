"""Parquet export for observatory data.

Reads JSONL files (signals, annotations, manifests) from a data directory
and writes deterministic Parquet files plus a ``MANIFEST.json`` metadata
envelope to an output directory.

Requires the ``query`` extra: ``pip install agent-diagnostics[query]``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# List-typed columns in signals that must be stored as list<string> in Parquet
# ---------------------------------------------------------------------------

_LIST_STRING_COLUMNS: tuple[str, ...] = (
    "tool_call_sequence",
    "files_read_list",
    "files_edited_list",
)

# Fixed string for Parquet ``created_by`` metadata to avoid embedding the
# pyarrow version, which would break byte-identical re-exports.
_PARQUET_CREATED_BY = "observatory-export"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_commit() -> str:
    """Return the current HEAD commit hash, or ``'unknown'`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _read_meta_sidecar(jsonl_path: Path) -> dict[str, Any]:
    """Read the ``.meta.json`` sidecar for a JSONL file, returning ``{}`` on error."""
    meta_path = jsonl_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning a list of dicts (one per line)."""
    if not path.is_file():
        return []
    results: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                results.append(json.loads(stripped))
    return results


def _ensure_list_columns(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure list-typed columns are actual Python lists, not JSON strings."""
    for col in _LIST_STRING_COLUMNS:
        val = row.get(col)
        if val is None:
            row[col] = []
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                row[col] = parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                row[col] = []
    return row


def _sort_key_signals(row: dict[str, Any]) -> tuple[str, str, str]:
    """Stable sort key for signals rows."""
    return (
        str(row.get("trial_id", "")),
        str(row.get("task_id", "")),
        str(row.get("model", "")),
    )


def _sort_key_annotations(row: dict[str, Any]) -> tuple[str, str, str]:
    """Stable sort key for annotation rows."""
    return (
        str(row.get("task_id", "")),
        str(row.get("category", "")),
        str(row.get("annotator", "")),
    )


def _sort_key_manifests(row: dict[str, Any]) -> tuple[str, str]:
    """Stable sort key for manifest rows."""
    return (
        str(row.get("manifest_id", "")),
        str(row.get("benchmark", "")),
    )


def _build_signals_table(
    rows: list[dict[str, Any]],
) -> "pyarrow.Table":  # noqa: F821
    """Build a pyarrow Table for signals with explicit list<string> columns."""
    import pyarrow as pa

    if not rows:
        return pa.table({})

    # Collect all column names from the union of all rows, preserving
    # insertion order from the first row as base ordering.
    all_columns: dict[str, None] = {}
    for r in rows:
        for k in r:
            all_columns.setdefault(k, None)
    col_names = list(all_columns.keys())

    # Build column arrays
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for col in col_names:
        values = [r.get(col) for r in rows]
        if col in _LIST_STRING_COLUMNS:
            # Ensure every value is a list of strings
            clean_values = []
            for v in values:
                if isinstance(v, list):
                    clean_values.append([str(x) for x in v])
                else:
                    clean_values.append([])
            arr = pa.array(clean_values, type=pa.list_(pa.string()))
            fields.append(pa.field(col, pa.list_(pa.string())))
        else:
            arr = pa.array(values, from_pandas=True)
            fields.append(pa.field(col, arr.type))
        arrays.append(arr)

    schema = pa.schema(fields)
    return pa.table(arrays, schema=schema)


def _build_generic_table(
    rows: list[dict[str, Any]],
) -> "pyarrow.Table":  # noqa: F821
    """Build a pyarrow Table from a list of flat dicts."""
    import pyarrow as pa

    if not rows:
        return pa.table({})

    all_columns: dict[str, None] = {}
    for r in rows:
        for k in r:
            all_columns.setdefault(k, None)
    col_names = list(all_columns.keys())

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for col in col_names:
        values = [r.get(col) for r in rows]
        arr = pa.array(values, from_pandas=True)
        fields.append(pa.field(col, arr.type))
        arrays.append(arr)

    schema = pa.schema(fields)
    return pa.table(arrays, schema=schema)


def _write_parquet(table: "pyarrow.Table", path: Path) -> None:  # noqa: F821
    """Write a pyarrow Table to Parquet with deterministic settings."""
    import pyarrow.parquet as pq

    # Replace schema metadata with a fixed value so the pyarrow version
    # string does not leak into the file and break byte-identical re-exports.
    fixed_metadata = {b"created_by": _PARQUET_CREATED_BY.encode()}
    table = table.replace_schema_metadata(fixed_metadata)

    pq.write_table(
        table,
        str(path),
        compression="zstd",
        use_dictionary=True,
        write_statistics=False,
        write_batch_size=len(table) if len(table) > 0 else 1,
    )


def _detect_versions(
    data_dir: Path,
) -> tuple[str, str]:
    """Detect schema_version and taxonomy_version from sidecar meta files.

    Reads ``.meta.json`` sidecars for signals.jsonl and annotations.jsonl.
    For taxonomy_version, collects all distinct non-empty values and fails
    if more than one is found.

    Returns ``(schema_version, taxonomy_version)``.
    """
    schema_version = ""
    taxonomy_versions: set[str] = set()

    for name in ("signals", "annotations"):
        meta = _read_meta_sidecar(data_dir / f"{name}.jsonl")
        sv = meta.get("schema_version", "")
        if sv and not schema_version:
            schema_version = sv
        tv = meta.get("taxonomy_version", "")
        if tv:
            taxonomy_versions.add(tv)

    if len(taxonomy_versions) > 1:
        raise ValueError(
            f"Multiple taxonomy versions found in sidecar files: {taxonomy_versions}. "
            "All JSONL files must share one taxonomy version."
        )

    taxonomy_version = taxonomy_versions.pop() if taxonomy_versions else ""
    return schema_version, taxonomy_version


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_parquet(
    data_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Export JSONL data to deterministic Parquet files with a manifest.

    Reads ``signals.jsonl``, ``annotations.jsonl``, and ``manifests.jsonl``
    from *data_dir*, writes ``signals.parquet``, ``annotations.parquet``,
    ``manifests.parquet``, and ``MANIFEST.json`` to *out_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing JSONL source files.
    out_dir:
        Directory to write Parquet files and MANIFEST.json.

    Returns
    -------
    dict
        The manifest dict written to MANIFEST.json.

    Raises
    ------
    ValueError
        If more than one distinct taxonomy_version is found across inputs.
    FileNotFoundError
        If *data_dir* does not exist.
    """
    data_path = Path(data_dir)
    out_path = Path(out_dir)

    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    # --- Load JSONL data ---
    signals_rows = _load_jsonl(data_path / "signals.jsonl")
    annotations_rows = _load_jsonl(data_path / "annotations.jsonl")
    manifests_rows = _load_jsonl(data_path / "manifests.jsonl")

    # --- Normalize list columns in signals ---
    for row in signals_rows:
        _ensure_list_columns(row)

    # --- Sort for determinism ---
    signals_rows.sort(key=_sort_key_signals)
    annotations_rows.sort(key=_sort_key_annotations)
    manifests_rows.sort(key=_sort_key_manifests)

    # --- Build and write Parquet ---
    signals_table = _build_signals_table(signals_rows)
    annotations_table = _build_generic_table(annotations_rows)
    manifests_table = _build_generic_table(manifests_rows)

    signals_pq = out_path / "signals.parquet"
    annotations_pq = out_path / "annotations.parquet"
    manifests_pq = out_path / "manifests.parquet"

    _write_parquet(signals_table, signals_pq)
    _write_parquet(annotations_table, annotations_pq)
    _write_parquet(manifests_table, manifests_pq)

    # --- Compute checksums ---
    sha256_per_file = {
        "signals.parquet": _sha256_file(signals_pq),
        "annotations.parquet": _sha256_file(annotations_pq),
        "manifests.parquet": _sha256_file(manifests_pq),
    }

    # --- Detect versions ---
    schema_version, taxonomy_version = _detect_versions(data_path)

    # --- Build manifest ---
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "taxonomy_version": taxonomy_version,
        "row_count": {
            "signals": len(signals_rows),
            "annotations": len(annotations_rows),
            "manifests": len(manifests_rows),
        },
        "sha256_per_file": sha256_per_file,
        "source_commit": _git_head_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = out_path / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def refresh_manifests(
    data_dir: str | Path,
) -> Path:
    """Rewrite ``manifests.jsonl`` from signals data in *data_dir*.

    Scans ``signals.jsonl`` and aggregates per-benchmark task counts to
    regenerate ``manifests.jsonl``.

    Parameters
    ----------
    data_dir:
        Directory containing ``signals.jsonl``.

    Returns
    -------
    Path
        Path to the rewritten ``manifests.jsonl``.
    """
    from agent_diagnostics.signals import write_jsonl

    data_path = Path(data_dir)
    signals_rows = _load_jsonl(data_path / "signals.jsonl")

    # Aggregate per-benchmark task counts
    benchmarks: dict[str, set[str]] = {}
    for row in signals_rows:
        bench = row.get("benchmark", "")
        task_id = row.get("task_id", "")
        if bench:
            benchmarks.setdefault(bench, set()).add(task_id)

    now = datetime.now(timezone.utc).isoformat()
    manifest_rows: list[dict[str, Any]] = []
    for i, (bench, tasks) in enumerate(sorted(benchmarks.items()), start=1):
        manifest_rows.append(
            {
                "manifest_id": f"manifest-{i:03d}",
                "benchmark": bench,
                "task_count": len(tasks),
                "created_at": now,
            }
        )

    output_path = data_path / "manifests.jsonl"
    write_jsonl(manifest_rows, output_path)
    return output_path
