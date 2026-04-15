"""Narrow-tall JSONL annotation store with atomic writes and advisory locking.

Stores annotation rows in a JSONL file (one JSON object per line) using a
narrow-tall schema: each row represents a single (trial, category, annotator)
combination rather than nesting categories inside a trial record.

Columns
-------
trial_id, category_name, confidence, evidence, annotator_type,
annotator_identity, taxonomy_version, annotated_at

Primary key
-----------
(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)

Concurrency
-----------
Writes acquire an advisory file lock (``fcntl.flock`` on Unix) around the
entire read-merge-write cycle so that concurrent processes do not corrupt the
file.

.. note::

   **Windows note:** ``fcntl`` is Unix-only. On Windows, replace with
   ``msvcrt.locking`` or a cross-platform library such as ``filelock``.
   The rest of the logic (temp-file + ``os.rename``) works on both platforms.

Atomic writes
-------------
Data is written to a temporary file in the same directory and then atomically
renamed over the target via ``os.rename``.  This guarantees that readers never
see a partially-written file.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agent_diagnostics import model_identity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MixedVersionError(Exception):
    """Raised when incoming version differs from existing data in the store."""


class DuplicatePKError(Exception):
    """Raised when a single batch contains rows with duplicate primary keys."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PK_FIELDS: tuple[str, ...] = (
    "trial_id",
    "category_name",
    "annotator_type",
    "annotator_identity",
    "taxonomy_version",
)

ROW_FIELDS: tuple[str, ...] = (
    "trial_id",
    "category_name",
    "confidence",
    "evidence",
    "annotator_type",
    "annotator_identity",
    "taxonomy_version",
    "annotated_at",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pk_tuple(row: dict[str, Any]) -> tuple[str, ...]:
    """Extract the primary-key fields from *row* as an immutable tuple."""
    return tuple(str(row.get(f, "")) for f in PK_FIELDS)


def _resolve_annotator_identity(raw: str) -> str:
    """Resolve a raw annotator identity to its logical form.

    If *raw* is a known Anthropic snapshot ID (e.g.
    ``"claude-haiku-4-5-20251001"``), it is resolved to the logical identity
    (e.g. ``"llm:haiku-4"``).  Otherwise it is returned unchanged — this
    covers logical IDs like ``"heuristic:rule-engine"`` and ``"llm:haiku-4"``
    that are already in canonical form.
    """
    try:
        return model_identity.resolve_identity(raw)
    except ValueError:
        return raw


def _parse_annotated_at(value: str | None) -> datetime:
    """Parse an ISO-8601 timestamp, falling back to epoch on failure."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _meta_path(path: Path) -> Path:
    """Return the sidecar ``.meta.json`` path for a JSONL file."""
    return path.with_suffix(".meta.json")


def _lock_path(path: Path) -> Path:
    """Return the advisory lock-file path for a JSONL file."""
    return path.parent / f".{path.name}.lock"


# ---------------------------------------------------------------------------
# AnnotationStore
# ---------------------------------------------------------------------------


class AnnotationStore:
    """Narrow-tall JSONL annotation store with atomic, locked writes.

    Parameters
    ----------
    path:
        Filesystem path to the ``annotations.jsonl`` file.  Parent directories
        are created automatically on the first write.
    """

    PK_FIELDS = PK_FIELDS
    ROW_FIELDS = ROW_FIELDS

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_annotations(
        self,
        rows: Sequence[dict[str, Any]],
        taxonomy_version: str,
        schema_version: str = "observatory-annotation-v1",
    ) -> int:
        """Merge *rows* into the store, enforcing PK uniqueness and version checks.

        Each dict in *rows* must contain at least the fields listed in
        :data:`ROW_FIELDS`.

        Returns the total number of rows written to the file after merging.

        Raises
        ------
        DuplicatePKError
            If *rows* contains two or more entries sharing the same primary key.
        MixedVersionError
            If the file already contains rows with a different
            ``taxonomy_version``, or if the sidecar records a different
            ``schema_version``.
        """
        # --- 1. Resolve annotator_identity in incoming rows ---
        resolved_rows: list[dict[str, Any]] = []
        for row in rows:
            resolved = dict(row)
            if "annotator_identity" in resolved:
                resolved["annotator_identity"] = _resolve_annotator_identity(
                    resolved["annotator_identity"]
                )
            resolved_rows.append(resolved)

        # --- 2. Check for duplicate PKs within the incoming batch ---
        seen_pks: dict[tuple[str, ...], int] = {}
        for idx, row in enumerate(resolved_rows):
            pk = _pk_tuple(row)
            if pk in seen_pks:
                raise DuplicatePKError(
                    f"Duplicate primary key in batch at indices {seen_pks[pk]} and {idx}: "
                    f"PK={dict(zip(PK_FIELDS, pk))}"
                )
            seen_pks[pk] = idx

        # --- 3. Acquire advisory lock and perform read-merge-write ---
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = _lock_path(self.path)

        lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
        try:
            # Advisory lock — blocks until exclusive access is granted.
            # Windows note: fcntl is Unix-only. On Windows, use
            # msvcrt.locking() or a cross-platform library like filelock.
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            existing_rows: list[dict[str, Any]] = []
            if self.path.is_file():
                existing_rows = _read_jsonl(self.path)

            # --- 4. Version checks against existing data ---
            if existing_rows:
                existing_versions = {r.get("taxonomy_version") for r in existing_rows}
                for ev in existing_versions:
                    if ev and ev != taxonomy_version:
                        raise MixedVersionError(
                            f"Incoming taxonomy_version={taxonomy_version!r} "
                            f"conflicts with existing version={ev!r} in {self.path}"
                        )

            meta_file = _meta_path(self.path)
            if meta_file.is_file():
                with meta_file.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                existing_sv = meta.get("schema_version", "")
                if existing_sv and existing_sv != schema_version:
                    raise MixedVersionError(
                        f"Incoming schema_version={schema_version!r} "
                        f"conflicts with existing schema_version={existing_sv!r} "
                        f"in {meta_file}"
                    )

            # --- 5. Merge: existing rows first, then incoming rows ---
            # For matching PKs, keep the row with the latest annotated_at.
            merged: dict[tuple[str, ...], dict[str, Any]] = {}

            for row in existing_rows:
                pk = _pk_tuple(row)
                merged[pk] = row

            for row in resolved_rows:
                pk = _pk_tuple(row)
                if pk in merged:
                    existing_ts = _parse_annotated_at(merged[pk].get("annotated_at"))
                    incoming_ts = _parse_annotated_at(row.get("annotated_at"))
                    if incoming_ts >= existing_ts:
                        merged[pk] = row
                else:
                    merged[pk] = row

            merged_rows = list(merged.values())

            # --- 6. Atomic write: temp file + os.rename ---
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.tmp.",
                suffix=".jsonl",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for row in merged_rows:
                        fh.write(json.dumps(row, default=str) + "\n")
                os.rename(tmp_path, str(self.path))
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # --- 7. Write sidecar meta ---
            meta_data = {
                "schema_version": schema_version,
                "taxonomy_version": taxonomy_version,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_fd, meta_tmp = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix=f".{meta_file.name}.tmp.",
                suffix=".json",
            )
            try:
                with os.fdopen(meta_fd, "w", encoding="utf-8") as fh:
                    json.dump(meta_data, fh, indent=2)
                os.rename(meta_tmp, str(meta_file))
            except BaseException:
                try:
                    os.unlink(meta_tmp)
                except OSError:
                    pass
                raise

            return len(merged_rows)

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def read_annotations(self) -> list[dict[str, Any]]:
        """Read annotations, deduplicating by PK (last-writer-wins by annotated_at).

        Returns an empty list if the file does not exist.
        """
        if not self.path.is_file():
            return []

        rows = _read_jsonl(self.path)

        # Deduplicate: for each PK, keep the row with the latest annotated_at
        best: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in rows:
            pk = _pk_tuple(row)
            if pk in best:
                existing_ts = _parse_annotated_at(best[pk].get("annotated_at"))
                incoming_ts = _parse_annotated_at(row.get("annotated_at"))
                if incoming_ts >= existing_ts:
                    best[pk] = row
            else:
                best[pk] = row

        return list(best.values())
