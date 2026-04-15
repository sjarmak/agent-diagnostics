"""Tests for agent_diagnostics.annotation_store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from agent_diagnostics.annotation_store import (
    AnnotationStore,
    DuplicatePKError,
    MixedVersionError,
    _resolve_annotator_identity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    trial_id: str = "trial-001",
    category_name: str = "tool_selection_error",
    confidence: float = 0.9,
    evidence: str = "Agent chose wrong tool",
    annotator_type: str = "llm",
    annotator_identity: str = "llm:haiku-4",
    taxonomy_version: str = "v3",
    annotated_at: str = "2026-04-15T10:00:00+00:00",
    **overrides: object,
) -> dict:
    """Build a single annotation row dict with sensible defaults."""
    row = {
        "trial_id": trial_id,
        "category_name": category_name,
        "confidence": confidence,
        "evidence": evidence,
        "annotator_type": annotator_type,
        "annotator_identity": annotator_identity,
        "taxonomy_version": taxonomy_version,
        "annotated_at": annotated_at,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


class TestResolveAnnotatorIdentity:
    """Tests for the _resolve_annotator_identity helper."""

    def test_snapshot_id_resolved_to_logical(self) -> None:
        result = _resolve_annotator_identity("claude-haiku-4-5-20251001")
        assert result == "llm:haiku-4"

    def test_logical_id_passes_through(self) -> None:
        result = _resolve_annotator_identity("llm:haiku-4")
        assert result == "llm:haiku-4"

    def test_heuristic_identity_passes_through(self) -> None:
        result = _resolve_annotator_identity("heuristic:rule-engine")
        assert result == "heuristic:rule-engine"

    def test_unknown_string_passes_through(self) -> None:
        result = _resolve_annotator_identity("some-custom-annotator")
        assert result == "some-custom-annotator"


# ---------------------------------------------------------------------------
# Basic upsert and read
# ---------------------------------------------------------------------------


class TestBasicUpsertAndRead:
    """Write rows, read them back."""

    def test_upsert_creates_file_and_returns_count(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        rows = [_make_row(), _make_row(category_name="context_window_exceeded")]
        count = store.upsert_annotations(rows, taxonomy_version="v3")

        assert count == 2
        assert store.path.is_file()

    def test_read_returns_upserted_rows(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        rows = [_make_row(), _make_row(category_name="context_window_exceeded")]
        store.upsert_annotations(rows, taxonomy_version="v3")

        result = store.read_annotations()
        assert len(result) == 2

        names = {r["category_name"] for r in result}
        assert names == {"tool_selection_error", "context_window_exceeded"}

    def test_read_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "nonexistent.jsonl")
        assert store.read_annotations() == []

    def test_rows_written_as_jsonl(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        lines = store.path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["trial_id"] == "trial-001"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "deep" / "nested" / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")
        assert store.path.is_file()


# ---------------------------------------------------------------------------
# PK uniqueness enforcement
# ---------------------------------------------------------------------------


class TestPKUniqueness:
    """Duplicate PK within a single batch raises DuplicatePKError."""

    def test_duplicate_pk_in_batch_raises(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        rows = [
            _make_row(annotated_at="2026-04-15T10:00:00+00:00"),
            _make_row(annotated_at="2026-04-15T11:00:00+00:00"),  # same PK
        ]
        with pytest.raises(DuplicatePKError, match="Duplicate primary key"):
            store.upsert_annotations(rows, taxonomy_version="v3")

    def test_error_names_offending_key(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        rows = [_make_row(), _make_row()]
        with pytest.raises(DuplicatePKError, match="trial-001"):
            store.upsert_annotations(rows, taxonomy_version="v3")

    def test_different_pks_accepted(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        rows = [
            _make_row(category_name="tool_selection_error"),
            _make_row(category_name="context_window_exceeded"),
        ]
        count = store.upsert_annotations(rows, taxonomy_version="v3")
        assert count == 2


# ---------------------------------------------------------------------------
# Mixed version refusal
# ---------------------------------------------------------------------------


class TestMixedVersionRefusal:
    """Writing a different taxonomy_version or schema_version raises MixedVersionError."""

    def test_taxonomy_version_mismatch_raises(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        new_row = _make_row(
            trial_id="trial-002",
            taxonomy_version="v4",
            annotated_at="2026-04-15T12:00:00+00:00",
        )
        with pytest.raises(MixedVersionError, match="v4.*v3|v3.*v4"):
            store.upsert_annotations([new_row], taxonomy_version="v4")

    def test_error_names_both_versions(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        new_row = _make_row(trial_id="trial-002", taxonomy_version="v4")
        with pytest.raises(MixedVersionError) as exc_info:
            store.upsert_annotations([new_row], taxonomy_version="v4")

        msg = str(exc_info.value)
        assert "v3" in msg
        assert "v4" in msg

    def test_schema_version_mismatch_raises(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations(
            [_make_row()],
            taxonomy_version="v3",
            schema_version="observatory-annotation-v1",
        )

        new_row = _make_row(trial_id="trial-002")
        with pytest.raises(MixedVersionError, match="schema_version"):
            store.upsert_annotations(
                [new_row],
                taxonomy_version="v3",
                schema_version="observatory-annotation-v2",
            )

    def test_same_version_accepted(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        new_row = _make_row(
            trial_id="trial-002",
            annotated_at="2026-04-15T12:00:00+00:00",
        )
        count = store.upsert_annotations([new_row], taxonomy_version="v3")
        assert count == 2


# ---------------------------------------------------------------------------
# Merge behavior
# ---------------------------------------------------------------------------


class TestMergeBehavior:
    """Upsert with existing file merges correctly."""

    def test_new_pk_is_appended(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        new_row = _make_row(
            trial_id="trial-002",
            annotated_at="2026-04-15T12:00:00+00:00",
        )
        count = store.upsert_annotations([new_row], taxonomy_version="v3")
        assert count == 2

        result = store.read_annotations()
        trial_ids = {r["trial_id"] for r in result}
        assert trial_ids == {"trial-001", "trial-002"}

    def test_matching_pk_keeps_latest_annotated_at(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        old_row = _make_row(
            confidence=0.5,
            annotated_at="2026-04-15T10:00:00+00:00",
        )
        store.upsert_annotations([old_row], taxonomy_version="v3")

        new_row = _make_row(
            confidence=0.95,
            annotated_at="2026-04-15T12:00:00+00:00",
        )
        store.upsert_annotations([new_row], taxonomy_version="v3")

        result = store.read_annotations()
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95

    def test_matching_pk_keeps_existing_when_older_incoming(
        self, tmp_path: Path
    ) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        old_row = _make_row(
            confidence=0.95,
            annotated_at="2026-04-15T12:00:00+00:00",
        )
        store.upsert_annotations([old_row], taxonomy_version="v3")

        older_row = _make_row(
            confidence=0.5,
            annotated_at="2026-04-15T08:00:00+00:00",
        )
        store.upsert_annotations([older_row], taxonomy_version="v3")

        result = store.read_annotations()
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Last-writer-wins dedup
# ---------------------------------------------------------------------------


class TestLastWriterWinsDedup:
    """read_annotations() deduplicates by PK, keeping latest annotated_at."""

    def test_dedup_keeps_latest(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")

        # Manually write duplicate PKs to the file (simulating concurrent appends)
        row_old = _make_row(confidence=0.5, annotated_at="2026-04-15T08:00:00+00:00")
        row_new = _make_row(confidence=0.99, annotated_at="2026-04-15T16:00:00+00:00")

        store.path.parent.mkdir(parents=True, exist_ok=True)
        with store.path.open("w") as fh:
            fh.write(json.dumps(row_old) + "\n")
            fh.write(json.dumps(row_new) + "\n")

        result = store.read_annotations()
        assert len(result) == 1
        assert result[0]["confidence"] == 0.99

    def test_dedup_preserves_distinct_pks(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")

        row_a = _make_row(category_name="cat_a")
        row_b = _make_row(category_name="cat_b")

        store.path.parent.mkdir(parents=True, exist_ok=True)
        with store.path.open("w") as fh:
            fh.write(json.dumps(row_a) + "\n")
            fh.write(json.dumps(row_b) + "\n")

        result = store.read_annotations()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Writes use temp-file + os.rename pattern."""

    def test_no_partial_file_on_success(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        # No leftover temp files
        tmp_files = list(tmp_path.glob(".*tmp*"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"

    def test_meta_sidecar_created(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        meta_path = tmp_path / "annotations.meta.json"
        assert meta_path.is_file()

        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == "observatory-annotation-v1"
        assert meta["taxonomy_version"] == "v3"
        assert "generated_at" in meta

    def test_rename_used_instead_of_direct_write(self, tmp_path: Path) -> None:
        """Verify os.rename is called during upsert (atomic swap)."""
        store = AnnotationStore(tmp_path / "annotations.jsonl")

        original_rename = os.rename
        rename_calls: list[tuple[str, str]] = []

        def tracking_rename(src: str, dst: str) -> None:
            rename_calls.append((src, dst))
            return original_rename(src, dst)

        with mock.patch("os.rename", side_effect=tracking_rename):
            store.upsert_annotations([_make_row()], taxonomy_version="v3")

        # At least one rename should target the annotations.jsonl file
        jsonl_renames = [
            (s, d) for s, d in rename_calls if d.endswith("annotations.jsonl")
        ]
        assert len(jsonl_renames) >= 1


# ---------------------------------------------------------------------------
# Identity resolution via model_identity
# ---------------------------------------------------------------------------


class TestIdentityResolution:
    """annotator_identity is resolved via model_identity — snapshot IDs never stored raw."""

    def test_snapshot_id_resolved_on_upsert(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        row = _make_row(annotator_identity="claude-haiku-4-5-20251001")
        store.upsert_annotations([row], taxonomy_version="v3")

        result = store.read_annotations()
        assert len(result) == 1
        assert result[0]["annotator_identity"] == "llm:haiku-4"

    def test_logical_id_preserved_on_upsert(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        row = _make_row(annotator_identity="llm:haiku-4")
        store.upsert_annotations([row], taxonomy_version="v3")

        result = store.read_annotations()
        assert result[0]["annotator_identity"] == "llm:haiku-4"

    def test_heuristic_identity_preserved(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        row = _make_row(
            annotator_type="heuristic",
            annotator_identity="heuristic:rule-engine",
        )
        store.upsert_annotations([row], taxonomy_version="v3")

        result = store.read_annotations()
        assert result[0]["annotator_identity"] == "heuristic:rule-engine"

    def test_raw_snapshot_never_in_jsonl(self, tmp_path: Path) -> None:
        """Ensure the raw snapshot ID is not written to disk."""
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        row = _make_row(annotator_identity="claude-haiku-4-5-20251001")
        store.upsert_annotations([row], taxonomy_version="v3")

        raw_content = store.path.read_text()
        assert "claude-haiku-4-5-20251001" not in raw_content
        assert "llm:haiku-4" in raw_content


# ---------------------------------------------------------------------------
# Advisory locking
# ---------------------------------------------------------------------------


class TestAdvisoryLocking:
    """Lock file is created during write operations."""

    def test_lock_file_exists_after_upsert(self, tmp_path: Path) -> None:
        store = AnnotationStore(tmp_path / "annotations.jsonl")
        store.upsert_annotations([_make_row()], taxonomy_version="v3")

        lock_file = tmp_path / ".annotations.jsonl.lock"
        assert lock_file.is_file()
