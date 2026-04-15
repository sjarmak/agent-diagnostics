# Plan: annotation-store

## Overview

Create `src/agent_diagnostics/annotation_store.py` implementing `AnnotationStore` class with narrow-tall JSONL schema and atomic, locked writes.

## Data Model

- **Narrow-tall JSONL schema**: one row per (trial, category) pair
- **Columns**: trial_id, category_name, confidence, evidence, annotator_type, annotator_identity, taxonomy_version, annotated_at
- **PK**: (trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)

## Implementation Steps

### Step 1: Define exceptions

- `MixedVersionError(Exception)` — raised when incoming taxonomy_version or schema_version differs from existing file data
- `DuplicatePKError(Exception)` — raised when incoming batch contains duplicate PK tuples

### Step 2: AnnotationStore class

- `__init__(self, path: Path)` — store path to annotations.jsonl
- Class constant `PK_FIELDS = ("trial_id", "category_name", "annotator_type", "annotator_identity", "taxonomy_version")`
- Class constant `ROW_FIELDS = ("trial_id", "category_name", "confidence", "evidence", "annotator_type", "annotator_identity", "taxonomy_version", "annotated_at")`

### Step 3: Identity resolution helper

- `_resolve_annotator_identity(raw: str) -> str`
- Try `model_identity.resolve_identity(raw)` — if it returns a value, use it
- If ValueError (unknown snapshot), pass through unchanged (already logical or non-model)
- This means snapshot IDs like "claude-haiku-4-5-20251001" get resolved to "llm:haiku-4"
- Logical IDs like "heuristic:rule-engine" or "llm:haiku-4" pass through unchanged

### Step 4: `_pk_tuple(row)` helper

- Extract the PK fields as a tuple for use as dict key

### Step 5: `upsert_annotations(self, rows, taxonomy_version, schema_version)` method

1. Resolve annotator_identity in each incoming row via \_resolve_annotator_identity
2. Check for duplicate PKs within the batch — raise DuplicatePKError naming the offending key
3. Lock file (fcntl.flock LOCK_EX on a lockfile, `.annotations.jsonl.lock`)
4. If file exists, read existing rows
5. Check taxonomy_version matches — if existing rows have a different version, raise MixedVersionError
6. Also store schema_version in sidecar .meta.json — check it matches if file exists
7. Merge: build dict keyed by PK tuple, existing rows first, then incoming rows (last-writer-wins by annotated_at for matching PKs)
8. Write merged rows to temp file in same directory
9. os.rename temp file over target file
10. Update .meta.json sidecar
11. Unlock
12. Return number of rows written

### Step 6: `read_annotations(self)` method

1. Read JSONL file
2. Deduplicate by PK, keeping row with latest annotated_at
3. Return list of dicts

### Step 7: Atomic write pattern

- Write to `annotations.jsonl.tmp.<pid>` in the same directory
- `os.rename()` is atomic on POSIX for same-filesystem renames
- Clean up temp file on failure

### Step 8: Advisory locking

- Use `fcntl.flock(fd, fcntl.LOCK_EX)` on a lock file
- Document: "Windows note: fcntl is Unix-only. On Windows, use msvcrt.locking or a cross-platform library like filelock."
- Use context manager pattern for lock acquisition/release

## Test Plan (Phase 4)

1. Basic upsert and read-back
2. PK uniqueness enforcement (duplicate in batch)
3. MixedVersionError for taxonomy_version mismatch
4. MixedVersionError for schema_version mismatch
5. Merge behavior with existing file
6. Last-writer-wins dedup by annotated_at
7. Atomic write verification (temp file created, renamed)
8. Identity resolution (snapshot -> logical)
9. Identity passthrough (already logical)
