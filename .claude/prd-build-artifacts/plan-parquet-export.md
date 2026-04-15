# Plan: parquet-export

## Overview

Create `export.py` with `export_parquet(data_dir, out_dir)` that reads JSONL files and writes deterministic Parquet files + MANIFEST.json. Add CLI subcommands `export` and `manifest refresh`.

## Files to Create/Modify

1. **CREATE** `src/agent_diagnostics/export.py` - Core export logic
2. **CREATE** `src/agent_diagnostics/export_manifest_schema.json` - JSON Schema for MANIFEST.json
3. **CREATE** `tests/test_export.py` - Tests
4. **MODIFY** `src/agent_diagnostics/cli.py` - Add `export` and `manifest` subcommands

## Implementation Details

### export.py

- `export_parquet(data_dir, out_dir) -> dict`:
  1. Read signals.jsonl, annotations.jsonl, manifests.jsonl from data_dir via `load_jsonl()`
  2. For signals: ensure list columns (tool_call_sequence, files_read_list, files_edited_list) are Python lists
  3. Sort each dataset deterministically (by trial_id for signals, by task_id+category for annotations, by manifest_id for manifests)
  4. Use pyarrow to build Tables with explicit schemas for list<string> columns
  5. Write with `pq.write_table()` using compression='zstd', write_statistics=False, use_dictionary=True, and a fixed `created_by` string
  6. Compute SHA256 of each output file
  7. Detect schema_version from sidecar .meta.json files
  8. Detect taxonomy_version from sidecar .meta.json files; fail if >1 distinct version found
  9. Get source_commit via `git rev-parse HEAD`
  10. Write MANIFEST.json
  11. Return manifest dict

- `refresh_manifests(data_dir) -> Path`:
  1. Read signals.jsonl from data_dir
  2. Extract manifest-like info (benchmark, task counts, etc.)
  3. Rewrite manifests.jsonl

### Deterministic Parquet

- Sort rows before writing
- `write_statistics=False` to avoid nondeterministic stats
- Fixed `created_by` string to avoid pyarrow version in metadata
- `use_dictionary=True` for consistent encoding

### export_manifest_schema.json

- JSON Schema draft-07 validating MANIFEST.json structure

### CLI additions

- `observatory export --format parquet --out data/export/ --data-dir data/`
- `observatory manifest refresh --data-dir data/`

## Test Plan

1. Basic export produces all 4 output files
2. Parquet files use zstd compression
3. List columns are native list<string>
4. MANIFEST.json validates against schema
5. Re-run produces byte-identical Parquet
6. pandas.read_parquet works
7. CLI integration for cmd_export
