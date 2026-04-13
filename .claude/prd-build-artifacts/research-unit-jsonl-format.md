# Research: unit-jsonl-format

## Current State

### cli.py

- `cmd_extract`: calls `extract_all(runs_dir)` returning `list[TrialSignals]`, writes with `json.dump(signals, f, indent=2, default=str)`
- `cmd_annotate`: reads signals via `json.load(f)`, calls `annotate_all(signals_list)`, writes result with `json.dump(annotations, f, indent=2, default=str)`. Result is `{"annotations": [...]}`
- No format detection logic exists

### signals.py

- `extract_all()` returns `list[TrialSignals]`
- `extract_signals()` returns a single `TrialSignals` dict
- No read/write helpers for serialization formats
- Has `_load_json()` internal helper

### test_cli.py

- Uses `argparse.Namespace` to construct args
- Mocks `extract_all` and `annotate_all`
- Tests check output file exists and content via `json.loads`

### Annotation schema

- `schema_version`: "observatory-annotation-v1"
- `taxonomy_version`: from taxonomy["version"]
- These will go in the `.meta.json` sidecar for JSONL

## Key Observations

- `annotate_all` is lazily imported but not defined in annotator.py — tests mock it
- Signals are a flat list; annotations are a dict with top-level metadata + annotations list
- For JSONL annotations, need to split: per-line annotation objects, metadata in sidecar
