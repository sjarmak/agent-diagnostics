# Plan: unit-jsonl-format

## Approach

### 1. Add IO helpers to signals.py

- `write_jsonl(data_list, path)`: writes one JSON object per line to `path`, plus a `.meta.json` sidecar with `schema_version`, `taxonomy_version`, `generated_at`
- `write_output(data, path)`: detects extension, delegates to json.dump or write_jsonl
- `load_signals(path)`: detects `.jsonl` vs `.json`, reads accordingly, returns list[dict]
- `load_annotations(path)`: reads annotations from .json or .jsonl (reconstructs envelope from sidecar)

### 2. Update cmd_extract in cli.py

- Replace `json.dump` with `write_output` that checks `output.suffix`
- `.json` -> legacy json.dump behavior
- `.jsonl` -> write_jsonl + meta sidecar

### 3. Update cmd_annotate in cli.py

- Input: use `load_signals` or `load_annotations` depending on format
- Output: same format detection as extract
- For .jsonl annotations output: write one annotation per line + meta sidecar

### 4. Tests

- JSONL write/read roundtrip for signals
- JSONL write/read roundtrip for annotations
- Meta sidecar content validation
- Format detection (extension-based)
- Both .json and .jsonl paths work for extract and annotate
