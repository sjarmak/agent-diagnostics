# Research: unit-ingest-command

## CLI Structure

- `cli.py` uses argparse with subparsers, `main()` registers all subcommands
- Each command is a `cmd_<name>(args)` function
- Pattern: validate inputs -> import needed modules -> call business logic -> write output -> print summary to stderr

## Key signals.py Functions

- `_is_valid_trial(data)`: checks `"agent_info" in data`
- `_is_excluded_path(trial_dir)`: checks path parts against excluded patterns
- `extract_signals(trial_dir, *, suite_mapping, ...)`: extracts signals from one trial dir
- `load_manifest(path)`: loads manifest JSON -> dict[str, str] mapping run dirs to benchmarks
- `write_jsonl(data_list, path)`: writes JSONL + sidecar .meta.json
- `_load_json(path)`: safe JSON loader returning None on error
- `extract_all()`: walks dirs with rglob("result.json"), filters, extracts - this is the pattern to follow

## State File Design

- Track `{trial_path: {mtime: float, size: int}}` in a JSON file
- On each run, compare current mtime/size of result.json against stored values
- Skip extraction if unchanged
