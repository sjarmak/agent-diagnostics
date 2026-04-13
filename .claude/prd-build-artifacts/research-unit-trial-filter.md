# Research: unit-trial-filter

## Findings

- `signals.py` has `extract_all()` at line 469 that walks `root_dir.rglob("result.json")` and calls `extract_signals()` per trial dir
- No existing filtering/validation of trial directories
- `_load_json()` helper exists for loading JSON files
- The `extract_signals()` function already loads `result.json` via `_load_json()`
- `_is_valid_trial(data)` needs to check for `agent_info` key in result.json data (harness summaries lack this)
- Excluded directory patterns: `__archived_invalid`, `__incomplete`, `__pre_sgenv_fix`, `__verifier_path_bug`, `__doubled_prefix`
- These are path-segment patterns to match against any part of the trial directory path
- Test file follows class-based organization with helpers at top
