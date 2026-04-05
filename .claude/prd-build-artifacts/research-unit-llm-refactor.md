# Research: unit-llm-refactor

## Duplicated Parsing Logic

Lines 339-365 in `annotate_trial_claude_code` and lines 424-452 in `_annotate_one_claude_code` share identical parsing logic:

1. Parse JSON envelope from stdout
2. Check `is_error` flag
3. Try `structured_output` dict -> `categories`
4. Fallback to `result` raw JSON string -> parse -> extract `categories`
5. Validate categories list type

## \_taxonomy_yaml() Hardcoded Filename

Line 90: `_package_data_path("taxonomy_v1.yaml")` duplicates the default filename from `taxonomy.py` line 57. Should use `load_taxonomy()` or a shared constant.

## taxonomy.py Cache

`_cached_taxonomy` and `_cached_path` are module-level globals. Tests need a cache reset fixture.

## Pure Helpers to Test

- `_read_text(path, max_chars)` - reads file, truncates
- `_load_json(path)` - loads JSON, returns None on error
- `_resolve_model_alias(model)` - maps to short alias
- `_resolve_model_api(model)` - maps to full model ID

## Current Test Count

278 tests across 12 test files. No `tests/conftest.py` exists yet.
