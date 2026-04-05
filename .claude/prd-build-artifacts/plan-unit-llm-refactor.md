# Plan: unit-llm-refactor

## Steps

1. Create 5+ fixture files in `tests/fixtures/claude_envelope_*.json`
2. Write contract tests for parsing behavior BEFORE extraction
3. Run contract tests to verify they pass against current inline code
4. Extract `_parse_claude_response()` helper in `llm_annotator.py`
5. Replace inline parsing in `annotate_trial_claude_code` and `_annotate_one_claude_code`
6. Fix `_taxonomy_yaml()` to use `load_taxonomy()` instead of hardcoded filename
7. Add `TAXONOMY_FILENAME` constant to `taxonomy.py` and use it
8. Create `tests/conftest.py` with taxonomy fixtures and cache reset
9. Write tests for pure helpers (`_read_text`, `_load_json`, `_resolve_model_alias`, `_resolve_model_api`)
10. Run all tests, fix failures
11. Verify line count decreased
