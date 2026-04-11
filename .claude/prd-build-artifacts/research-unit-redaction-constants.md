# Research: unit-redaction-constants

## Codebase Findings

### llm_annotator.py (line 223-227)

- `build_prompt` filters signals at line 223-227: excludes `tool_calls_by_name` and `trial_path`, and `None` values
- Need to ADD REDACTED_SIGNAL_FIELDS to the exclusion set

### signals.py

- Imports from `agent_diagnostics.tool_registry` and `agent_diagnostics.types`
- Extracts `reward`, `passed`, and `exception_info` (via `exception_crashed`) — these are the fields that leak ground truth into LLM prompts

### taxonomy.py

- `valid_category_names()` returns `Set[str]`
- `load_taxonomy()` returns dict with `categories` key
- Supports v1 and v2 formats

### types.py

- `TrialSignals` is a TypedDict with 26 keys including `reward` and `passed`
- No `exception_info` key in TrialSignals — but `exception_crashed` exists as a bool

### Test patterns

- Tests use pytest, `tmp_path`, fixtures from conftest.py
- FIXTURES_DIR pattern used in multiple test files
- conftest.py provides taxonomy fixtures

### Constants file

- No existing constants.py — will create new
- Should be a frozenset for immutability (per coding style rules)

### Contracts test dir

- No existing tests/contracts/ directory — will create
