# Test Results: unit-redaction-constants

## AC1: REDACTED_SIGNAL_FIELDS import check

- **PASS**: `python3 -c "from agent_diagnostics.constants import REDACTED_SIGNAL_FIELDS; ..."` exits 0

## AC2: Contract tests

- **PASS**: `pytest tests/contracts/taxonomy_v3_contract.py -v` — 11 passed

## AC3: build_prompt excludes redacted fields

- **PASS**: New test `test_filters_redacted_signal_fields` verifies reward, passed, exception_info are excluded from prompt output

## AC4: No regressions in test_llm_annotator.py

- **PASS**: `pytest tests/test_llm_annotator.py -v` — 77 passed
- Fixed pre-existing test `test_contains_signals_section` which passed `reward` (now redacted) — updated to use `total_turns` instead
