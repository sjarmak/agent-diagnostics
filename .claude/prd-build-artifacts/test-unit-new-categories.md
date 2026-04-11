# Test Results: unit-new-categories

## Acceptance Criteria

| #   | Criterion                                                                                          | Result                                    |
| --- | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| AC1 | `find tests/fixtures/trials ...` returns >= 12                                                     | PASS (12)                                 |
| AC2 | `pytest tests/test_annotator.py -v -k "reward_hacking or ..."` exits 0 with >= 1 test per category | PASS (20 tests, all 6 categories covered) |
| AC3 | `python3 -c "...load_taxonomy('taxonomy_v3.yaml')...assert all 6 names..."` exits 0                | PASS                                      |

## Full Suite

- 543 tests passed, 0 failed
- No regressions introduced

## Changes Made

- taxonomy_v3.yaml: Added 3 new categories (tool_argument_error, premature_termination, verification_skipped)
- annotator.py: Added 3 heuristic checkers, updated to validate against v3 taxonomy
- test_annotator.py: Added 6 test classes (20 new tests)
- test_e2e_golden.py: Updated to validate against v3 taxonomy (consistent with annotator)
- test_taxonomy_schema.py: Relaxed non-derived signal_dependencies constraint (non-derived categories can have hint deps)
- 12 new fixture directories with expected.json and trajectory.json
