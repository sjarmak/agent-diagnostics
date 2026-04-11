# Test Results: unit-prompt-quarantine

## Acceptance Criteria

| #   | Criterion                                                                    | Result       |
| --- | ---------------------------------------------------------------------------- | ------------ |
| 1   | `pytest tests/test_prompt_injection.py -v` exits 0                           | PASS (6/6)   |
| 2   | `pytest tests/test_prompt_injection.py::test_delimiter_breakout -v` exits 0  | PASS         |
| 3   | `grep -c 'untrusted_trajectory' src/agent_diagnostics/llm_annotator.py` >= 2 | PASS (3)     |
| 4   | `pytest tests/test_llm_annotator.py -v` exits 0 (no regressions)             | PASS (77/77) |

## Test Summary

- `test_quarantine_boundary_present` — verifies UUID-nonce `<untrusted_trajectory>` open/close tags
- `test_empty_trajectory_also_quarantined` — empty trajectory still wrapped in tags
- `test_system_instruction_mentions_ignore` — directive appears before quarantine zone
- `test_directive_mentions_adversarial` — "adversarial" mentioned in directive
- `test_injection_attempt_quarantined` — malicious payload contained inside quarantine boundaries
- `test_delimiter_breakout` — attacker's fake nonce differs from real UUID nonce
