# Plan: unit-prompt-quarantine

## Steps

1. Add `import uuid` to imports in `llm_annotator.py`
2. Generate `nonce = str(uuid.uuid4())` at start of `build_prompt`
3. Add quarantine directive to the system instruction (first `parts.append`)
4. Replace trajectory section: remove markdown header + code fences, wrap with `<untrusted_trajectory boundary="...">` tags
5. Update existing test `test_contains_trajectory_section` to check for new tag format
6. Create `tests/test_prompt_injection.py` with 4 tests:
   - test_quarantine_boundary_present
   - test_system_instruction_mentions_ignore
   - test_adversarial_trajectory_quarantined
   - test_delimiter_breakout
7. Run all tests, fix any regressions
