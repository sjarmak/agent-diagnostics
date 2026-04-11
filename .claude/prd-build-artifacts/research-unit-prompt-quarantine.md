# Research: unit-prompt-quarantine

## Current State

- `build_prompt` at line 185 of `llm_annotator.py` constructs annotation prompts
- Trajectory content is inserted raw into the prompt between lines 214-221
- No `uuid` import exists; imports are at lines 15-24
- Trajectory section uses markdown code fences (```json) — no untrusted content quarantining
- Existing tests in `test_llm_annotator.py` cover `build_prompt` via `TestBuildPrompt` class (lines 105-175)
- Tests check for `## Trajectory` header — this will need updating since the header changes

## Key Observations

- The trajectory section header "## Trajectory (truncated)" will be replaced by `<untrusted_trajectory>` tags
- Existing test `test_contains_trajectory_section` checks for "## Trajectory" — may break if header removed
- The `test_no_trajectory_shows_placeholder` test checks for "(no trajectory available)" — still valid inside tags
- Signal fields are filtered via REDACTED_SIGNAL_FIELDS — trajectory quarantining is complementary
