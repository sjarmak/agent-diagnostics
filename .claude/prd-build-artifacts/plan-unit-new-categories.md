# Plan: unit-new-categories

## Step 1: Add missing categories to taxonomy_v3.yaml

- Add `tool_argument_error` to ToolUse dimension
- Add `premature_termination` to Metacognition dimension
- Add `verification_skipped` to Metacognition dimension

## Step 2: Add heuristic checkers to annotator.py

- `_check_tool_argument_error`: error_count high relative to tool_calls_total
- `_check_premature_termination`: total_turns < 3 with reward < 1.0
- `_check_verification_skipped`: edit_tool_calls > 0 but no test/verify in tool_call_sequence
- Register all 3 in `_ALL_CHECKERS` tuple

## Step 3: Create 12 fixture directories

- 2 per category, each with expected.json and trajectory.json

## Step 4: Add tests to test_annotator.py

- Test class per new category with signal-based detection tests
- Tests for the 3 heuristic-detectable categories must verify annotate_trial returns them
- Tests for reward_hacking, fabricated_success, hallucinated_api verify they exist in taxonomy

## Step 5: Run tests and verify acceptance criteria
