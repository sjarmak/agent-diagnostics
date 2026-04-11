# Plan: unit-taxonomy-v3

## Step 1: Fix validate_categories bug (MH-1)

- In `src/agent_diagnostics/llm_annotator.py` line 252, replace:
  `taxonomy_names = {cat["name"] for cat in load_taxonomy()["categories"]}`
  with:
  ```python
  from agent_diagnostics.taxonomy import valid_category_names
  taxonomy_names = valid_category_names()
  ```
- Remove unused `load_taxonomy` import at top of file (line 28) if no other usage exists. Check first.

## Step 2: Create taxonomy_v3.yaml (MH-2)

- Use v2 dimensions format
- Keep ALL existing v2 categories with new fields added
- Move wrong_tool_choice from Retrieval to new ToolUse dimension
- Add new dimensions: ToolUse, Faithfulness, Metacognition, Integrity, Safety
- Add premature_commit and clean_success (mentioned in spec scope)
- Every category gets: severity, derived_from_signal, signal_dependencies
- Mark incomplete_solution, near_miss, minimal_progress as derived_from_signal: true

## Step 3: Update taxonomy.py

- Add `_is_v3()` helper checking version == "3.0.0"
- Ensure \_extract_categories works with v3 (it should since v3 uses dimensions format)
- No other changes needed since \_extract_categories preserves all fields

## Step 4: Write tests

- `tests/test_llm_annotator_taxonomy_compat.py`: validate_categories works with v1 and v2 formats via mocking
- `tests/test_taxonomy_schema.py`: validate v3 schema has all required fields

## Step 5: Run all tests, fix failures
