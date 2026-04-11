# Plan: unit-redaction-constants

## Step 1: Create src/agent_diagnostics/constants.py

- Define `REDACTED_SIGNAL_FIELDS: frozenset[str]` containing `{"reward", "passed", "exception_info"}`
- Keep file minimal — just the constant

## Step 2: Modify src/agent_diagnostics/llm_annotator.py

- Add import: `from agent_diagnostics.constants import REDACTED_SIGNAL_FIELDS`
- Modify build_prompt line 223-227: add REDACTED_SIGNAL_FIELDS to the exclusion set in the dict comprehension

## Step 3: Modify src/agent_diagnostics/signals.py

- Add import: `from agent_diagnostics.constants import REDACTED_SIGNAL_FIELDS`
- No logic changes

## Step 4: Create tests/contracts/**init**.py (empty)

## Step 5: Create tests/contracts/taxonomy_v3_contract.py

- Contract: signal_dependencies format must be flat string names (not dotted paths)
- Contract: valid_category_names() must return Set[str]
- Contract: fixture schema — fixture dirs must contain expected.json and trajectory.json

## Step 6: Run all tests

- pytest tests/contracts/taxonomy_v3_contract.py -v
- pytest tests/test_llm_annotator.py -v
- Acceptance criterion 1: import check
- Fix any failures
