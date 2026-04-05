# Test Results: unit-cli-tests

## Coverage

- cli.py: 99% (215 statements, 2 missed)
- Target: >= 70% — PASSED

## Tests: 25 passed, 0 failed

- All 8+ cmd\_\* functions tested (cmd_extract, cmd_annotate, cmd_report, cmd_llm_annotate, cmd_train, cmd_predict, cmd_ensemble, cmd_validate)
- Error paths: missing file, invalid JSON, no trajectories, schema errors, unknown categories
- cmd_llm_annotate: mocked annotate_trial_llm and load_taxonomy
- cmd_validate: mocked jsonschema.validate and valid_category_names
- All 345 existing tests still pass
