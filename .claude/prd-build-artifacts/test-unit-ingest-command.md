# Test Results: unit-ingest-command

## Run

```
python3 -m pytest tests/test_cli.py -v
38 passed, 6 warnings in 0.28s
```

## Tests Added (9 tests in TestCmdIngest)

| Test                                | Status | Acceptance Criteria                            |
| ----------------------------------- | ------ | ---------------------------------------------- |
| test_missing_runs_dir_exits         | PASS   | Error handling                                 |
| test_basic_ingest                   | PASS   | Writes JSONL with one trial per line           |
| test_invalid_trial_filtered         | PASS   | Uses \_is_valid_trial to filter                |
| test_excluded_path_filtered         | PASS   | Uses \_is_excluded_path to filter              |
| test_manifest_flag                  | PASS   | --manifest flag works for benchmark resolution |
| test_manifest_missing_exits         | PASS   | Error handling for missing manifest            |
| test_state_file_creation            | PASS   | --state creates state JSON with mtime/size     |
| test_incremental_skip               | PASS   | Second run skips unchanged trials              |
| test_incremental_reextracts_changed | PASS   | Modified trials are re-extracted               |

## Existing Tests

All 29 pre-existing tests continue to pass, including the updated subcommand help test.

## Warnings

6 UserWarnings from extract_signals about missing suite_mapping (expected for tests without --manifest).
