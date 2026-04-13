# Plan: unit-ingest-command

## Steps

1. Add `cmd_ingest(args)` function to cli.py:
   - Accept args: runs_dir, output, manifest (optional), state (optional)
   - Validate runs_dir exists
   - If --manifest provided, call `load_manifest()` to get suite_mapping
   - If --state provided, load existing state file (or start empty)
   - Walk runs_dir with rglob("result.json")
   - For each result.json: check \_is_excluded_path, load JSON, check \_is_valid_trial
   - If state mode: check mtime/size of result.json, skip if unchanged
   - Call extract_signals() with suite_mapping
   - Add trial_path to each signal dict
   - Collect all signals, write with write_jsonl()
   - If state mode: update and save state file
   - Print summary to stderr

2. Register `ingest` subparser in `main()`:
   - --runs-dir (required)
   - --output (required)
   - --manifest (optional)
   - --state (optional)

3. Add tests to test_cli.py:
   - TestCmdIngest class
   - test_basic_ingest: create trial dirs with result.json, run ingest, verify JSONL output
   - test_incremental_skip: run twice, verify state file created, second run skips unchanged
   - test_manifest_flag: provide manifest, verify suite_mapping passed to extract_signals
   - test_state_file_creation: verify state file written with correct structure
   - test_invalid_trial_filtered: create harness summary (no agent_info), verify skipped
   - test_excluded_path_filtered: create trial in excluded dir, verify skipped
