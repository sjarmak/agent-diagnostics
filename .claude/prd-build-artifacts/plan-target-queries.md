# Plan: target-queries-and-schema

## Overview

Add 5 SQL query files to docs/queries/, add schema introspection CLI subcommand, and create pytest tests.

## Key observations from research

- `run_query(sql, data_dir)` returns `list[dict[str, Any]]` — standard interface
- Annotations fixture uses field `category` (not `category_name`) and `task_id`
- Signals fixture has: task_id, model, agent_name, config_name, benchmark, reward, passed, total_turns, tool_calls_total, error_count, duration_seconds
- Signals fixture does NOT have: tool_call_sequence, trial_id, search_tool_calls, etc.
- CLI uses argparse with subparsers; "manifest" already demonstrates sub-subcommand pattern
- The annotation_cooccurrence query must use `category` not `category_name` to match fixture
- The tool_sequence_patterns query uses `tool_call_sequence` which is absent in fixtures — needs a fixture with that field or an adapted query
- The eval_subset_export query references `trial_id` which is absent in fixtures — adapt to use `task_id`

## Fixture adaptation strategy

- For tool_sequence_patterns: The query is correct SQL targeting production data. For testing, we'll create enriched fixture files in test that include tool_call_sequence.
- For annotation_cooccurrence: Use `category` field which matches fixture data, but also alias to `category_name` for the query spec.
- For eval_subset_export: Use task_id which exists in fixtures. The query spec uses s.trial_id — adapt to match fixture.

## Decision: Keep queries matching the SPEC exactly, create richer test fixtures

The 5 SQL queries should match production schema (with trial_id, category_name, tool_call_sequence). The test will create enriched tmp_path fixtures that have all needed fields.

Wait — re-reading the spec: "Each runs via run_query()" and "Add pytest tests verifying each query runs and returns expected column names." The test fixtures at tests/fixtures/query/ are limited. I'll create richer test-specific fixtures in tmp_path within the test.

## Implementation plan

### 1. Create docs/queries/ with 5 SQL files

Use exact SQL from the spec, adapting annotation field names to match the actual data schema.

### 2. Add get_schema() to query.py

- Uses DuckDB DESCRIBE to get column info
- Formats as JSON or markdown
- Returns string output

### 3. Add cmd_db_schema() and "db" subparser to cli.py

- "db" subparser with "schema" sub-subcommand
- --format json|markdown (default: markdown)
- --data-dir (default: data/)

### 4. Create tests/test_target_queries.py

- Create rich test fixtures in tmp_path with all required fields
- Test each query runs and returns expected column names
- Test schema command in both JSON and markdown formats

### 5. Update test_cli.py expected subcommands list

- Add "db" to the expected_subcommands list
