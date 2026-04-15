# Plan: duckdb-query

## Steps

1. Create `tests/fixtures/query/` directory with fixture JSONL files:
   - `signals.jsonl` — 5 rows with realistic signal fields
   - `annotations.jsonl` — 5 rows in narrow-tall format
   - `manifests.jsonl` — 2 rows with manifest metadata

2. Create `src/agent_diagnostics/query.py`:
   - `run_query(sql: str, data_dir: str | Path = "data/") -> list[dict]`
     - Create DuckDB in-memory connection
     - For each table in ("signals", "annotations", "manifests"):
       - Check `data_dir/export/{name}.parquet` -> create view via `read_parquet()`
       - Else check `data_dir/{name}.jsonl` -> create view via `read_json_auto()`
       - Skip if neither exists
     - Execute SQL, fetch results, return as list[dict]
   - `format_table(rows: list[dict]) -> str` — ASCII table formatter

3. Update `src/agent_diagnostics/cli.py`:
   - Add `cmd_query(args)` function calling `run_query` and printing formatted output
   - Add `query` subparser with positional `sql` arg and optional `--data-dir`

4. Create `tests/test_query.py` with tests:
   - test_run_query_basic_select
   - test_auto_register_jsonl
   - test_auto_register_parquet (write parquet in test via duckdb)
   - test_parquet_preferred_over_jsonl
   - test_join_across_tables
   - test_invalid_sql_raises
   - test_missing_table_skipped
   - test_cmd_query_integration
   - test_help_includes_query_subcommand

5. Run tests, fix any failures.

6. Commit.
