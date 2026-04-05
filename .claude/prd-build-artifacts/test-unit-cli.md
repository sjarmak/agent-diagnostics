# Test Results: unit-cli

## tests/test_cli.py — 6/6 passed

- test_main_importable: PASSED
- test_help_output_includes_all_subcommands: PASSED (all 8 subcommands present)
- test_no_judge_flag_in_help: PASSED (--judge absent from llm-annotate)
- test_cli_imports_use_agent_observatory: PASSED (no bare observatory imports)
- test_dunder_main_imports_from_agent_observatory: PASSED
- test_pyproject_has_scripts_entry: PASSED

## Full suite — 278/278 passed in 0.40s

No regressions introduced.
