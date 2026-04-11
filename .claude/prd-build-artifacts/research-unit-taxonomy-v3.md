# Research: unit-taxonomy-v3

## Bug (MH-1)

- `llm_annotator.py:252` has `taxonomy_names = {cat["name"] for cat in load_taxonomy()["categories"]}` which directly accesses `["categories"]` — only works for v1 format.
- `taxonomy.py` already provides `valid_category_names()` which handles both v1 and v2 via `_extract_categories()`.
- Fix: replace line 252 with call to `valid_category_names()`.

## Taxonomy v3 (MH-2)

- v2 format uses `dimensions` list with nested `categories`.
- `_extract_categories()` already handles v2 format by iterating `taxonomy["dimensions"]` and collecting categories.
- `_is_v2()` checks for `"dimensions"` key — v3 also uses dimensions, so it will pass.
- Need v3-specific detection: check `version == "3.0.0"`.
- New fields per category: severity, derived_from_signal, signal_dependencies.
- `_extract_categories()` returns full category dicts, so new fields are preserved automatically.

## Existing v2 categories to keep (14 total)

retrieval_failure, query_churn, wrong_tool_choice, over_exploration, edit_verify_loop_failure, incomplete_solution, near_miss, minimal_progress, rate_limited_run, exception_crash, stale_context, decomposition_failure, premature_commit, clean_success

Note: premature_commit and clean_success are listed in scope but NOT in v2 yaml. The spec says "keep ALL existing v2 categories" — v2 has: retrieval_failure, query_churn, wrong_tool_choice, missing_code_navigation, decomposition_failure, stale_context, multi_repo_scope_failure, local_remote_mismatch, verifier_mismatch, edit_verify_loop_failure, incomplete_solution, near_miss, minimal_progress, exception_crash, rate_limited_run, success_via_code_nav, success_via_semantic_search, success_via_local_exec, success_via_commit_context, success_via_decomposition, insufficient_provenance, task_ambiguity, over_exploration.

The spec listed 14 categories including premature_commit and clean_success which don't exist in v2. Will include all actual v2 categories plus premature_commit and clean_success as new categories.

## Test patterns

- Tests use pytest, fixtures from conftest.py
- `reset_taxonomy_cache` fixture resets module-level cache
- Tests import from `agent_diagnostics.llm_annotator` and `agent_diagnostics.taxonomy`
