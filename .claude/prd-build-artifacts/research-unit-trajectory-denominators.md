# Research: unit-trajectory-denominators

## Checker Classification

### Reward-Only (no trajectory needed)

- `_check_rate_limited` — only `rate_limited` flag
- `_check_exception_crash` — only `exception_crashed` flag
- `_check_incomplete_solution` — only `reward`
- `_check_near_miss` — only `reward`
- `_check_minimal_progress` — only `reward`

### Trajectory-Dependent (all others — 27 checkers)

All remaining checkers use at least one of: tool_calls_total, edit_tool_calls, search_tool_calls, code_nav_tool_calls, semantic_search_tool_calls, unique_files_read, unique_files_edited, files_edited_list, error_count, retry_count, trajectory_length, tool_call_sequence, duration_seconds, patch_size_lines, total_turns.

Categories: task_ambiguity, retrieval_failure, query_churn, wrong_tool_choice, missing_code_navigation, decomposition_failure, edit_verify_loop_failure, stale_context, multi_repo_scope_failure, local_remote_mismatch, verifier_mismatch, over_exploration, tool_argument_error, premature_termination, verification_skipped, verification_skip, premature_commit, planning_absence, tool_underutilization, reward_hacking, clean_success, success_via_code_nav, success_via_semantic_search, success_via_local_exec, success_via_commit_context, success_via_decomposition, insufficient_provenance.

## Report Structure (current)

- `_category_counts()` returns flat dict {name: count}
- `_render_markdown()` shows "Category Frequency" table with count only
- JSON has `category_counts` as flat dict
- No denominator info anywhere

## Key Fields

- `has_trajectory` already exists in TrialSignals
- `reward: float | None` — None means no verifier result
- Annotations passed to report.py are dicts with `categories`, `passed`, `reward` etc.
