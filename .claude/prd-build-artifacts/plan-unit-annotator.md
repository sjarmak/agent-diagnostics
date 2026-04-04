# Plan: unit-annotator

## File: src/agent_observatory/annotator.py

### Helper functions

1. `_get(signals, key, default)` — safe TypedDict access via `.get()`
2. `_count_tools_in_sequence(sequence, tool_set)` — count tool_call_sequence entries matching a frozenset
3. `_has_bash_test_pattern(sequence)` — detect Bash tool calls (proxy for local exec)
4. `_has_git_calls(sequence)` — detect git-related Bash calls (proxy for commit context)

### Heuristic functions (each returns Optional[CategoryAssignment])

#### Neutral (check first — these override or coexist)

1. `_check_rate_limited` — rate_limited=True -> rate_limited_run (confidence 0.9)
2. `_check_exception_crash` — exception_crashed=True -> exception_crash (confidence 0.9)
3. `_check_task_ambiguity` — reward=0, few tool calls (<5), short duration (<60s), no crash/rate_limit -> task_ambiguity (confidence 0.4)

#### Failure modes (reward < 1.0 and not passed)

4. `_check_retrieval_failure` — reward=0, search_tool_calls=0 or very low, unique_files_read=0 or very low
5. `_check_query_churn` — reward<1, search_tool_calls>5, unique_files_read low relative to searches
6. `_check_wrong_tool_choice` — reward<1, high search calls but zero code_nav calls, many files read manually
7. `_check_missing_code_navigation` — reward<1, code_nav_tool_calls=0, multiple files read
8. `_check_decomposition_failure` — reward<1, multiple files edited but reward=0, or edit before reads
9. `_check_edit_verify_loop_failure` — reward<1, retry_count>3 or error_count>3
10. `_check_stale_context` — reward<1, high trajectory_length, many tool calls (proxy)
11. `_check_multi_repo_scope_failure` — reward=0, unique_files_edited<=1, unique_files_read low
12. `_check_local_remote_mismatch` — reward=0, error_count high, few edits
13. `_check_verifier_mismatch` — reward=0, patch_size_lines>0, edit_tool_calls>0 (agent did work but got 0)
14. `_check_over_exploration` — reward=0, tool_calls_total>100, few edits
15. `_check_incomplete_solution` — 0 < reward < 1
16. `_check_near_miss` — reward >= 0.5 and < 1.0
17. `_check_minimal_progress` — 0 < reward < 0.5

#### Success modes (reward >= 1.0 or passed=True)

18. `_check_success_via_code_nav` — reward=1, code_nav_tool_calls > threshold
19. `_check_success_via_semantic_search` — reward=1, semantic_search_tool_calls > 0
20. `_check_success_via_local_exec` — reward=1, has Bash tool calls in sequence (proxy for test execution)
21. `_check_success_via_commit_context` — reward=1, git-related tool calls in sequence
22. `_check_success_via_decomposition` — reward=1, unique_files_edited >= 3
23. `_check_insufficient_provenance` — reward>0, very few tool calls, no clear search/nav pattern

### Main function

```python
def annotate_trial(
    signals: TrialSignals,
    *,
    tool_registry: ToolRegistry = DEFAULT_REGISTRY,
) -> list[CategoryAssignment]:
```

- Call all 23 heuristic functions
- Collect non-None results
- Validate all names against valid_category_names()
- Return list sorted by confidence descending

## File: tests/test_annotator.py

### Test cases

1. test_import — verify import works
2. test_retrieval_failure — reward=0, passed=False, search=0, files_read=0
3. test_success_via_code_nav — reward=1, passed=True, code_nav>5
4. test_rate_limited — rate_limited=True
5. test_exception_crash — exception_crashed=True
6. test_custom_tool_registry — custom registry with different tool names
7. test_all_names_valid — any result has name in valid_category_names()
8. test_multiple_categories — signals triggering multiple categories
9. test_incomplete_solution — 0 < reward < 1
10. test_near_miss — reward=0.8
11. test_minimal_progress — reward=0.2
12. test_over_exploration — many tool calls, reward=0
