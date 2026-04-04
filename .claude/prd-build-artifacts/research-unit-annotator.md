# Research: unit-annotator

## Data Structures

### TrialSignals (TypedDict, total=False) — 26 keys

All optional due to `total=False`. Must use safe access pattern like `signals.get(key, default)`.

Key signals for heuristic rules:

- `reward: float` — 0.0=fail, 1.0=pass, partial in between
- `passed: bool` — binary pass/fail
- `rate_limited: bool` — whether run was rate limited
- `exception_crashed: bool` — whether agent crashed
- `search_tool_calls: int` — count of search tool usage
- `edit_tool_calls: int` — count of edit tool usage
- `code_nav_tool_calls: int` — count of code nav tool usage
- `semantic_search_tool_calls: int` — count of semantic search usage
- `unique_files_read: int` — distinct files read
- `unique_files_edited: int` — distinct files edited
- `tool_calls_total: int` — total tool calls
- `total_turns: int` — conversation turns
- `error_count: int` — errors encountered
- `retry_count: int` — retries
- `trajectory_length: int` — trajectory steps
- `duration_seconds: float` — wall clock time
- `patch_size_lines: int` — lines changed
- `tool_call_sequence: list[str]` — ordered tool names

### CategoryAssignment (frozen dataclass)

- `name: str` — must be in valid_category_names()
- `confidence: float` — 0.0 to 1.0
- `evidence: Optional[str]` — brief explanation

### ToolRegistry (frozen dataclass)

- `search_tools: frozenset[str]`
- `edit_tools: frozenset[str]`
- `code_nav_tools: frozenset[str]`
- `semantic_search_tools: frozenset[str]`

### valid_category_names() -> Set[str]

Returns 23 category names from taxonomy_v1.yaml.

## 23 Categories and Detection Hints Summary

### Failure (15)

1. retrieval_failure — reward=0, search calls low/zero, key files never opened
2. query_churn — many search queries (>5) with low overlap, few file reads
3. wrong_tool_choice — task needs xref but no xref tools used; excessive grep
4. missing_code_navigation — no go_to_def/find_refs calls, many manual file reads
5. decomposition_failure — multi-file task, edits before understanding scope, partial solution
6. edit_verify_loop_failure — multiple edit-test cycles (>3) with failures
7. stale_context — file read early, edited later without re-read
8. multi_repo_scope_failure — task spans directories, agent only explored one
9. local_remote_mismatch — environment errors, commands not available
10. verifier_mismatch — reward=0 but solution looks correct (hard to detect heuristically)
11. over_exploration — tool_calls_total far above mean, reward=0, many reads few edits
12. incomplete_solution — 0 < reward < 1
13. near_miss — reward >= 0.5 and < 1.0
14. minimal_progress — 0 < reward < 0.5
15. exception_crash — exception_crashed=True

### Success (5)

16. success_via_code_nav — reward=1, code_nav_tool_calls > 0
17. success_via_semantic_search — reward=1, semantic_search_tool_calls > 0
18. success_via_local_exec — reward=1, edit-test loop present
19. success_via_commit_context — reward=1, git history tools used
20. success_via_decomposition — reward=1, multiple files edited in sequence

### Neutral (3)

21. insufficient_provenance — reward > 0, few tool calls, unclear reasoning path
22. rate_limited_run — rate_limited=True
23. task_ambiguity — hard to detect heuristically (low tool calls, reward=0, short duration)
