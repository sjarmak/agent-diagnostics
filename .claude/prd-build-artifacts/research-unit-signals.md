# Research: unit-signals

## Existing Modules

### types.py

- `TrialSignals` TypedDict with `total=False` and 26 keys
- Keys: task_id, model, config_name, benchmark, reward, passed, total_turns, tool_calls_total, search_tool_calls, edit_tool_calls, code_nav_tool_calls, semantic_search_tool_calls, unique_files_read, unique_files_edited, files_read_list, files_edited_list, error_count, retry_count, trajectory_length, has_result_json, has_trajectory, duration_seconds, rate_limited, exception_crashed, patch_size_lines, tool_call_sequence

### tool_registry.py

- `ToolRegistry` frozen dataclass with search_tools, edit_tools, code_nav_tools, semantic_search_tools (all frozenset[str])
- `DEFAULT_REGISTRY` instance with Claude Code + Sourcegraph MCP tools
- Has `.all_tools` property returning union

### CSB signals.py (reference)

- `_load_json()` — safe JSON loader returning None on error
- `_extract_reward()` — reads verifier_result.rewards.reward or .score
- `_detect_trajectory_patterns()` — computes query churn, edit-verify cycles, repeated failures, code nav, semantic search, git tools
- `_benchmark_from_path()` — CSB-specific path parsing (NOT portable)
- `_model_from_path()` — CSB-specific model keyword matching (NOT portable)
- `_config_from_path()` — CSB-specific path parsing (NOT portable)
- `_load_suite_mapping()` — loads from CSB configs dir (NOT portable)
- `_MODEL_KEYWORDS` — hardcoded model map (NOT portable)
- `extract_signals()` — main function, returns flat dict
- `extract_all()` — walks directory tree, yields trial dirs with result.json

## Key Differences: CSB vs Observatory

1. CSB uses hardcoded tool sets; observatory uses injectable ToolRegistry
2. CSB path parsing (\_benchmark_from_path etc.) becomes injectable callables
3. CSB \_MODEL_KEYWORDS becomes injectable model_keywords dict
4. CSB suite_mapping loaded from file; observatory accepts as parameter
5. Observatory TrialSignals has different keys than CSB signal dict (e.g., total_turns vs trajectory_steps, unique_files_read vs raw counts)
6. Trajectory format: steps[].tool_calls[].function_name + .arguments, steps[].observation

## Trajectory Format

```json
{
  "steps": [
    {
      "tool_calls": [
        { "function_name": "Grep", "arguments": { "pattern": "foo" } }
      ],
      "observation": { "results": [{ "content": "..." }] }
    }
  ]
}
```

## File Path Extraction

- Read: args.file_path
- Edit: args.file_path
- Write: args.file_path
- mcp**sourcegraph**read_file: args.path or args.file_path
