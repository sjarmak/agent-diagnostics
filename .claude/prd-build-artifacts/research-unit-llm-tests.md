# Research: unit-llm-tests

## annotate_trial_claude_code (lines 333-395)

- Takes trial_dir, signals, model; returns list[dict]
- Calls `_find_claude_cli()` (uses `shutil.which("claude")`)
- Reads instruction.txt and trajectory.json from trial_dir/agent/
- Builds prompt, runs `subprocess.run` with claude CLI args
- On success: parses JSON envelope via `_parse_claude_response`, then `validate_categories`
- Error paths: returncode!=0, TimeoutExpired, JSONDecodeError, generic Exception

## annotate_trial_api (lines 508-560)

- Takes trial_dir, signals, model; returns list[dict]
- Imports anthropic SDK (raises ImportError if missing)
- Creates Anthropic() client, calls messages.create
- Parses raw text response (strips code fences), json.loads, extracts categories
- Error paths: JSONDecodeError, generic Exception

## \_parse_claude_response (lines 280-314)

- Already has contract tests in TestParseClaudeResponse class
- Handles: is_error, structured_output, raw JSON fallback, code fences, non-list

## \_find_claude_cli (lines 322-330)

- Uses shutil.which("claude"), raises FileNotFoundError if None

## validate_categories (lines 250-272)

- Already has tests in TestValidateCategories class

## Existing test file (tests/test_llm_annotator.py)

- 540 lines, covers: imports, build_prompt, validate_categories, truncate_trajectory, summarise_step, dispatch, import error, taxonomy resolution, \_parse_claude_response, helpers
- Missing: actual backend tests for annotate_trial_claude_code and annotate_trial_api

## Fixtures available

- claude_envelope_structured_output.json, claude_envelope_raw_fallback.json
- claude_envelope_is_error.json, claude_envelope_empty_result.json
- claude_envelope_code_fences.json, claude_envelope_raw_list.json
