# Plan: unit-llm-tests

## TestAnnotateTrialClaudeCode

1. Helper: create tmp trial dir with agent/instruction.txt and agent/trajectory.json
2. Mock shutil.which to return "/usr/bin/claude" (for \_find_claude_cli)
3. Mock subprocess.run for each scenario:
   - **success_structured_output**: returncode=0, stdout=structured_output fixture JSON
   - **success_raw_fallback**: returncode=0, stdout=raw_fallback fixture JSON
   - **subprocess_failure**: returncode=1, stderr="error msg"
   - **timeout**: subprocess.run raises subprocess.TimeoutExpired
   - **bad_json**: returncode=0, stdout="not json"
   - **is_error_response**: returncode=0, stdout=is_error fixture JSON

## TestAnnotateTrialApi

1. Same tmp trial dir setup
2. Mock anthropic.Anthropic to return mock client with mock messages.create
3. Scenarios:
   - **success**: messages.create returns mock message with valid JSON text
   - **api_error**: messages.create raises Exception
   - **non_list_response**: messages.create returns message with non-list JSON

## Implementation notes

- Use @pytest.fixture for trial dir setup
- Mock at `agent_diagnostics.llm_annotator.shutil.which` and `agent_diagnostics.llm_annotator.subprocess.run`
- For API tests, mock `anthropic` import within the function
