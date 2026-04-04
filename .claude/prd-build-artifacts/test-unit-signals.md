# Test Results: unit-signals

## Summary

- **45 tests** in `tests/test_signals.py`: ALL PASSED
- **132 tests** across full suite (`tests/`): ALL PASSED
- **0 regressions**

## Test Coverage by Scenario

| Test Class              | Count | Status |
| ----------------------- | ----- | ------ |
| TestImports             | 3     | PASS   |
| TestBasicExtraction     | 12    | PASS   |
| TestMissingTrajectory   | 3     | PASS   |
| TestExceptionCrash      | 2     | PASS   |
| TestRateLimit           | 2     | PASS   |
| TestSuiteMapping        | 3     | PASS   |
| TestCustomToolRegistry  | 1     | PASS   |
| TestWarningEmission     | 3     | PASS   |
| TestExtractAll          | 3     | PASS   |
| TestBenchmarkResolver   | 2     | PASS   |
| TestTaskIdNormalizer    | 2     | PASS   |
| TestToolCallSequence    | 1     | PASS   |
| TestFileListExtraction  | 3     | PASS   |
| TestRetryDetection      | 1     | PASS   |
| TestErrorCount          | 1     | PASS   |
| TestRewardScoreFallback | 1     | PASS   |
| TestNoCsbDependencies   | 1     | PASS   |

## Acceptance Criteria Verification

- [x] `from agent_observatory.signals import extract_signals` succeeds
- [x] `from agent_observatory.signals import extract_all` succeeds
- [x] extract_signals returns TrialSignals dict with all 26 keys
- [x] extract_signals accepts keyword args: tool_registry, suite_mapping, benchmark_resolver, task_id_normalizer, model_keywords
- [x] Correct reward, passed=True, tool counts, tool_call_sequence from synthetic data
- [x] UserWarning emitted when no suite_mapping or benchmark_resolver
- [x] benchmark populated correctly with suite_mapping
- [x] Custom ToolRegistry used for tool categorization
- [x] exception_crashed=True when exception_info present
- [x] rate_limited=True when exception_info contains rate_limit
- [x] extract_all walks directory tree and returns list of TrialSignals
- [x] No hardcoded CSB paths or imports
- [x] pytest tests/test_signals.py passes with 0 failures
- [x] pytest tests/ passes with 0 failures
