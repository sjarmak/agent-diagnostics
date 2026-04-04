# Test Results: unit-blend-labels

## Import Check

- `from agent_observatory.blend_labels import blend` — OK

## Test Results: tests/test_blend_labels.py

- 10/10 passed, 0 failures

| Test                                                                           | Status |
| ------------------------------------------------------------------------------ | ------ |
| TestBasicBlend::test_returns_required_keys                                     | PASS   |
| TestBasicBlend::test_basic_blend_with_data                                     | PASS   |
| TestLLMPriority::test_llm_categories_take_priority                             | PASS   |
| TestHeuristicOnlyTrusted::test_only_trusted_categories_included                | PASS   |
| TestHeuristicOnlyTrusted::test_heuristic_only_skipped_if_no_trusted            | PASS   |
| TestMaxHeuristicSamples::test_cap_limits_heuristic_only_count                  | PASS   |
| TestCalibrationFile::test_calibration_selects_trusted_categories               | PASS   |
| TestCalibrationFile::test_calibration_threshold_exact_boundary                 | PASS   |
| TestDefaultTrustedCategories::test_default_categories_used_without_calibration | PASS   |
| TestBlendMetadata::test_metadata_counts_correct                                | PASS   |

## Full Suite: tests/

- 163/163 passed, 0 failures
