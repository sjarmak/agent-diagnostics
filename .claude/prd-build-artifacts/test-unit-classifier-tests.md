# Test Results: unit-classifier-tests

## Coverage

- **97%** coverage on `agent_diagnostics.classifier` (target: >= 85%)
- 36 tests collected, 36 passed

## Missing Lines

- 264-265: skipped categories branch in train (minor path)
- 360: predict_all skip when no categories predicted
- 411, 435, 437: minor evaluate paths

## Tests Added

- `TestToFloat`: 6 tests for `_to_float` edge cases (unconvertible string, object, None, bools, numeric string)
- `TestScale`: 1 test for empty matrix defaults
- `TestTrainBinaryLR`: 3 tests for degenerate inputs (all-positive, all-negative, zero-variance)
- `TestPredictAll`: 2 tests for annotation document structure and field validation
- `TestFormatEvalMarkdown`: 3 tests for markdown output, no_classifier rows, skipped categories
- `TestEvaluateNoClassifier`: 1 test for no_classifier status path
- `TestTrainValidation`: 1 test for non-list annotations error
