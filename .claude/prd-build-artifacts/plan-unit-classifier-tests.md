# Plan: unit-classifier-tests

## Steps

1. Add `TestToFloat` class with edge cases: unconvertible string, TypeError objects
2. Add `TestTrainBinaryLR` class testing all-same labels and zero-variance features
3. Add `TestFormatEvalMarkdown` class testing output format, table structure, no_classifier handling
4. Add `TestPredictAll` class testing with a trained model and mock taxonomy
5. Add test for `_scale` with empty matrix
6. Add test for evaluate with category missing from model (no_classifier path)
7. Run coverage, iterate until >= 85%
