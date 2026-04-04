# Test Results: unit-calibrate

## Summary

- **21/21 tests passed** in test_calibrate.py
- **182/182 tests passed** across full test suite
- All imports verified

## Test Coverage

- TestCompareAnnotations: 6 tests (perfect agreement, known TP/FP/FN, no shared trials, empty, document format, macro avg)
- TestCohenKappa: 6 tests (perfect agreement k=1, perfect disagreement k<0, random k~0, degenerate all-same, length mismatch, empty)
- TestCompareCrossModel: 4 tests (basic, no shared trials, uncalibrated detection, empty files)
- TestFormatMarkdown: 2 tests (produces markdown, empty summary)
- TestFormatCrossModelMarkdown: 3 tests (produces markdown, empty summary, uncalibrated listed)

## Import Verification

```python
from agent_observatory.calibrate import compare_annotations, cohen_kappa, compare_cross_model
# OK
```

## No CSB Imports

Confirmed: no `from observatory.` imports in calibrate.py.
