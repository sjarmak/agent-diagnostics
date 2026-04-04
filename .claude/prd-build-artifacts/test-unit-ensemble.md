# Test Results: unit-ensemble

## Summary

- **16/16 tests passed** in tests/test_ensemble.py
- **198/198 tests passed** across full test suite (0 failures)
- Import `from agent_observatory.ensemble import ensemble_annotate, ensemble_all` succeeds

## Test Coverage

### TestEnsembleAnnotate (8 tests)

- Returns list of dicts with name/confidence/evidence/source keys
- HEURISTIC_ONLY categories sourced from heuristic tier
- exception_crash detected via heuristic
- rate_limited_run detected via heuristic
- Classifier categories filtered by min_f1 threshold
- Classifier skips HEURISTIC_ONLY categories even if predicted
- Classifier evidence includes eval_f1 annotation
- Empty result when no matches

### TestHeuristicOnly (2 tests)

- HEURISTIC_ONLY is frozenset
- Contains all 5 structural categories

### TestEnsembleAll (5 tests)

- Produces annotation document with correct schema
- Annotations have correct structure (task_id, trial_path, etc.)
- Source key stripped from output categories
- Empty signals list returns empty annotations
- Annotator identity includes tier info

### TestNoCsbImports (1 test)

- AST verification: no imports from observatory (CSB)
