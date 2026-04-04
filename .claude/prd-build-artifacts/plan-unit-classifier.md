# Plan: unit-classifier

## Implementation

1. Create `src/agent_observatory/classifier.py` ported from CSB with:
   - Updated import: `from agent_observatory.taxonomy import load_taxonomy`
   - Renamed FEATURE_NAMES: has_exception→exception_crashed, wall_clock_seconds→duration_seconds, trajectory_steps→trajectory_length
   - All pure-Python math functions preserved
   - All public API functions preserved

2. Create `tests/test_classifier.py` with:
   - `test_signals_to_features` — TrialSignals dict conversion
   - `test_train_save_load_roundtrip` — train with synthetic data, save/load
   - `test_predict_trial_structure` — verify output dict keys
   - `test_evaluate_returns_metrics` — per-category precision/recall/F1
   - `test_no_numpy_sklearn_imports` — grep source for forbidden imports
   - `test_feature_names_list` — verify FEATURE_NAMES is a list of strings

## Files

- `src/agent_observatory/classifier.py` (new)
- `tests/test_classifier.py` (new)
