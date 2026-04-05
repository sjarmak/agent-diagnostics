# Research: unit-classifier-tests

## Functions to Test

### `_to_float(val)`

- None -> 0.0, bool -> 1.0/0.0, numeric -> float, unconvertible -> 0.0
- Missing coverage: lines 62-63 (unconvertible string returning 0.0)

### `_scale(features_matrix)`

- Empty matrix returns default means/stds
- Missing coverage: lines 90-91

### `_train_binary_lr(X, y, means, stds, ...)`

- When all labels are same (pos==0 or neg==0), w_pos=w_neg=1.0
- Missing coverage: lines 135-136

### `predict_all(signals_list, model, threshold)`

- Calls predict_trial for each signal, builds annotation document
- Imports load_taxonomy internally
- Missing coverage: lines 349-376

### `format_eval_markdown(eval_results, model)`

- Formats eval results as markdown table
- Handles no_classifier status entries
- Missing coverage: lines 467-492

### `evaluate` additional paths

- no_classifier status when category not in model classifiers
- Missing coverage: lines 421-425

## Existing Tests

- 19 tests covering signals_to_features, FEATURE_NAMES, train, save/load, predict_trial, evaluate basics, no-forbidden-imports, imports
- Current coverage: 80%
