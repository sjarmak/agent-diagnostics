# Research: unit-classifier

## CSB Reference Analysis

The CSB `observatory/classifier.py` implements:

- 21 FEATURE_NAMES for numeric signal extraction
- Pure-Python logistic regression (no numpy/sklearn)
- Functions: `_sigmoid`, `_dot`, `_scale`, `_standardize`, `_train_binary_lr`, `_predict_proba`
- Public API: `train()`, `predict_trial()`, `predict_all()`, `evaluate()`, `save_model()`, `load_model()`, `signals_to_features()`, `format_eval_markdown()`
- `_align_labels()` helper for matching LLM annotations with signal features

## Key Mappings (CSB → TrialSignals)

| CSB FEATURE_NAME   | TrialSignals key  | Notes   |
| ------------------ | ----------------- | ------- |
| has_exception      | exception_crashed | boolean |
| wall_clock_seconds | duration_seconds  | float   |
| trajectory_steps   | trajectory_length | int     |
| reward             | reward            | same    |
| passed             | passed            | same    |
| tool_calls_total   | tool_calls_total  | same    |
| has_trajectory     | has_trajectory    | same    |

CSB features NOT in TrialSignals (will keep as-is, extracted from signal dict):

- input_tokens, output_tokens, cost_usd
- search_calls_keyword, search_calls_nls, search_calls_deepsearch
- mcp_ratio, ttfr, query_churn_count, edit_verify_cycles
- repeated_tool_failures, has_code_nav_tools, has_semantic_search, has_git_tools

## Package Style

- Uses `from __future__ import annotations`
- Type annotations on all function signatures
- Frozen dataclasses for immutable types
- `from agent_observatory.taxonomy import load_taxonomy` is the import pattern
