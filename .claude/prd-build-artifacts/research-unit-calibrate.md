# Research: unit-calibrate

## Source: /home/ds/CodeScaleBench/observatory/calibrate.py

### Module overview

Agreement and calibration analysis for annotation comparison. Two modes:

1. Heuristic-vs-LLM: per-category TP/FP/FN/precision/recall/F1
2. Cross-model: Cohen's kappa inter-rater reliability

### Public API

- `compare_annotations(heuristic_file, llm_file)` -> dict with shared_trials, categories, macro_avg
- `format_markdown(summary)` -> markdown string
- `cohen_kappa(a_labels, b_labels)` -> float
- `compare_cross_model(model_a_file, model_b_file)` -> dict with shared_trials, categories, uncalibrated_categories, macro_kappa
- `format_cross_model_markdown(summary, model_a_name, model_b_name)` -> markdown string
- `UNCALIBRATED_THRESHOLD = 0.4` constant

### Private helpers

- `_load_annotations(path)` -> dict[str, set[str]] - loads JSON annotation files

### Dependencies

- json, pathlib.Path, typing (stdlib only)
- NO CSB imports found

### Port assessment

Direct copy — no changes needed except verifying no `from observatory.` imports exist (confirmed: none present).
