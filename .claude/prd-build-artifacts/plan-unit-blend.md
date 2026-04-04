# Plan: unit-blend-labels

## Files

1. `src/agent_observatory/blend_labels.py` — direct port from CSB
2. `tests/test_blend_labels.py` — pytest tests with tmp_path

## Implementation

- Copy CSB `blend_labels.py` verbatim (no CSB imports exist, no changes needed)
- Verify no `from observatory.` imports

## Tests (test_blend_labels.py)

1. `test_basic_blend` — LLM + heuristic files produce valid output with schema_version, annotations, blend_metadata
2. `test_llm_priority` — shared trial uses LLM categories, not heuristic
3. `test_heuristic_only_trusted` — heuristic-only trials include only trusted categories
4. `test_max_heuristic_samples_cap` — cap limits heuristic-only count
5. `test_with_calibration_file` — calibration F1 selects trusted categories
6. `test_without_calibration_default_categories` — default 5 categories used
7. `test_blend_metadata_counts` — verify llm_trials, heuristic_only_trials, total_blended counts

## Helpers

- `_make_annotation_file(tmp_path, name, annotations)` — writes JSON annotation doc to tmp_path
- `_make_calibration_file(tmp_path, categories)` — writes calibration JSON
