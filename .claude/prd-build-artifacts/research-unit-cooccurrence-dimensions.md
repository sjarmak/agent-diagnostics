# Research: unit-cooccurrence-dimensions

## Key findings

- `report.py` has `generate_report(annotations, output_dir)` that produces `reliability_report.md` and `.json`
- `_render_markdown()` accepts stats, cat_counts, etc. and returns markdown string
- `generate_report()` computes stats, calls `_render_markdown()`, writes both files
- JSON companion is a dict with keys: generated_at, corpus_stats, category_counts, category_by_config, category_by_suite, paired_comparisons
- Taxonomy loaded via `from agent_diagnostics.taxonomy import load_taxonomy` returns dict with `dimensions` list, each has `name` and `categories` list
- 11 dimensions in taxonomy_v3: Retrieval, ToolUse, Reasoning, Execution, Environment, Faithfulness, Metacognition, Integrity, Safety, Strategy, Observability
- Each annotation has: task_id, trial_path, reward, passed, categories (list of {name, confidence, evidence})
- Existing test file uses `_make_annotations()` helper and pytest fixtures

## Integration points

- New functions: `co_occurrence_matrix()`, `dimension_aggregation()`
- `_render_markdown()` needs new `dimension_summary` parameter for Dimension Summary section
- `generate_report()` needs to call both new functions and pass results to renderer and JSON
