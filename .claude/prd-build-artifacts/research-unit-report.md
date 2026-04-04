# Research: unit-report

## Source Analysis

CSB module: `/home/ds/CodeScaleBench/observatory/report.py`

### Public API

- `generate_report(annotations: dict, output_dir: Path) -> tuple[Path, Path]`
  - Takes annotation document dict and output directory
  - Returns (md_path, json_path)

### Internal Functions

- `_load_taxonomy_polarity()` — maps category name to polarity (failure/success/neutral)
- `_corpus_stats(annotations)` — total, passed, failed, pass_rate, avg_reward, configs, benchmarks
- `_category_counts(annotations)` — Counter of category names across all annotations
- `_category_by_config(annotations)` — nested dict config -> category -> count
- `_category_by_suite(annotations)` — nested dict benchmark -> category -> count
- `_core_task_name(trial_path)` — extracts canonical task name for pairing
- `_paired_comparison(annotations)` — cross-config failure delta analysis
- `_top_categories_with_examples(annotations, polarity_map, polarity)` — top N categories with examples
- `_render_markdown(...)` — renders full MD report

### Required Change

- `_load_taxonomy_polarity`: replace `from observatory.taxonomy import load_taxonomy` with `from agent_observatory.taxonomy import load_taxonomy`

### Dependencies

- `agent_observatory.taxonomy.load_taxonomy` — already available in the package
- Standard library: json, collections, datetime, pathlib, typing

### Markdown Sections

1. Corpus Statistics
2. Category Frequency
3. Category Breakdown by Config (aka "Config Breakdown")
4. Configuration Comparison (Paired) — conditional
5. Category Breakdown by Benchmark Suite
6. Top Failure Categories
7. Success Mode Summary

### JSON Keys

- generated_at, corpus_stats, category_counts, category_by_config, category_by_suite, paired_comparisons
