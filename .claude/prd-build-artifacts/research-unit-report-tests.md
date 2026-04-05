# Research: unit-report-tests

## Functions to test

### \_core_task_name(trial_path: str) -> str

- Strips trailing `/`, takes last path segment
- Strips `__hash` suffix (rsplit on `__`)
- Strips prefixes: `baseline_`, `bl_`, `mcp_`, `sgonly_` (first match only)
- Returns lowercased result
- Patterns: bare task, task with hash, task with prefix, task with prefix+hash, path with dirs

### \_paired_comparison(annotations: list[dict]) -> list[dict]

- Groups annotations by (core_task, config) -> set of category names
- Finds config pairs sharing >= 20 tasks
- Computes per-category delta (A - B)
- Returns top 5 introduced_by_a (positive delta) and introduced_by_b (negative delta, shown as abs)
- Sorted by shared_tasks descending

### \_top_categories_with_examples(annotations, polarity_map, polarity, top_n=3, examples_per=2)

- Counts categories matching given polarity
- Returns top_n categories with examples_per example trials each
- Each example has task_id, config_name, reward, evidence

### \_category_by_suite(annotations: list[dict]) -> dict[str, dict[str, int]]

- Groups by benchmark suite -> category -> count
- Sorted by total annotation count desc, categories within sorted by count desc

### \_render_markdown — "no success annotations" branch

- Lines 345-357: if top_successes is empty list, renders "No success-mode annotations found in this corpus."

## Existing test coverage

- Tests cover generate_report end-to-end (file creation, sections, JSON keys, corpus stats, category counts, config breakdown, empty annotations)
- No direct unit tests for \_core_task_name, \_paired_comparison, \_top_categories_with_examples, \_category_by_suite, or \_render_markdown
