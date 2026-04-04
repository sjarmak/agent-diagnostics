# Plan: unit-report

## Implementation

1. Copy CSB `report.py` to `src/agent_observatory/report.py`
2. Change single import: `from observatory.taxonomy` → `from agent_observatory.taxonomy`
3. No other changes needed — module is already generic

## Tests (tests/test_report.py)

1. `test_generate_report_creates_files` — verify .md and .json files exist
2. `test_md_contains_required_sections` — check all 5 required sections present
3. `test_json_has_required_keys` — corpus_stats, category_counts, category_by_config
4. `test_corpus_stats_correct` — verify total, passed, failed, pass_rate calculations
5. `test_category_counts_correct` — verify category counting
6. `test_config_breakdown_groups` — verify grouping by config_name
7. `test_empty_annotations` — empty list produces valid minimal report

## Fixtures

- Synthetic annotation document with 3-4 annotations, mix of passed/failed, multiple configs and categories
