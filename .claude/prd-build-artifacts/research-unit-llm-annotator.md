# Research: unit-llm-annotator

## Source Analysis

- CSB source: `/home/ds/CodeScaleBench/observatory/llm_annotator.py` (832 lines)
- Two backends: claude-code (subprocess) and Anthropic API
- Judge scoring section (lines 653-832): `_build_judge_input`, `_extract_code_changes`, `judge_trial` — CSB-specific, must be REMOVED
- Private helpers to promote: `_build_prompt`, `_validate_categories`, `_truncate_trajectory`, `_summarise_step`
- Uses `Path(__file__).parent` for taxonomy — must switch to `_package_data_path`
- `annotate_trial_api` guards anthropic import with `logger.error + return []` — must change to `raise ImportError`

## Target Package

- Package: `agent_observatory` at `/home/ds/agent-observatory/src/agent_observatory/`
- `taxonomy.py` exports `load_taxonomy`, `_package_data_path`
- Optional dep: `anthropic>=0.30` under `[llm]` extra
- Python >=3.10, hatchling build

## Key Changes

1. Remove: `_build_judge_input`, `_extract_code_changes`, `judge_trial`
2. Remove: all `sys.path` manipulation, `csb_metrics` imports
3. Change import: `from agent_observatory.taxonomy import load_taxonomy`
4. Change `_taxonomy_yaml()` to use `_package_data_path`
5. Promote 4 private functions to public (drop leading underscore)
6. Change anthropic ImportError handling to raise
7. Keep all other functions as-is
