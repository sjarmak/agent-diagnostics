# Research: unit-blend-labels

## Source: /home/ds/CodeScaleBench/observatory/blend_labels.py

### Overview

`blend_labels.py` blends LLM and heuristic annotations into a unified training set. LLM labels are ground truth; heuristic labels supplement for trusted categories.

### Function: `blend()`

- **Params**: `heuristic_file`, `llm_file`, `calibration_file=None`, `heuristic_trust_threshold=0.7`, `max_heuristic_samples=2000`
- **Returns**: dict with `schema_version`, `taxonomy_version`, `generated_at`, `annotator`, `blend_metadata`, `annotations`

### Key Logic

1. Loads heuristic and LLM annotation JSON files
2. Determines trusted heuristic categories:
   - With calibration_file: categories with F1 >= threshold from `cal["categories"][cat]["f1"]`
   - Without calibration_file: hardcoded set of 5 defaults (`rate_limited_run`, `exception_crash`, `near_miss`, `over_exploration`, `edit_verify_loop_failure`)
3. Indexes both by `trial_path`
4. For each trial path (sorted):
   - If LLM annotation exists: uses all LLM categories, supplements with trusted heuristic categories not already present
   - If heuristic-only and under cap: includes only trusted categories (skips trial if none)
   - Otherwise: skips
5. Returns Observatory schema document with `blend_metadata` containing counts

### Imports

- `json`, `datetime`, `pathlib.Path`, `typing.Any` — no CSB imports

### Schema Compatibility

- Output matches `observatory-annotation-v1` schema (annotator type "blended" is in the enum)
- Categories include extra `source` field (not in schema's category_assignment — this is fine, `additionalProperties: false` in schema would reject it, but blend output is for training, not strict validation)

### Port Notes

- Direct copy — no `from observatory.` imports to replace
- Only need to change module to `from agent_observatory.blend_labels import blend`
