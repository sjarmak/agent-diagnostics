# Plan: writer-refactor

## Goal

Refactor four annotation writers to also write through AnnotationStore.upsert_annotations() in narrow-tall schema, in addition to their existing legacy JSON output.

## Design Decisions

1. **Additive change**: Each writer command gets a new `--annotations-out` arg (default: `data/annotations.jsonl`). When provided, it writes narrow-tall rows via AnnotationStore. The existing `--output` behavior is UNCHANGED for backwards compatibility.

2. **Conversion function**: A shared helper `_annotations_to_narrow_rows()` converts the nested annotation format (trial -> categories[]) to narrow-tall rows suitable for AnnotationStore.

3. **Annotator identities**:
   - cmd_annotate: annotator_type='heuristic', annotator_identity='heuristic:rule-engine'
   - cmd_llm_annotate: annotator_type='llm', annotator_identity resolved from model arg via \_API_MODEL_MAP -> resolve_identity (e.g., "haiku" -> "claude-haiku-4-5-20251001" -> "llm:haiku-4")
   - cmd_predict: annotator_type='classifier', annotator_identity='classifier:trained-model'
   - cmd_ensemble: annotator_type='ensemble', annotator_identity='ensemble:heuristic+classifier'

4. **trial_id**: Each signal dict already has `trial_id` from extract_signals. For signals that don't have it, compute via compute_trial_id().

5. **ensemble_all**: Add optional `annotations_out` parameter. When provided, writes via AnnotationStore. Return value unchanged for backwards compat.

## Files Modified

- `src/agent_diagnostics/cli.py` - Add --annotations-out arg to annotate, llm-annotate, predict, ensemble. Add \_annotations_to_narrow_rows() helper. Add AnnotationStore writes.
- `src/agent_diagnostics/ensemble.py` - Add optional annotations_out param to ensemble_all.
- `tests/test_writer_refactor.py` - New integration test file.

## Implementation Steps

1. Add `_annotations_to_narrow_rows()` helper in cli.py
2. Modify cmd_annotate to add --annotations-out and AnnotationStore write
3. Modify cmd_llm_annotate similarly
4. Modify cmd_predict similarly
5. Modify cmd_ensemble similarly
6. Modify ensemble_all to accept annotations_out param
7. Add --annotations-out arg to parser for all 4 commands
8. Write integration tests

## Backwards Compatibility

- Existing --output flag unchanged
- ensemble_all return value unchanged
- All existing tests must pass without modification
- ensemble_all still exported in **init**.py
