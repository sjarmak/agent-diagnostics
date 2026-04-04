# Plan: unit-ensemble

## Implementation Steps

1. Create `src/agent_observatory/ensemble.py`:
   - Import from agent_observatory.annotator (annotate_trial as heuristic_annotate)
   - Import from agent_observatory.classifier (load_model, predict_trial, signals_to_features)
   - Import from agent_observatory.taxonomy (load_taxonomy)
   - Define HEURISTIC_ONLY as frozenset
   - Port ensemble_annotate: adapt CategoryAssignment attribute access, drop corpus_stats
   - Port ensemble_all: remove compute_corpus_stats, call heuristic_annotate without corpus_stats

2. Create `tests/test_ensemble.py`:
   - Test ensemble_annotate returns list of dicts with name/confidence/evidence/source
   - Test HEURISTIC_ONLY categories come from heuristic tier
   - Test classifier categories filtered by min_f1
   - Test ensemble_all produces annotation document with correct structure
   - Verify no CSB imports via ast inspection

## Risks

- CategoryAssignment attribute vs dict access is the main adaptation
- Mocking annotate_trial must return CategoryAssignment objects, not dicts
