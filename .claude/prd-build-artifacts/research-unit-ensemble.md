# Research: unit-ensemble

## CSB ensemble.py Interface

- `ensemble_annotate(signals, model, corpus_stats, classifier_threshold, classifier_min_f1)` -> list[dict]
- `ensemble_all(signals_list, model, ...)` -> annotation document dict
- `HEURISTIC_ONLY` set: exception_crash, rate_limited_run, near_miss, minimal_progress, edit_verify_loop_failure
- Uses `from observatory.annotator import annotate_trial, compute_corpus_stats`
- Uses `from observatory.classifier import load_model, predict_trial, signals_to_features`
- Accesses heuristic results as dicts: `c["name"]`, `c["confidence"]`, `c["evidence"]`

## Our annotator.py Interface

- `annotate_trial(signals, *, tool_registry=DEFAULT_REGISTRY)` -> list[CategoryAssignment]
- CategoryAssignment is frozen dataclass with `.name`, `.confidence`, `.evidence` attributes
- No `corpus_stats` parameter — thresholds are internal
- No `compute_corpus_stats` function exported

## Our classifier.py Interface

- `predict_trial(signals, model, threshold)` -> list[dict] with name/confidence/evidence keys
- `load_model(path)` -> dict
- `signals_to_features(signals)` -> list[float]

## Key Adaptations Needed

1. Import from agent_observatory, not observatory
2. Access CategoryAssignment via attributes (.name, .confidence, .evidence) not dict keys
3. Drop corpus_stats parameter from ensemble_annotate
4. Drop compute_corpus_stats from ensemble_all
5. Call annotate_trial(signals) without corpus_stats
6. HEURISTIC_ONLY should be frozenset (immutable)
