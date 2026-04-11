# Research: unit-new-categories

## Existing State

### taxonomy_v3.yaml

- Already contains `reward_hacking` (Integrity), `fabricated_success` (Integrity), `hallucinated_api` (Faithfulness), `verification_skip` (Metacognition)
- Missing: `tool_argument_error`, `premature_termination`, `verification_skipped`
- Categories follow pattern: name, dimension, description, polarity, severity, derived_from_signal, signal_dependencies, detection_hints

### annotator.py

- 23 checker functions following `_check_<name>(signals, registry) -> Optional[CategoryAssignment]` pattern
- All checkers registered in `_ALL_CHECKERS` tuple
- Uses `_get()` helper for safe signal access, `_assignment()` for creating CategoryAssignment
- `annotate_trial()` iterates all checkers, filters by valid taxonomy names, sorts by confidence

### test_annotator.py

- One class per category with test methods
- Uses `_names()` helper to extract category names from results
- Tests construct TrialSignals dicts and call `annotate_trial()`

### Fixtures (tests/fixtures/trials/)

- Each directory has `expected.json` (categories array) and `trajectory.json` (steps array)
- 14 existing fixture directories (one per existing category that has fixtures)

### types.py - TrialSignals keys available

- `error_count`, `tool_calls_total`, `total_turns`, `trajectory_length`, `reward`, `edit_tool_calls`, `tool_call_sequence`

## Key Findings

- 3 of 6 categories already exist in taxonomy (reward_hacking, fabricated_success, hallucinated_api)
- `verification_skip` exists but `verification_skipped` is a new distinct name
- Need to add 3 new categories to taxonomy: tool_argument_error, premature_termination, verification_skipped
- Need 3 new heuristic checkers (tool_argument_error, premature_termination, verification_skipped)
- Need 12 new fixture directories (2 per all 6 categories)
- Need test classes for all 6 categories
