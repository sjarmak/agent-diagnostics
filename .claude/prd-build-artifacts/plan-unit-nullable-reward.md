# Plan: unit-nullable-reward

## Step 1: types.py

- Add `has_verifier_result: bool` to TrialSignals TypedDict
- Change `reward: float` to `reward: float | None` (using Optional for compatibility)
- Change Annotation.reward from `float` to `Optional[float]`
- Change TrialInput.reward property return type to `float | None`

## Step 2: signals.py

- Line 420: Change `"reward": reward if reward is not None else 0.0` to `"reward": reward`
- Add `"has_verifier_result": reward is not None` to the signals dict (between `has_trajectory` and `duration_seconds`)
- The `_extract_reward` function already returns `float | None` -- no change needed

## Step 3: annotator.py

- All checkers already use `_get(signals, "reward", None)` and guard against None -- NO CHANGES NEEDED

## Step 4: classifier.py

- `_to_float()` already handles None -> 0.0 -- NO CHANGES NEEDED
- `signals_to_features()` already handles None via `_to_float` -- NO CHANGES NEEDED
- `predict_all()` line 381 already handles None reward -- NO CHANGES NEEDED

## Step 5: ensemble.py

- Line 110 already handles None reward -- NO CHANGES NEEDED

## Step 6: report.py

- Line 32 already filters None rewards -- NO CHANGES NEEDED
- Line 213: `"reward": a.get("reward", 0)` -- minor, but functionally fine since it's display

## Step 7: test_types.py

- Update key count from 26 to 27
- Add `has_verifier_result` to expected_keys set
- Update Annotation test to use Optional[float] reward

## Step 8: test_signals.py

- Update expected key count from 26 to 27 (in class name and assertion)
- Add `has_verifier_result` to expected_keys set
- Add test for None reward when no verifier_result
- Add test for has_verifier_result=True/False
- Update reward=0.0 assertions for no-verifier trials to reward=None

## Step 9: test_annotator.py

- Add test class for None reward path (no crash, returns appropriate results)

## Step 10: test_classifier.py

- Add test for None reward in signals_to_features

## Step 11: test_ensemble.py

- Add test for None reward handling in ensemble_all output

## Step 12: test_report.py

- Add test for None reward in corpus_stats (avg_reward calculation)
