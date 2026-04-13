# Research: unit-nullable-reward

## Reward usage sites

### types.py

- Line 24: `reward: float` in TrialSignals TypedDict -- CHANGE to `reward: float | None`
- Line 68: `reward(self) -> float` in TrialInput Protocol -- CHANGE to `float | None`
- Line 102: `reward: float` in Annotation dataclass -- CHANGE to `Optional[float]`

### signals.py

- Line 367: `reward: float | None = None` -- already nullable internally
- Line 370: `passed = reward is not None and reward > 0` -- already handles None
- Line 420: `"reward": reward if reward is not None else 0.0` -- KEY CHANGE: keep None
- Need to add `has_verifier_result` field to output

### annotator.py - reward comparisons in checkers:

- \_check_task_ambiguity (L75): `reward = _get(signals, "reward", None)` + `reward == 0.0` -- already guards None
- \_check_retrieval_failure (L109): `reward = _get(signals, "reward", None)` + `reward > 0.0` -- already guards None
- \_check_query_churn (L133): same pattern -- already guards None
- \_check_wrong_tool_choice (L153): same -- already guards None
- \_check_missing_code_navigation (L173): same -- already guards None
- \_check_decomposition_failure (L195): same + `reward == 0.0` -- already guards None
- \_check_edit_verify_loop_failure (L215): same -- already guards None
- \_check_stale_context (L241): same -- already guards None
- \_check_multi_repo_scope_failure (L265): same -- already guards None
- \_check_local_remote_mismatch (L285): same -- already guards None
- \_check_verifier_mismatch (L306): same -- already guards None
- \_check_over_exploration (L326): same -- already guards None
- \_check_tool_argument_error (L347): `reward is not None and reward >= 1.0` -- already guards None
- \_check_premature_termination (L369): same pattern -- already guards None
- \_check_verification_skipped (L391): same -- already guards None
- \_check_premature_commit (L417): same -- already guards None
- \_check_planning_absence (L457): same -- already guards None
- \_check_verification_skip (L490): same -- already guards None
- \_check_tool_underutilization (L520): same -- already guards None
- \_check_reward_hacking (L545): `reward is None or reward <= 0.0` -- already guards None
- \_check_clean_success (L582): `reward is not None and reward >= 1.0` -- already guards None
- \_check_incomplete_solution (L605): `reward is None` -> return None -- already guards
- \_check_near_miss (L621): same -- already guards
- \_check_minimal_progress (L637): same -- already guards
- _check_success_via_\* functions: all use `reward is not None and reward >= 1.0` -- already guard
- \_check_insufficient_provenance (L773): `reward is None or reward <= 0.0` -- already guards

KEY FINDING: All annotator checkers already handle None reward gracefully! They all use `_get(signals, "reward", None)` and check for None before comparisons.

### classifier.py

- Line 54-63: `_to_float()` already handles None -> 0.0
- Line 66-68: `signals_to_features()` uses `_to_float(signals.get(name))` -- already handles None
- Line 381: `reward = sig.get("reward")` then `float(reward) if reward is not None else 0.0` -- already handles None

### ensemble.py

- Line 100-101: `reward = sig.get("reward")` then `float(reward) if reward is not None else 0.0` -- already handles None

### report.py

- Line 32: `rewards = [a["reward"] for a in annotations if a.get("reward") is not None]` -- already filters None!
- Line 213: `"reward": a.get("reward", 0)` in examples -- should handle None

## Summary

The codebase is already largely prepared for nullable reward. The main changes needed:

1. types.py: Add `has_verifier_result: bool`, change `reward: float` to `reward: float | None`
2. signals.py line 420: Stop coercing None to 0.0
3. signals.py: Add `has_verifier_result` to output dict
4. Tests: Update key count assertions (26 -> 27), add None-reward test paths
