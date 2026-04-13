# Plan: unit-trial-filter

## Steps

1. Add `_EXCLUDED_DIR_PATTERNS` constant tuple in signals.py (internal helpers section)
2. Add `_is_valid_trial(data: dict) -> bool` function that:
   - Returns False if `agent_info` key is missing from data
   - Returns True otherwise
3. Add `_is_excluded_path(trial_dir: Path) -> bool` function that:
   - Checks if any part of the path matches excluded patterns
   - Returns True if any pattern matches
4. Wire into `extract_all()`: load result.json, call `_is_valid_trial()` and `_is_excluded_path()`, skip if invalid
5. Write tests covering all acceptance criteria
6. Run tests, fix failures
