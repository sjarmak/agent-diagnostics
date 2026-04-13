# Research: unit-manifest-integration

## Existing Code Analysis

### \_resolve_benchmark() in signals.py (lines 301-328)

- Takes: task_id, suite_mapping, benchmark_resolver, trial_dir
- Priority: benchmark_resolver first, then suite_mapping prefix match on task_id, then suite_mapping prefix match on dir name
- Returns str | None

### suite_mapping usage

- Dict[str, str] mapping prefix -> benchmark name
- Used in extract_signals() and extract_all()
- Passed through to \_resolve_benchmark()

### TrialSignals TypedDict in types.py

- 26 fields, total=False
- No benchmark_source field currently
- Need to add benchmark_source: str

### extract_signals() in signals.py (lines 336-468)

- Calls \_resolve_benchmark() at line 412
- Sets benchmark = result or ""
- No tracking of HOW benchmark was resolved

### Key insight

- \_resolve_benchmark returns the benchmark name but not the source
- Need to either modify it to return source info or determine source in extract_signals()
- Simplest: modify \_resolve_benchmark to return a tuple (benchmark, source) or add a parallel function

## Files to modify

1. types.py - add benchmark_source to TrialSignals
2. signals.py - add load_manifest(), directory resolver, update \_resolve_benchmark/extract_signals
3. test_signals.py - add tests for all new functionality
