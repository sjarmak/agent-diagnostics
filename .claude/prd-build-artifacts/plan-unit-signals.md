# Plan: unit-signals

## Steps

1. Create `src/agent_observatory/signals.py` with:
   - `_load_json(path)` — safe JSON loader
   - `_extract_reward(data)` — from verifier_result.rewards
   - `_parse_trajectory(traj, registry)` — extract tool counts, file lists, errors, retries, sequence, patch size
   - `_resolve_model(result, model_keywords)` — model from result.json or keyword matching
   - `_resolve_benchmark(task_id, suite_mapping, benchmark_resolver, trial_dir)` — benchmark from suite_mapping prefix or callable
   - `extract_signals(trial_dir, *, tool_registry, suite_mapping, benchmark_resolver, task_id_normalizer, model_keywords)` — main function returning TrialSignals
   - `extract_all(root_dir, **kwargs)` — walk tree, collect TrialSignals list

2. Update `src/agent_observatory/__init__.py` to export extract_signals and extract_all

3. Create `tests/test_signals.py` with fixtures and test scenarios per acceptance criteria

4. Run tests, fix failures

5. Commit
