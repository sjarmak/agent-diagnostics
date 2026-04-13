# PRD Build Log: Agent Trace Dataset Pipeline

## 2026-04-12T00:00:00Z — Decomposition complete — 8 units across 3 layers

| Layer | Units                                                                                       |
| ----- | ------------------------------------------------------------------------------------------- |
| 0     | unit-trial-filter, unit-nullable-reward, unit-openhands-registry, unit-manifest-integration |
| 1     | unit-trajectory-denominators, unit-jsonl-format                                             |
| 2     | unit-ingest-command, unit-cooccurrence-dimensions                                           |

Baseline: 651 tests.

## Layer 0 landed: unit-trial-filter (0fda72e), unit-nullable-reward (7904811), unit-openhands-registry (841e2b1), unit-manifest-integration (ebec0ed)

- \_is_valid_trial() predicate, \_is_excluded_path() for 5 dir patterns, wired into extract_all()
- reward: float|None, has_verifier_result: bool, None-safe heuristic checkers
- OPENHANDS_REGISTRY, get_registry_for_agent(), auto-detect in extract_signals()
- load_manifest(), directory-name benchmark fallback, benchmark_source provenance
- 716 tests passing

## Layer 1 landed: unit-trajectory-denominators (51eea82), unit-jsonl-format (945ba7e)

- CHECKER_REQUIRES_TRAJECTORY dict, split report sections with correct denominators
- write_jsonl/load_signals helpers, .meta.json sidecar, CLI format detection
- 763 tests passing
