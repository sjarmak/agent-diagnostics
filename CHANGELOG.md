# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `observatory report` now uses `--output-dir` for the output directory,
  matching the naming convention already used by `observatory calibrate`
  (`--output-dir` for dir-writing commands; `--output` for file-writing
  commands like `annotate`, `ingest`, `extract`). `--output` continues to
  work as a deprecated alias on `report` — it emits a `DeprecationWarning`
  and is slated for removal in 1.0 (`agent-diagnostics-xw7`).

## [0.8.1] — 2026-04-19

Patch release fixing three UX issues surfaced by a clean-room smoke of 0.8.0
from PyPI.

### Fixed
- `observatory report --annotations` now accepts `.jsonl` input
  (one-record-per-line, as emitted by `observatory annotate --output X.jsonl`)
  in addition to the existing `.json` document and bare-list forms. Previously
  chaining `annotate` → `report` with a `.jsonl` intermediate crashed with
  `json.decoder.JSONDecodeError: Extra data`. Malformed JSONL lines now
  produce a clean `logger.error` + `sys.exit(1)` instead of a raw traceback.
- `observatory annotate` records now carry `annotation_result_status`
  (`"ok"` when categories fire, `"no_categories"` otherwise), matching the
  LLM-annotate v2 contract that 0.8.0's CHANGELOG claimed was already
  end-to-end.
- `observatory calibrate` logs a `WARNING` when `shared_trials=0` pointing
  at the likely trial_path join-key mismatch (filesystem path vs
  `trial_id_short` dir names). Previously the mismatch silently produced an
  empty report; still exits 0 since empty output can be a legitimate result.

### Internal
- `tests/test_taxonomy.py::test_version_string` now pins the semver shape
  instead of a literal `"0.7.0"`, which was stale after the 0.8.0 bump and
  tempted a no-op release-prep edit. Follow-up (`agent-diagnostics-9py`)
  replaces the hardcoded `__version__` with `importlib.metadata`.

## [0.8.0] — 2026-04-19

### Added
- Calibration observability: `compute_ece`, `compute_brier`, and
  `reliability_diagram` as pure-stdlib functions in `calibrate.py`.
  `compare_annotations` now emits per-category `ece`, `brier`, and
  `reliability_bins` alongside the legacy TP/FP/FN/precision/recall/f1 keys.
- New `observatory calibrate` CLI subcommand with `--predictor` and mutually
  exclusive `--reference`/`--golden-dir`; writes `calibration.md` and
  `calibration.json` under `--output-dir`.
- Golden regression corpus of 45 real benchmark trials under
  `tests/fixtures/golden_corpus/` for end-to-end calibration smoke tests.
- Content-hash extract cache: path-independent, auto-invalidating caching
  layer for `extract_signals` keyed by SHA-256 of trial contents.
- Pipeline DAG runner for composing ingest → annotate → report as a DAG
  with per-stage caching and resumability.
- Comparative analysis sections in the reliability report (cross-model and
  cross-taxonomy-version comparisons).
- `trajectory_length` and `total_turns` analytic fields propagated through
  `cmd_annotate` onto every annotation record.
- `annotation_result_status` field propagated end-to-end through the v2
  annotation schema.

### Changed
- `src/` refactored from `print()` to structured `logging`; CLI entrypoint
  is the only handler-installing site.
- `llm_annotator.py` split into an 8-module package
  (`src/agent_diagnostics/llm_annotator/`) for clearer separation between
  dispatch, prompt building, caching, and backend adapters.
- `report.py` split into a 7-module package (`src/agent_diagnostics/report/`);
  `report/orchestration.py` trimmed to 133 lines.
- `reliability_diagram` return type upgraded to a `TypedDict` for mypy
  coverage; `pytest.approx` usage unified across calibration tests.
- `llm_annotator.dispatch` uses lazy attribute lookup so backends are only
  imported on first use.

### Fixed
- `observatory calibrate` internal temp directory (`--golden-dir` path) is
  now pinned to mode `0o700` and the composed `reference.json` to `0o600`
  regardless of caller umask. Portability hedge against non-CPython
  interpreters where `mkdtemp` may not default to owner-only.
- `observatory calibrate --golden-dir` now validates the path exists before
  calling `_collect_golden_corpus`; previously an unhandled
  `FileNotFoundError` leaked to the CLI as a traceback.
- `cmd_annotate` carries analytic fields (trajectory_length, total_turns,
  annotation_result_status) into annotation records rather than dropping
  them.
- Pre-existing mypy errors in the `report` package resolved.

### Documentation
- README gained a "Calibration metrics" section explaining ECE, Brier,
  reliability diagrams, and how to interpret the direction arrow.
- `cmd_calibrate` docstring documents the permission contract:
  `--output-dir` respects caller umask (and may contain corpus-derived
  content on shared hosts); internal temp artefacts are always owner-only.
- MH-0 verification command updated for the evolved `build_prompt`
  signature.

## [0.7.0] and earlier

No changelog was maintained prior to 0.8.0. See `git log` and PyPI release
history for 0.5.0 → 0.7.0.
