# PRD: Extract Observatory into Standalone `agent-observatory` PyPI Package

## Problem Statement

CodeScaleBench's `observatory/` subpackage — the Agent Reliability Observatory — is a
general-purpose framework for analyzing why coding agents succeed or fail. It provides a
23-category behavioral taxonomy, JSON annotation schema, signal extraction, heuristic/LLM/
ensemble annotators, a pure-Python classifier, and CLI tooling. Despite being general-purpose
in design, it is currently coupled to CSB through hardcoded paths, `sys.path` manipulation,
and private API imports — making it unusable by other benchmarks or agent frameworks.

An audit of the coupling surface reveals the problem is concentrated, not pervasive: only 2
of 10+ observatory modules (`signals.py`, `llm_annotator.py`) have CSB-specific code. The
remaining modules (taxonomy, annotator, classifier, ensemble, report, calibrate, blend_labels,
cli) are already fully generic. This makes extraction feasible with targeted refactoring.

## Goals & Non-Goals

### Goals

- Publish `agent-observatory` on PyPI as a standalone package that any agent benchmark can use
- Define a formal trial data contract (`TrialSignals` TypedDict) that decouples observatory from CSB's directory layout
- Provide extension points (callable injection) for benchmark-specific behavior (suite detection, task ID normalization, judge scoring)
- Bundle taxonomy definitions, annotation schema, and exemplars as package data
- Maintain backwards compatibility for CSB — no regression in `csb diagnose` or `csb report` functionality
- Preserve git history for extracted files

### Non-Goals

- Building a setuptools entry-point plugin registry (overkill for current consumer count)
- Extracting CSB's scoring infrastructure (`csb_metrics.judge`) into observatory
- Supporting non-Python consumers (no REST API, no language bindings)
- Guaranteeing taxonomy stability — taxonomy versions independently from package version
- Publishing `annotations/` directory data (CSB-specific run artifacts)

## Requirements

### Must-Have

- Requirement: Promote private functions in `llm_annotator.py` to public API
  - Acceptance: `diagnose.py` imports only public (non-underscore-prefixed) symbols from `agent_observatory`. Running `grep -r "from observatory\." lib/ scripts/ | grep "\._"` returns zero matches after migration.

- Requirement: Move judge scoring (`_build_judge_input`, `judge_trial`) out of observatory into CSB adapter code
  - Acceptance: `observatory/llm_annotator.py` contains zero imports of `csb_metrics`. `grep -r "csb_metrics" observatory/` returns zero results.

- Requirement: Make suite_mapping a parameter, not a hardcoded path
  - Acceptance: `extract_signals()` accepts an optional `suite_mapping: dict[str, str] | None` parameter. When `None`, `benchmark` field in signals is `None` with a `warnings.warn()` logged. No `Path(__file__).parent.parent` references remain in `signals.py`.

- Requirement: Define `TrialSignals` TypedDict as the formal contract between signal extraction and downstream pipeline
  - Acceptance: `agent_observatory.types.TrialSignals` is importable. All 26 signal keys are typed. `annotator.py`, `classifier.py`, `ensemble.py` type-hint their inputs as `TrialSignals`.

- Requirement: Bundle taxonomy YAMLs, annotation schema JSON, and exemplars as package data
  - Acceptance: `pip install agent-observatory && python -c "from agent_observatory.taxonomy import load_taxonomy; t = load_taxonomy(); assert len(t['categories']) >= 23"` succeeds in a clean virtualenv.

- Requirement: CSB continues to work identically after migration
  - Acceptance: All existing tests in `tests/test_taxonomy_v1_spec.py`, `tests/test_taxonomy_v2_calibration.py`, `tests/test_diagnose.py`, and `tests/test_report_browser.py` pass. `csb diagnose` and `csb report` produce identical output on the same input.

### Should-Have

- Requirement: Accept callable injection for benchmark-specific operations
  - Acceptance: `extract_signals()` accepts `benchmark_resolver: Callable[[Path], str | None] = None` and `task_id_normalizer: Callable[[str], str] = None`. CSB adapter passes its implementations. Default behavior (no resolver) works without error.

- Requirement: Emit warnings when running without benchmark-specific context
  - Acceptance: Running observatory standalone on a trial directory without suite_mapping logs a `UserWarning` about reduced annotation quality (specifically: `missing_code_navigation` heuristic disabled).

- Requirement: Phase 1 internal refactor ships as a commit to CSB before any extraction
  - Acceptance: A commit exists where observatory/ still lives in CSB but all private-to-public promotions and parameter injection are complete. All tests pass at this commit.

- Requirement: Exclude trained model files (`model.json`, `model_60.json`) from package data — load from user-specified paths
  - Acceptance: `classifier.load_model()` accepts a `model_path: Path` parameter. Package installs without model files. `pip show agent-observatory` shows package size < 500KB (taxonomy + schema + exemplars only).

### Nice-to-Have

- Requirement: uv workspace configuration for monorepo-style development
  - Acceptance: `uv sync` from CSB root installs observatory as an editable dependency. `uv run pytest tests/` passes.

- Requirement: `_MODEL_KEYWORDS` dict (Anthropic model name mapping) is pluggable
  - Acceptance: `extract_signals()` accepts `model_keywords: dict[str, str] | None` parameter for non-Claude agent analysis.

- Requirement: Preserve full git history for observatory files in the new package repo
  - Acceptance: `git log --follow observatory/taxonomy.py` in the new repo shows commits from before extraction.

## Design Considerations

### Monorepo vs Full Extraction

**Tension**: The prior art research recommends uv workspaces (iterate on interface in-repo before publishing). The migration analysis argues full extraction is simpler because no `pyproject.toml` exists yet.

**Resolution**: Start with Phase 1 (internal refactor) in the monorepo. This validates the interface changes risk-free. Then evaluate: if the refactored observatory has zero CSB imports, full extraction is clean. If coupling remains, use uv workspaces to iterate further.

### Plugin Architecture: Callables vs Entry Points

**Tension**: inspect_ai uses setuptools entry points for extension discovery. The technical analysis recommends simpler callable injection.

**Resolution**: Use callable injection (function parameters) for v0.x. If external consumer count grows beyond 3, consider entry points for v1.0. Callable injection is sufficient, simpler to debug, and doesn't require packaging infrastructure.

### OpenHands Cautionary Tale

OpenHands extracted `openhands-aci` as a separate PyPI package, then archived it and folded it back in. The lesson: premature extraction of tightly-coupled components creates maintenance overhead exceeding reuse benefits. Our mitigation: Phase 1 proves the interface is clean before we extract.

### Silent Degradation Risk

Without `suite_mapping.json`, the `benchmark` signal field is `None`, which silently disables the `missing_code_navigation` heuristic in `annotator.py`. The standalone package must log a warning when operating without benchmark context, so users understand the quality trade-off.

### Taxonomy Versioning

The taxonomy (v1, v2) versions independently from the package. The annotation schema includes a `taxonomy_version` field. Document the compatibility matrix: "agent-observatory 0.x supports taxonomy v1 and v2."

## Current State

The `agent-observatory` package lives at `~/agent-observatory/` as a standalone repo,
separate from CodeScaleBench (`~/CodeScaleBench/`). Phase -1 (taxonomy-first extraction)
is complete:

- `src/agent_observatory/` with taxonomy.py, taxonomy_v1.yaml, taxonomy_v2.yaml,
  annotation_schema.json, exemplars/, py.typed
- `pyproject.toml` (hatchling, PyYAML dep, optional extras for llm/validation/dev)
- 22 tests passing, 40KB wheel builds cleanly
- Uses `importlib.resources` for data file resolution (premortem P0 mitigation)
- Zero CSB coupling — completely standalone

## Build Phases

### Phase 1: Types & Heuristic Annotator [THIS REPO]

Add the data contract types and the heuristic annotator — the first module that
actually *does something* beyond taxonomy lookup.

1. Add `src/agent_observatory/types.py`:
   - `TrialSignals` TypedDict (26 keys from CSB's signals.py, generalized)
   - `TrialInput` Protocol defining what data observatory needs to consume [from premortem]
   - `CategoryAssignment`, `Annotation`, `AnnotationDocument` dataclasses
2. Add `src/agent_observatory/tool_registry.py`:
   - Injectable `ToolRegistry` replacing hardcoded `_SEARCH_TOOLS`, `_EDIT_TOOLS`,
     `_CODE_NAV_TOOLS`, `_SEMANTIC_SEARCH_TOOLS` frozensets [from premortem]
   - Ship a `DEFAULT_REGISTRY` with Claude Code + Sourcegraph MCP tools
   - External consumers can create custom registries for their agent's tool vocabulary
3. Port `annotator.py` from CSB — rewrite to consume `TrialSignals` + `ToolRegistry`
   instead of hardcoded assumptions
4. Add tests for annotator with both default and custom tool registries
5. Bump to v0.2.0

**Acceptance**: `from agent_observatory.types import TrialSignals` importable.
`from agent_observatory.annotator import annotate_trial` works with a hand-built
TrialSignals dict (no filesystem, no CSB).

### Phase 2: Signal Extraction [THIS REPO]

Port the generic parts of signals.py — the file-reading, trajectory parsing,
and pattern detection. Leave CSB-specific path parsing behind.

1. Port `src/agent_observatory/signals.py`:
   - `extract_signals(trial_dir, *, tool_registry, suite_mapping, metadata_resolvers)` [from premortem]
   - All CSB-specific logic (path parsing, model keywords) goes into optional
     `metadata_resolvers` callables, NOT hardcoded
   - Warn when running without suite_mapping (silent degradation fix) [from premortem]
2. Fix optional-dependency guards: `raise ImportError` not `logger.error` [from premortem]
3. Add tests with synthetic trial fixtures (not CSB data)
4. Bump to v0.3.0

**Acceptance**: `extract_signals()` runs against a hand-crafted trial dir with
`result.json` and `trajectory.json`. No CSB paths, no `sys.path` manipulation.

### Phase 3: Classifier & Ensemble [THIS REPO]

Port the pure-Python classifier and ensemble pipeline.

1. Port `src/agent_observatory/classifier.py` (pure Python, zero heavy deps)
   - Accept model weights from user-specified path (not bundled) [from premortem]
   - Commit to pure-Python only — no numpy dual code path [from premortem]
2. Port `src/agent_observatory/ensemble.py` — combines heuristic + classifier
3. Port `src/agent_observatory/calibrate.py` and `blend_labels.py`
4. Add tests
5. Bump to v0.4.0

### Phase 4: LLM Annotator & CLI [THIS REPO]

Port the LLM annotation pipeline and CLI. This is the highest-coupling module.

1. Port `src/agent_observatory/llm_annotator.py`:
   - Promote all private functions that CSB's diagnose.py imports to public API
   - Remove judge scoring entirely (stays in CSB)
   - Remove `sys.path` manipulation
   - Keep both `api` and `claude-code` backends
2. Port `src/agent_observatory/report.py`
3. Port `src/agent_observatory/cli.py` and `__main__.py`
4. Bump to v0.5.0 (feature-complete)
5. **PyPI publish gate in CI**: tarfile inspection fails on deny-list (JSON >10KB, .env) [from premortem]

### Phase 5: CSB Switchover [IN CSB REPO]

Wire CSB to depend on the published package.

1. Add `agent-observatory>=0.5` to CSB dependencies
2. Create `lib/csb/observatory_adapter.py`:
   - `CodeScaleBenchAdapter` implementing CSB-specific input mapping [from premortem]
   - Suite mapping loader, model keywords, path parsing
   - Judge integration (`_build_judge_input`, `judge_trial`) — stays in CSB
3. Update imports in `diagnose.py`, `report.py`, `generate_baseline_submission.py`, `calibration_report.py`
4. Replace `_REPO_ROOT / "observatory" / ...` paths with package API calls [from premortem]
5. Create root `pyproject.toml` for CSB's own dependencies [from premortem]
6. Update `Dockerfile.eval`
7. **Container integration test**: run `csb diagnose` against fixture, validate output [from premortem]
8. Remove `observatory/` directory from CSB
9. All existing CSB tests must pass identically

## Open Questions

1. **PyPI name availability**: Is `agent-observatory` taken? Check before first publish.
2. ~~**External consumers**~~ **RESOLVED**: Ship taxonomy-first now, add tooling iteratively.
3. **claude-code CLI backend**: Keep built-in or make pluggable? (Recommend: keep built-in, it's the primary use case)
4. ~~**annotations/ directory**~~ **RESOLVED**: Not included in package. CSB-specific run artifacts stay in CSB.
5. **Trained models**: Ship a default model with the classifier, or require users to train their own?

## Convergence Record

Three positions were debated: Extract Now, Monorepo First, Taxonomy-First Minimal.

**Resolved consensus**: (1) Phase 0 internal refactor is mandatory regardless of strategy. (2) A separate repo is the end state — disagreement was on timing. (3) Taxonomy/schema/types are the most stable layer. (4) Judge scoring is architecturally misplaced and must move to CSB.

**Winning strategy**: Monorepo First with Taxonomy-First's discipline. Tiered API stability (stable taxonomy + provisional tooling) published via uv workspace. Fastest path to PyPI, real API validation via CSB as consumer, no premature commitment to API boundaries.

**Preserved dissent**: Extract Now argues the coupling is concentrated enough (2/10+ modules) that a clean cut is feasible today. If Phase 0 refactor proves the boundary is cleaner than expected, skip straight to full extraction.

## Premortem Risk Summary

Five independent failure agents analyzed this project. **All five rated their failure scenario as Critical severity / High likelihood** — an unusually strong consensus that surfaced risks the original plan missed.

**Top 3 risks (by cross-cutting convergence):**

1. **The input contract is missing** (Technical + Scope + Operational): TrialSignals defines outputs but `signals.py` encodes 30+ hardcoded tool names, CSB directory layout, and Anthropic model keywords as implicit input assumptions. External adopters cannot use the package without rewriting signal extraction. **Mitigation**: TrialInput protocol + ToolRegistry + CodeScaleBenchAdapter (added to Phase 0).

2. **Sensitive data in tracked files** (Security + Operational): `observatory/annotations/` contains 17MB of tracked JSON with internal trial paths, config names, and behavioral evidence. `git subtree split` from main would carry sensitive history. **Mitigation**: Data classification inventory + MANIFEST.in exclusions + clean-state export instead of subtree split (added as Phase -1).

3. **Filesystem-path coupling** (Operational + Integration): 6+ locations in CSB resolve observatory data via `_REPO_ROOT / "observatory" / ...` paths that break when observatory is pip-installed. **Mitigation**: Replace with `importlib.resources` package API (added to Phase -1).

Full premortem: `premortem_agent_observatory.md`

## Research Provenance

This PRD was produced by a 3-agent divergent research process:

- **Prior Art & Ecosystem Patterns**: Researched HELM, inspect_ai, SWE-bench, OpenHands, lm-eval, BigCode. Found that no major ML eval framework has extracted its taxonomy layer as a standalone package. inspect_ai's framework/eval split is the closest precedent. OpenHands' failed extraction is a cautionary tale.
- **First-Principles Technical Architecture**: Read all observatory source files. Found coupling concentrated in 2/10+ modules. Designed the three-layer split and callable injection points. Discovered the pure-Python classifier (zero heavy deps) as a packaging advantage.
- **Migration Mechanics & Risk**: Identified missing pyproject.toml, proposed three-phase migration, caught the silent suite_mapping degradation bug, and designed the backwards-compatible switchover.

**Convergence**: All agents agreed on promoting private APIs, removing judge scoring from observatory, and phased migration. **Divergence**: Monorepo vs full extraction timing; callable injection vs entry points.
