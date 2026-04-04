# Premortem: Extract Observatory into Standalone `agent-observatory` PyPI Package

## Risk Registry

| #   | Failure Lens             | Severity     | Likelihood | Score  | Root Cause                                                                                                                                                            | Top Mitigation                                                                                                                 |
| --- | ------------------------ | ------------ | ---------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Technical Architecture   | Critical (4) | High (3)   | **12** | TrialSignals defines output contract but not input contract; signals.py encodes CSB directory layout, tool vocabularies, and path conventions as implicit assumptions | Define abstract TrialInput protocol + pluggable ToolRegistry; extract CSB-specific logic into CodeScaleBenchAdapter            |
| 2   | Security & Compliance    | Critical (4) | High (3)   | **12** | No data classification step; annotations/ contains internal trial paths, config names, evidence strings; git subtree split would carry sensitive history              | Data classification inventory before Phase 0; MANIFEST.in exclusions; NO git subtree split from main — clean-state export only |
| 3   | Operational              | Critical (4) | High (3)   | **12** | 6+ locations where CSB resolves observatory data files via `_REPO_ROOT / "observatory" / ...` filesystem paths instead of package API                                 | Replace all filesystem-path references with importlib.resources-based package API before any extraction                        |
| 4   | Integration & Dependency | Critical (4) | High (3)   | **12** | CSB has zero formal dependency management; adding observatory's pyproject.toml gives its pins uncontested control over shared deps                                    | Create root pyproject.toml declaring CSB's own dependencies; pin PyYAML with upper bound for v0.x                              |
| 5   | Scope & Requirements     | Critical (4) | High (3)   | **12** | TrialSignals TypedDict is a projection of CSB's directory layout and tool ecosystem, not a general-purpose interface                                                  | Separate taxonomy from tooling at package boundary; gate publish on one non-CSB integration test                               |

## Cross-Cutting Themes

### Theme 1: The Input Contract Is Missing (Lenses: Technical, Scope, Operational)

Three independent failure agents identified the same core vulnerability: **observatory's coupling to CSB runs far deeper than the 2-module surface suggests.** The TrialSignals TypedDict defines what signals.py _outputs_ but not what it _assumes_:

- `signals.py` hardcodes 30+ tool names in `_SEARCH_TOOLS`, `_EDIT_TOOLS`, `_CODE_NAV_TOOLS`, `_SEMANTIC_SEARCH_TOOLS` frozensets (Claude Code + Sourcegraph MCP specific)
- `_iter_trial_dirs()` hardcodes `_raw/` directory convention and `agent/` subdirectory requirement
- `_benchmark_from_path()`, `_model_from_path()`, `_config_from_path()` parse CSB-specific path segments
- `_MODEL_KEYWORDS` hardcodes 5 Anthropic model aliases
- `_detect_trajectory_patterns()` depends on the above tool name sets

**Combined severity**: This is the #1 risk. Every external adopter hits this wall. The coupling audit counted _import statements_ but missed _semantic assumptions_.

### Theme 2: Sensitive Data in Tracked Files (Lenses: Security, Operational)

The `observatory/annotations/` directory contains 5 tracked JSON files totaling 17MB with:

- Internal trial paths (`runs/official/_raw/...`)
- Configuration names and task IDs from private benchmarks
- Verbatim behavioral evidence strings from agent traces
- The repo's own CLAUDE.md warns main has "sensitive content"

`git subtree split --prefix=observatory` (planned for Phase 2/3) would carry this history into a public repo. The default `python -m build` would include these files in the sdist/wheel.

### Theme 3: Filesystem-Path Coupling (Lenses: Operational, Integration)

CSB code resolves observatory data files via filesystem paths, not package APIs:

- `lib/csb/diagnose.py:23`: `_SCHEMA_PATH = _REPO_ROOT / "observatory" / "annotation_schema.json"`
- `lib/csb/report.py:41`: `_REPO_ROOT / "observatory" / "taxonomy_v2.yaml"`
- `tests/test_taxonomy_v2_calibration.py`: 5 hardcoded `_REPO_ROOT / "observatory"` references

These silently break in containers or when observatory is pip-installed rather than a sibling directory.

### Theme 4: Silent Degradation Patterns (Lenses: Integration, Operational)

Observatory's optional dependency guards (anthropic, jsonschema) log errors but don't raise exceptions. When installed without extras, annotation pipelines produce empty results silently. Combined with deferred imports in CSB (inside function bodies), failures only surface at runtime during actual annotation — not at startup, not in CI.

## Mitigation Priority List

| Priority | Mitigation                                                                                | Failure Modes Addressed  | Implementation Cost |
| -------- | ----------------------------------------------------------------------------------------- | ------------------------ | ------------------- |
| **P0**   | Data classification inventory + MANIFEST.in exclusions for annotations/                   | Security, Operational    | Low                 |
| **P0**   | Replace `_REPO_ROOT / "observatory" / ...` paths with `importlib.resources` package API   | Operational, Integration | Medium              |
| **P0**   | Do NOT use `git subtree split` from main — clean-state export only (no history)           | Security                 | Low                 |
| **P1**   | Define `TrialInput` protocol + `CodeScaleBenchAdapter` for input-side abstraction         | Technical, Scope         | High                |
| **P1**   | Make tool name sets injectable via `ToolRegistry` (default to Claude Code tools)          | Technical, Scope         | Medium              |
| **P1**   | Create root `pyproject.toml` for CSB's own dependencies before adding workspace           | Integration              | Low                 |
| **P1**   | Change optional-dependency guards from silent `logger.error` to raising `ImportError`     | Integration, Operational | Low                 |
| **P2**   | Add CI import-boundary lint (fail if observatory/ imports outside observatory/)           | Technical, Integration   | Low                 |
| **P2**   | Add container integration test for `csb diagnose` pipeline end-to-end                     | Operational              | Medium              |
| **P2**   | Pin PyYAML with upper bound (`>=6.0,<7`) for v0.x releases                                | Integration              | Low                 |
| **P2**   | Gate Phase 1 publish on one non-CSB integration test (synthetic if needed)                | Scope                    | Medium              |
| **P2**   | Add PyPI publish gate in CI: tarfile inspection, deny-list for large JSONs                | Security                 | Low                 |
| **P3**   | Version-compatibility check: `csb diagnose` validates `observatory.__version__` at import | Operational              | Low                 |
| **P3**   | Commit to pure-Python OR numpy for classifier — not both code paths                       | Technical                | Low                 |
| **P3**   | Add pre-push hook blocking pushes to upstream on non-public branches                      | Security                 | Low                 |

## Design Modification Recommendations

### 1. Add Phase -1: Data Scrub & Filesystem Decoupling (before Phase 0)

**What**: Before any API refactoring, (a) classify and exclude sensitive annotation data, (b) replace all `_REPO_ROOT / "observatory"` filesystem paths in CSB with package-level data APIs.

**Addresses**: Security (data leak), Operational (path breakage), Integration (container failures)

**Effort**: 1-2 days. Low risk, high impact.

### 2. Redefine the Package Boundary Around Inputs, Not Just Outputs

**What**: The current plan defines `TrialSignals` TypedDict as the contract. This is necessary but insufficient. Add: (a) `TrialInput` protocol defining what data observatory needs to consume, (b) `ToolRegistry` for injectable tool name vocabularies, (c) `CodeScaleBenchAdapter` that implements the CSB-specific input mapping. Ship the adapter as a separate extras or keep it in CSB.

**Addresses**: Technical Architecture (input coupling), Scope (unusable for non-CSB consumers)

**Effort**: 3-5 days. This is the most important architectural change and should happen in Phase 0.

### 3. Formalize CSB's Own Dependencies Before Adding Workspace

**What**: Create a root `pyproject.toml` for CSB declaring its actual dependencies (PyYAML, anthropic, etc.) with version constraints. Do this _before_ adding observatory as a workspace member, so the resolver has complete information.

**Addresses**: Integration (resolver conflicts), Operational (silent PyYAML breakage)

**Effort**: 1 day. Pure configuration, zero code changes.

### 4. Abandon git subtree split — Use Clean Export

**What**: Instead of `git subtree split --prefix=observatory` (which carries full history including sensitive commits), do a clean-state export: copy current files into a fresh repo with a single initial commit. Accept the history loss as a security trade-off.

**Addresses**: Security (sensitive history exposure)

**Effort**: Trivial. Changes the extraction script, not the code.

### 5. Eliminate Silent Degradation

**What**: Change all optional-dependency import guards in observatory from `logger.error` + return None to `raise ImportError("pip install agent-observatory[llm]")`. Add `observatory.__version__` compatibility check in CSB's consumer code.

**Addresses**: Integration (empty annotation results), Operational (invisible failures)

**Effort**: 1 hour. Pure code cleanup.

## Revised Phase Plan (incorporating premortem mitigations)

```
Phase -1: Data Scrub & Path Decoupling          [P0 mitigations, 1-2 days]
  - Data classification inventory for annotations/
  - MANIFEST.in / pyproject.toml exclusions
  - Replace _REPO_ROOT paths with importlib.resources
  - Root pyproject.toml for CSB dependencies
  - Fix silent degradation in optional imports

Phase 0: Internal Refactor                       [P1 mitigations, 3-5 days]
  - Promote private APIs to public
  - Move judge scoring to CSB
  - Parameterize suite_mapping
  - Define TrialInput protocol + ToolRegistry
  - Add TrialSignals TypedDict
  - Add CodeScaleBenchAdapter
  - CI import-boundary lint

Phase 1: Workspace + Publish                     [Original plan, 2-3 days]
  - observatory/pyproject.toml as uv workspace member
  - Tiered API stability (stable taxonomy + provisional tooling)
  - PyPI publish gate (tarfile inspection)
  - Container integration test
  - Gate on one non-CSB integration test

Phase 2: Decision Gate                           [At 3 months or first external consumer]
  - Clean-state export (NOT git subtree split)
  - Separate repo if API stable + adoption exists
```

## Full Failure Narratives

_See individual agent outputs above for complete narratives._

| Lens                     | Key Quote                                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Technical Architecture   | "The extraction treated the observatory's output format as the API boundary, but the true coupling surface was the input assumptions"                           |
| Integration & Dependency | "CSB having zero formal dependency management and observatory introducing the repo's first pyproject.toml makes resolver conflicts near-certain"                |
| Operational              | "The schema file was no longer at the filesystem path the code expected... failures only surfaced in CI and in-container runs, not in local dev"                |
| Scope & Requirements     | "14 of the 26 signal keys were either None or meaningless for their trajectory formats... we'd published our implementation details as if they were a standard" |
| Security & Compliance    | "The annotation files had been git-tracked in the repository and were included in the sdist... git subtree split from main would carry full history"            |
