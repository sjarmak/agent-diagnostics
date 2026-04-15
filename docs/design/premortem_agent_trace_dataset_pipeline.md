# Premortem: Agent Trace Dataset Pipeline

## Risk Registry

| #   | Failure Lens             | Severity     | Likelihood | Score  | Root Cause                                                                                                  | Top Mitigation                                                                                                       |
| --- | ------------------------ | ------------ | ---------- | ------ | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 1   | Technical Architecture   | Critical (4) | High (3)   | **12** | Brittle `agent_info` key-existence check with no versioned schema; `None > 0.5` evaluates silently to False | Versioned schema validator + explicit None-safety audit of all 32 heuristic rules                                    |
| 2   | Integration & Dependency | Critical (4) | High (3)   | **12** | No schema contracts or integration tests between pipeline and CodeScaleBench data                           | Integration smoke test against real `_raw/` data; MANIFEST version check; warn on unknown agent frameworks           |
| 3   | Operational              | Critical (4) | High (3)   | **12** | No orchestration, staleness detection, or contract enforcement between pipeline stages                      | Pipeline manifest (Makefile/toml) with stage dependencies; corpus_hash in model.json; mandatory validate step        |
| 4   | Scale & Evolution        | Critical (4) | High (3)   | **12** | Pipeline assumes static, homogeneous corpus at fixed scale                                                  | Partition JSONL by benchmark/period; cache at (content-hash, taxonomy-version); trajectory normalization abstraction |
| 5   | Scope & Requirements     | High (3)     | High (3)   | **9**  | No stakeholder discovery before committing to analysis views                                                | 5-question structured interviews with 2+ reps per audience; quantify has_trajectory coverage bias                    |

## Cross-Cutting Themes

### Theme 1: Silent failure is the dominant failure mode (all 5 lenses)

Every narrative describes the pipeline producing output that looks normal but is wrong. `None > 0.5` returns False silently. Ghost rows produce zero-signal rows indistinguishable from real data. New directory layouts zero out trajectories without errors. Stale classifiers predict confidently wrong. **This is the project's #1 vulnerability.** The pipeline has no anomaly detection, no health gates, no staleness checks.

### Theme 2: CodeScaleBench is a moving target (tech, integration, ops)

Three lenses independently identified that the sibling-repo dependency — raw filesystem paths, no version pinning, no schema contracts — is fragile. Directory layouts have already changed multiple times (evidenced by `__pre_sgenv_fix`, `__archived_invalid` suffixes). MANIFEST.json is at 23% coverage and evolving. New agent frameworks keep appearing. The pipeline treats all of this as stable.

### Theme 3: The "frozen artifacts" never get unfrozen (ops, tech, scale)

The 500 Haiku labels, model.json classifier, and taxonomy version are all frozen with vague "refresh after cleanup" triggers. Three lenses predict these artifacts become permanently stale because no concrete staleness threshold or calendar trigger exists. The ops narrative specifically describes `model.json` being 6+ months old with no mechanism to detect this.

### Theme 4: has_trajectory partition bias (scope, tech)

The convergence debate's key insight — using `has_trajectory=True` as a clean partition — creates its own bias. Scope-lens found it silently excludes most non-SWE-bench trials. Tech-lens found it can flip to False for all new runs if directory layouts change. The partition is a useful shortcut but not a substitute for proper quality gates.

## Mitigation Priority List

| Priority | Mitigation                                                                                                                                                                  | Failure modes addressed | Effort |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------ |
| **1**    | **None-safety audit**: grep all 32 heuristic rules + classifier + ensemble + report for bare `reward ==`, `reward >`, `reward <` comparisons. Wrap in explicit None checks. | Tech, Ops               | Low    |
| **2**    | **Post-extraction health gate**: assert valid-trial ratio > 40%, has_trajectory ratio within 10% of baseline, benchmark non-null rate > 80%. Halt pipeline on violation.    | Tech, Integration, Ops  | Low    |
| **3**    | **Integration smoke test**: CI or pre-ingest test against real `_raw/` sample (10 trials) asserting schema expectations.                                                    | Integration, Tech       | Low    |
| **4**    | **Corpus hash in model.json**: embed SHA256 of training signals. `predict`/`ensemble` warn on mismatch. Add `--max-age` flag.                                               | Ops, Tech               | Low    |
| **5**    | **MANIFEST version check**: parse `version` field; reject unrecognized schemas. Type-check mapping values.                                                                  | Integration             | Low    |
| **6**    | **Unknown-framework warning**: log warning when `agent_info.name` doesn't match any registry instead of silent DEFAULT_REGISTRY fallback.                                   | Integration, Scale      | Low    |
| **7**    | **Pipeline orchestration** (Makefile or toml): declare stage dependencies, input/output paths, staleness rules.                                                             | Ops                     | Medium |
| **8**    | **has_trajectory bias quantification**: compute coverage by benchmark and model; embed in report metadata.                                                                  | Scope, Tech             | Low    |
| **9**    | **Stakeholder question backlog**: 5-question interviews with 2+ reps per audience before locking Phase 1 analysis views.                                                    | Scope                   | Medium |
| **10**   | **Partition JSONL by benchmark/period**: avoid full-corpus in-memory loads at scale.                                                                                        | Scale                   | Medium |
| **11**   | **Trajectory normalization abstraction**: canonical format converter per framework, run before heuristic checkers.                                                          | Scale, Integration      | High   |

## Top 5 Design Modifications

### 1. Add pipeline health gates (addresses: Tech, Integration, Ops)

After extraction, before annotation: assert valid-trial ratio, has_trajectory ratio, benchmark coverage, and reward-null rate are within expected bounds. Halt with explicit error on violation. This single change prevents the dominant failure mode (silent degradation) across three lenses.

**Effort:** Low — ~50 lines in `extract_all()` or a new `_validate_extraction()` function.

### 2. None-safety audit of all downstream consumers (addresses: Tech, Ops)

Before shipping MH-2 (nullable reward), systematically grep and fix every comparison against `reward` in annotator.py, classifier.py, ensemble.py, report.py, and any notebooks. Python's `None > 0.5 == False` is the single most likely source of silent corruption.

**Effort:** Low — mechanical grep-and-fix, but must be exhaustive.

### 3. Integration smoke test against real data (addresses: Integration, Tech)

Add a test (CI or pre-ingest) that runs `extract_signals()` on 10 real trial directories from `_raw/` and asserts: schema keys present, has_trajectory rate > 0, benchmark non-null, tool_calls > 0 for trajectory trials, no garbage benchmark strings.

**Effort:** Low — but requires a stable sample of real data checked into the test fixtures or a known-good path.

### 4. Corpus-model version binding (addresses: Ops, Tech)

Embed `corpus_hash` (SHA256 of signals file) and `extracted_at` timestamp in model.json during training. Have `predict`/`ensemble` commands check this hash against the current signals file and warn on mismatch. Add `--max-age` flag to reject stale models.

**Effort:** Low — metadata fields in train output, assertion in predict/ensemble.

### 5. Stakeholder question backlog before Phase 1 analysis lock (addresses: Scope)

Before committing to co-occurrence matrix and dimension rollup as primary outputs, conduct structured interviews with 2+ benchmark designers, agent developers, and model evaluators. Document their top 3 questions each. Validate that the planned analysis views actually answer them.

**Effort:** Medium — calendar time, not code time. But prevents building the wrong analysis.
