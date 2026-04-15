# Premortem: Behavioral Taxonomy & Annotation Improvements

## Risk Registry

| #   | Failure Lens             | Severity     | Likelihood | Score  | Root Cause                                                                                    | Top Mitigation                                           |
| --- | ------------------------ | ------------ | ---------- | ------ | --------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1   | Technical Architecture   | Critical (4) | High (3)   | **12** | Dual backends without shared abstraction; 5 MHs modify same file with no common interface     | Extract `LLMBackend` Protocol before Wave 2              |
| 2   | Integration & Dependency | Critical (4) | High (3)   | **12** | Hardcoded model IDs + CLI flags with no contract stability; silent `return []` on API failure | Dynamic model resolution + loud failures                 |
| 3   | Scope & Requirements     | Critical (4) | High (3)   | **12** | Taxonomy categories derived from papers, not validated against real trajectories              | Category validation sprint on 50 real trials             |
| 4   | Team & Process           | Critical (4) | High (3)   | **12** | No cross-unit contract tests; agents interpret `signal_dependencies` format differently       | Shared contract fixture checked in before Wave 1         |
| 5   | Security & Correctness   | Critical (4) | High (3)   | **12** | XML delimiter quarantine bypassable; redaction gap between signals.py and build_prompt        | Nonce-delimited boundary + shared REDACTED_SIGNAL_FIELDS |

All five lenses independently rated Critical/High. This is a strong signal that the PRD has systemic risks that must be addressed before execution.

## Cross-Cutting Themes

### Theme 1: Dual Backend Without Shared Abstraction (Arch + Deps + Process)

Three of five lenses independently identified that `llm_annotator.py` contains two fully duplicated pipelines (CLI subprocess + SDK) with no shared interface. This causes:

- Every cross-cutting concern (cache, structured output, AnnotationResult) must be implemented twice (Arch)
- Every Anthropic API change requires double migration with no shared adapter (Deps)
- Multi-agent merge conflicts multiply when 5 MHs modify the same file with parallel code paths (Process)

**Combined severity**: This is the single highest-risk structural issue. The PRD explicitly deferred the `LLMBackend` Protocol (MH-12) but scheduled 5 MHs against the file that needs it most.

### Theme 2: Signal Redaction Gap (Scope + Security)

Two lenses found that MH-5 (signal redaction in `signals.py`) and `build_prompt()` in `llm_annotator.py` apply independent filtering with no shared constant. The `build_prompt` function at line 222-228 constructs its own filtered dict excluding only `tool_calls_by_name` and `trial_path` — reward, passed, exception_crashed still reach the LLM judge. Units A (MH-5) and E (MH-6) are declared independent, but both affect what the judge sees.

**Combined severity**: The entire motivation for MH-5 (prevent label leakage) is undermined if redaction doesn't happen at the `build_prompt` boundary.

### Theme 3: Cache Correctness Tied to AnnotationResult Scope (Arch + Security)

Two lenses found that the cache (MH-8) stores raw `list[dict]` because `AnnotationResult` (MH-11) is scoped to `types.py` + `llm_annotator.py` only and unwrapped at the module boundary. This means:

- Cache cannot distinguish "no categories found" from "LLM was manipulated into returning empty" (Security)
- Cache cannot distinguish "LLM returned empty" from "LLM call timed out and returned error" (Arch)
- Taxonomy version/content drift can cause cache hits that serve stale results (Arch)

**Combined severity**: Cache poisoning is permanent (no TTL, no invalidation) and silent.

### Theme 4: Taxonomy Not Validated Against Real Data (Scope only, but critical)

Only one lens surfaced this, but it's the deepest risk: 17+ new categories were derived from research papers without testing against a single real trajectory from the target benchmarks. The PRD explicitly defers the golden corpus. Synthetic fixtures are tautological (written to match category definitions). Categories like `hallucinated_api` may trigger on zero real trajectories.

### Theme 5: Multi-Agent Coordination on llm_annotator.py (Process + Arch)

Two lenses found that the sequential ordering constraint (MH-1→MH-6→MH-8→MH-9→MH-11) is necessary but insufficient. Agents can start from stale worktrees, resolve merge conflicts by dropping earlier fixes, and produce tautological tests that pass numerically but miss cross-unit contract violations.

## Mitigation Priority List

Ranked by: failure modes addressed x severity x implementation cost.

| #   | Mitigation                                                                                                                                                                                                             | Failure Modes Addressed | Cost   | Priority                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------ | ------------------------ |
| 1   | **Shared `REDACTED_SIGNAL_FIELDS` frozenset** imported by both `signals.py` and `build_prompt()` in `llm_annotator.py`; MH-5 acceptance must verify final rendered prompt excludes reward                              | Scope, Security         | Low    | **Immediate**            |
| 2   | **Extract minimal `LLMBackend` Protocol** (just `annotate()` + `parse()`) before Wave 2, even if MH-12 full split is deferred                                                                                          | Arch, Deps, Process     | Medium | **Immediate**            |
| 3   | **Shared contract fixture** (`tests/contracts/taxonomy_v3_contract.py`) defining exact `signal_dependencies` format, `valid_category_names()` return type, and `AnnotationResult` semantics — checked in before Wave 1 | Process, Scope          | Low    | **Immediate**            |
| 4   | **Nonce-delimited quarantine boundary** or base64-encode trajectory content; add delimiter-breakout test to MH-6                                                                                                       | Security                | Low    | **Immediate**            |
| 5   | **Propagate `AnnotationResult` into cache layer** — never cache `Error`, flag `NoCategoriesFound` distinctly                                                                                                           | Arch, Security          | Low    | **Before MH-8**          |
| 6   | **Make annotation failures loud** — replace `return []` on backend errors with `AnnotationResult.Error`; never silently degrade                                                                                        | Deps, Security          | Low    | **Before MH-8**          |
| 7   | **Purely content-addressed cache key** — `sha256(rendered_prompt_text + model_id)` instead of mixing semantic version strings with content hashes                                                                      | Arch                    | Low    | **In MH-8**              |
| 8   | **Category validation sprint** — label 50 real trajectories with proposed categories, measure inter-annotator kappa, cut categories with zero positives                                                                | Scope                   | Medium | **Before MH-10**         |
| 9   | **Reconcile `FEATURE_NAMES` in classifier.py with `TrialSignals` keys** — identify features that are never populated by `extract_signals()`                                                                            | Scope                   | Low    | **Before MH-3**          |
| 10  | **API smoke test** (`@pytest.mark.integration`, skipped without credentials) hitting real endpoint for both backends                                                                                                   | Deps                    | Low    | **In MH-9**              |
| 11  | **Dynamic model ID resolution** via `anthropic.models.list()` with fallback to hardcoded map                                                                                                                           | Deps                    | Medium | **In MH-9**              |
| 12  | **Cross-unit regression test** in Wave 1 exit gate: load taxonomy → valid_category_names → filter derived_from_signal → feed into blend() with heuristic-only trial → assert non-empty                                 | Process                 | Low    | **Wave 1 gate**          |
| 13  | **Merge-base assertion** for `llm_annotator.py` — each unit's PR includes expected SHA of last-known-good version                                                                                                      | Process                 | Low    | **In /prd-build config** |

## Design Modification Recommendations

### R1: Add MH-0: Redaction + Contract Foundation (pre-Wave 1)

**What**: Before Wave 1 starts, create:

- `src/agent_diagnostics/constants.py` with `REDACTED_SIGNAL_FIELDS` frozenset
- `tests/contracts/taxonomy_v3_contract.py` defining exact schema expectations
- Fix `build_prompt()` to use `REDACTED_SIGNAL_FIELDS` for its signal filtering

**Addresses**: Security (redaction gap), Process (contract ambiguity), Scope (redaction not at correct boundary)
**Effort**: Small — 2-3 files, ~100 lines

### R2: Add MH-0.5: Minimal LLMBackend Protocol (pre-Wave 2)

**What**: Extract a `LLMBackend` Protocol with `annotate(prompt, model) -> RawResponse` and `parse(response) -> list[dict]` from the existing dual backends. Implement `CLIBackend` and `APIBackend`. Keep everything in `llm_annotator.py` (no package split) but behind the protocol.

**Addresses**: Arch (dual implementation), Deps (single adapter to patch), Process (merge conflicts on parallel code paths)
**Effort**: Medium — refactoring within existing file, ~200 lines changed

### R3: Strengthen MH-6 quarantine to nonce-delimited boundary

**What**: Replace `<untrusted_trajectory>` with `<untrusted_trajectory boundary="RANDOM_NONCE">` and validate that the closing tag includes the matching nonce. Add delimiter-breakout test.

**Addresses**: Security (injection via delimiter breakout)
**Effort**: Small — 10 lines in `build_prompt`, 1 new test

### R4: Extend AnnotationResult to cache layer in MH-8

**What**: Don't unwrap `AnnotationResult` before caching. Cache `Ok` results only. `NoCategoriesFound` gets cached with a flag. `Error` is never cached.

**Addresses**: Arch (cache correctness), Security (cache poisoning)
**Effort**: Small — MH-11 scoping change, ~30 lines in cache module

### R5: Category validation gate before MH-10

**What**: Before adding the 6 new categories, run a lightweight validation: label 20-50 real trajectories, confirm at least 3 of 6 categories have true positives. Cut categories with zero matches.

**Addresses**: Scope (building categories nobody uses)
**Effort**: Medium — requires access to real trajectory data, 1-2 days human effort

## Full Failure Narratives

### 1. Technical Architecture Failure

The project collapsed around `llm_annotator.py`. Wave 2 revealed that MH-8 (caching) and MH-9 (structured output) required fundamentally different implementations for CLI vs SDK backends, with no shared abstraction. Cache key formula mixed semantic versions with content hashes, causing stale hits when taxonomy files changed between runs. Two independent taxonomy access paths (`load_taxonomy()` cached vs `_taxonomy_yaml()` uncached) caused version/content drift. The deferred `LLMBackend` Protocol meant every cross-cutting concern was implemented twice with subtly different semantics.

**Root cause**: Deferring the `LLMBackend` Protocol while scheduling 5 must-haves against the un-abstracted dual-pipeline file.
**Severity**: Critical | **Likelihood**: High

### 2. Integration & Dependency Failure

Anthropic shipped a breaking CLI change (replacing `--json-schema` with `--tool-use`), deprecated short model aliases, and released `anthropic==1.0` with a redesigned return type — all in one quarter. The CLI backend silently returned `[]` on every failure. Hardcoded model IDs in `_API_MODEL_MAP` went stale. The cache served annotations keyed to now-invalid model IDs. No smoke test hit real endpoints.

**Root cause**: Two independent Anthropic integration paths with hardcoded model IDs and no contract test against the real API.
**Severity**: Critical | **Likelihood**: High

### 3. Scope & Requirements Failure

Taxonomy v3 categories derived from MAST/SWE-bench/tau-bench papers saw near-zero adoption because they didn't match real trajectory failure modes. `hallucinated_api` triggered on zero real trajectories. `signal_dependencies` metadata was hand-authored and produced an empty trusted set when used in `blend_labels.py`, causing a 12-point F1 drop. The `AnnotationResult` reduced scope created a permanent unwrap seam.

**Root cause**: Taxonomy designed from papers, not validated against real trajectories from target benchmarks.
**Severity**: Critical | **Likelihood**: High

### 4. Team & Process Failure

Wave 1 sequential chain broke down when agents started from stale worktrees. Unit E resolved merge conflicts by dropping Unit C's taxonomy access fix. Unit D interpreted `signal_dependencies` format differently from Unit C. Acceptance tests were tautological — agents tested their own implementations against their own fixtures. The 782-line mocked test file passed with zero failures despite 3 units modifying the file it tests.

**Root cause**: No machine-readable interface contract between taxonomy producer (Unit C) and consumers (Units D, E, F, H).
**Severity**: Critical | **Likelihood**: High

### 5. Security & Correctness Failure

`<untrusted_trajectory>` quarantine was bypassable via embedded closing tags in trajectory content. Signal redaction in `signals.py` didn't propagate to `build_prompt()` which applied its own independent filtering. Cache stored raw `list[dict]` with no way to distinguish legitimate empty results from manipulated ones. Poisoned cache entries persisted indefinitely.

**Root cause**: XML delimiter quarantine without escaping in a system processing untrusted content; redaction at wrong boundary.
**Severity**: Critical | **Likelihood**: High
