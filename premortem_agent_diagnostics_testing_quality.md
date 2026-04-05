# Premortem: agent-diagnostics v0.5.1 Testing & Quality Improvement

## Risk Registry

| #   | Failure Lens             | Severity     | Likelihood | Score | Root Cause                                                                               | Top Mitigation                                                                                        |
| --- | ------------------------ | ------------ | ---------- | ----- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 1   | Integration & Dependency | Critical (4) | High (3)   | 12    | Mock boundaries capture transport, not response-parsing contracts; no upper SDK pin      | Pin `anthropic<1.0`, capture real CLI envelope fixtures, replace hardcoded `taxonomy_v1.yaml`         |
| 2   | Team & Process           | Critical (4) | High (3)   | 12    | No phase gate between Phase 1 and Phase 2; parallel agents build on stale code           | Hard gate: Phase 1 merged + 278 tests green before any Phase 2 agent starts                           |
| 3   | Technical Architecture   | Critical (4) | High (3)   | 12    | Mocks at wrong abstraction level; refactored parser untested against realistic envelopes | Write `_parse_claude_response` contract tests with 5+ envelope fixtures before refactor               |
| 4   | Scope & Requirements     | High (3)     | High (3)   | 9     | Coverage target incentivizes shallow mocking over behavioral verification                | Define behavioral scenarios, not just % targets; resolve 3 open questions before Phase 2              |
| 5   | Operational              | High (3)     | High (3)   | 9     | CLI silent-success on empty results; hardcoded paths break on new benchmark formats      | Add end-to-end golden-path test; make extract_all warn/fail on 0 trials; parameterize trajectory path |

## Cross-Cutting Themes

### Theme 1: Silent Failure / Empty-List Returns (all 5 lenses)

Every lens identified the same architectural flaw: functions return `[]` or `None` on failure, which is indistinguishable from legitimate empty results. Specific instances:

- `annotate_trial_claude_code` returns `[]` on parse failure (technical, integration)
- `extract_all` returns `[]` on directory mismatch (operational)
- `_load_json` returns `None` on FileNotFoundError (operational)
- `cmd_llm_annotate` filter reduces to 0 trials silently (operational)
- Mocked tests return `[]` and pass regardless (scope)

**Combined severity**: Critical. This is the single most dangerous pattern in the codebase -- bugs produce valid-looking empty output instead of errors.

### Theme 2: Mock-Reality Drift (technical, integration, scope)

Three lenses independently concluded that mocking `subprocess.run` and `anthropic.Anthropic` with hardcoded JSON blobs creates tests that pass forever regardless of external changes. The claude CLI envelope, SDK API surface, and taxonomy format can all change without any test detecting it.

**Combined severity**: Critical. The test suite provides false confidence.

### Theme 3: Phase Coordination Gap (process, technical)

Two lenses identified that Phase 2 parallelization assumes Phase 1 is complete and merged, but no mechanism enforces this. Parallel agents building on pre-Phase-1 code will duplicate the refactoring work and produce merge conflicts in every taxonomy-touching test file.

**Combined severity**: Critical for execution. The plan's phasing is correct but unenforced.

### Theme 4: Deferred Open Questions as Hidden Blockers (scope, integration)

Two lenses flagged the 3 open questions (claude CLI envelope shape, asyncio.run in Jupyter, jsonschema dependency) as not-actually-optional. You can't write meaningful LLM annotator tests without knowing the input format.

**Combined severity**: High. Deferred questions become deferred bugs.

## Mitigation Priority List

| Priority | Mitigation                                                                                                | Failure Modes Addressed       | Severity | Cost   |
| -------- | --------------------------------------------------------------------------------------------------------- | ----------------------------- | -------- | ------ |
| 1        | **Hard phase gate**: Phase 1 must merge, 278 tests green, conftest.py exists before Phase 2 starts        | Process, Technical            | Critical | Low    |
| 2        | **Capture real claude CLI envelope** as test fixture before writing any LLM annotator tests               | Integration, Technical, Scope | Critical | Low    |
| 3        | **Write `_parse_claude_response` contract tests** with 5+ envelope variants before the Phase 1 refactor   | Technical, Integration        | Critical | Low    |
| 4        | **Pin `anthropic>=0.30,<1.0`** in pyproject.toml                                                          | Integration                   | Critical | Low    |
| 5        | **Add end-to-end golden-path test**: extract -> annotate -> report against a checked-in fixture directory | Operational, Technical, Scope | High     | Medium |
| 6        | **Make `extract_all` exit non-zero** (via CLI) when 0 trials found in a non-empty directory               | Operational                   | High     | Low    |
| 7        | **Replace hardcoded `taxonomy_v1.yaml`** in `_taxonomy_yaml()` with `load_taxonomy()` call                | Integration                   | Critical | Low    |
| 8        | **Resolve open questions** before Phase 2: envelope shape, asyncio.run behavior, jsonschema dep           | Scope, Integration            | High     | Medium |
| 9        | **Assign strict file ownership** in Phase 2: each agent owns exactly one test file, conftest.py frozen    | Process                       | Critical | Low    |
| 10       | **Reset taxonomy cache in test fixtures** to prevent cross-test pollution from `_cached_taxonomy` global  | Process                       | High     | Low    |
| 11       | **Parameterize trajectory path** in `cmd_llm_annotate` instead of hardcoding `agent/trajectory.json`      | Operational                   | High     | Low    |
| 12       | **Add install smoke test**: pip install into temp venv, run `observatory validate` against fixture        | Operational                   | Medium   | Medium |

## Design Modification Recommendations

### 1. Add contract tests for `_parse_claude_response` BEFORE refactoring (Priority 2+3)

**What**: Before Phase 1 extraction, capture a real `claude -p --output-format json` response, commit it as `tests/fixtures/claude_envelope_*.json` (multiple variants), and write tests that the current parsing code passes against them. Then extract the helper and verify the same tests pass.

**Addresses**: Technical (mock-reality drift), Integration (envelope format changes), Scope (deferred questions)

**Effort**: 1 hour added to Phase 1

### 2. Enforce hard phase gates with verification scripts (Priority 1+9)

**What**: Add a `scripts/phase_gate.sh` that runs `pytest -q && git diff --name-only main | head` to verify Phase 1 is clean before Phase 2 launch. Each Phase 2 agent must branch from post-Phase-1 main. Assign file ownership: CLI agent = test_cli.py only, etc.

**Addresses**: Process (stale baselines, merge conflicts), Technical (contradictory mocks)

**Effort**: 30 min added to Phase 1

### 3. Add behavioral scenario requirements alongside coverage % (Priority 5+8)

**What**: Supplement M1-M4 coverage targets with a checklist of specific scenarios:

- `cmd_llm_annotate` with trial whose `reward` is None
- `annotate_trial_claude_code` when claude CLI returns `is_error: true`
- `extract_all` against directory with 0 matching result.json files
- `_paired_comparison` with >= 20 shared tasks
- `_train_binary_lr` with all-same labels

**Addresses**: Scope (shallow mocking), Technical (edge cases), Operational (silent failures)

**Effort**: Tests take the same time; this just specifies WHICH behaviors to test

### 4. Fix silent-failure pattern in hot paths (Priority 6+7)

**What**: Make `extract_all` return a result object or raise when directory exists but 0 trials match. Replace `_taxonomy_yaml()` hardcoded filename with `load_taxonomy()`. Add logging at WARNING level in `_load_json` when files are missing.

**Addresses**: Operational (silent empty results), Integration (taxonomy version drift)

**Effort**: 1 hour, can be done in Phase 1

### 5. Pin SDK upper bound (Priority 4)

**What**: Change `anthropic>=0.30` to `anthropic>=0.30,<1.0` in pyproject.toml.

**Addresses**: Integration (breaking SDK changes)

**Effort**: 1 line change in Phase 0

## Full Failure Narratives

### 1. Technical Architecture Failure (Critical/High)

The mock boundaries were drawn at the transport layer (subprocess.run, anthropic.Anthropic) rather than the parsing contract layer. Phase 1's `_parse_claude_response` refactor changed parsing semantics without tests failing because mocks never produced realistic envelope variants. Coverage hit 80% but the fallback parsing path was never exercised. Production broke when claude CLI renamed `structured_output` to `structured_response`.

### 2. Integration & Dependency Failure (Critical/High)

Three dependency surfaces drifted simultaneously: anthropic SDK broke backward compatibility (no upper pin), claude CLI changed its JSON envelope format (no contract test), and taxonomy evolved from v1 to v2 while `_taxonomy_yaml()` hardcoded `taxonomy_v1.yaml`. All returned empty lists silently. Mocked tests stayed green throughout.

### 3. Operational Failure (High/High)

New benchmark formats used different file names (`evaluation_result.json`, `trajectory/steps.json`). `extract_all` returned empty lists silently, CLI exited 0, reports rendered valid empty Markdown. `cmd_validate` failed post-install because `annotation_schema.json` wasn't in package_data. No end-to-end test caught any of this.

### 4. Scope & Requirements Failure (High/High)

Coverage target incentivized shallow mocking over behavioral testing. CLI went from 6% to 70% via argparse.Namespace tests that mocked every downstream module. Three deferred open questions (CLI envelope shape, asyncio in Jupyter, jsonschema dep) all turned out to be critical. Phase 1 refactoring was cut short under pressure, leaving duplication intact.

### 5. Team & Process Failure (Critical/High)

Phase 2 agents started from pre-Phase-1 baselines. Four parallel agents wrote tests using hardcoded taxonomy strings that Phase 1 had replaced with conftest.py fixtures. Merge resolution took longer than the original test-writing. Taxonomy module-level cache (`_cached_taxonomy`, `_cached_path`) caused flaky tests post-merge. Coverage stalled at 78%.
