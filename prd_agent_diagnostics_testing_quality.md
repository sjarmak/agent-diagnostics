# PRD: agent-diagnostics v0.5.1 Testing & Quality Improvement

## Problem Statement

The agent-diagnostics package has solid core logic (taxonomy, annotator, ensemble, signals all at 90-100% coverage) but its user-facing layers are undertested: CLI at 6%, LLM annotator at 35%, report at 75%. Overall coverage sits at 69%, below the 80% production-readiness target.

Additionally, the repository has significant hygiene issues: 2,174 `.venv` files and 32 `__pycache__` bytecode files are tracked in git despite `.gitignore` rules (committed before the gitignore existed), stale `dist/` artifacts from the old `agent_observatory` package name remain in the index, and production-readiness files (LICENSE, coverage config) are missing.

## Goals & Non-Goals

### Goals

- Reach 80%+ overall line coverage with meaningful tests (not mock-heavy stubs)
- Clean git history of incorrectly tracked files (.venv, \_\_pycache\_\_, stale dist)
- Add coverage enforcement configuration so the target cannot regress
- Refactor duplicated LLM annotator parsing logic to reduce test surface

### Non-Goals

- 100% coverage -- async batch wrappers and `__main__.py` can remain uncovered if justified
- PyPI publication workflow (trusted publisher, twine, testpypi)
- CLI entry point rename (keep `observatory` for now)
- CI/CD pipeline (GitHub Actions) -- separate effort
- Adding new features or changing public API

## Implementation Phases

> Sequencing resolved via structured debate (3 independent advocates for tests-first,
> refactor-first, and hygiene-first converged on this ordering).

### Phase 0: Hygiene Foundation (30 min)

One atomic commit, zero source code changes, zero merge conflict risk.

| Req | What                    | Acceptance                                                                                                     |
| --- | ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| M6  | Purge tracked artifacts | `git ls-files .venv/` and `git ls-files dist/` and `git ls-files '*__pycache__*'` all return empty             |
| M7  | Complete .gitignore     | Contains: `.venv/`, `dist/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `.coverage`, `.pytest_cache/`, `htmlcov/` |
| S4  | Add LICENSE             | `LICENSE` file at repo root with Apache 2.0 full text                                                          |

> **RISK (premortem)**: Pin `anthropic>=0.30,<1.0` in pyproject.toml during this phase.
> No upper bound means breaking SDK releases install freely. (Integration failure, score 12/12)

**Why first**: The .gitignore is "decorative" -- every pattern it lists has tracked files that predate it. Any subsequent PR carries 2,174+ artifact files in diffs. Running `pytest --cov` generates `.coverage` and `htmlcov/` that will pollute `git status` without proper ignores.

### Phase 1: LLM Annotator Refactor + Helper Tests (2 hrs)

Two parallel tracks that merge before the main test sprint.

| Req          | What                                          | Acceptance                                                                                                                                                        |
| ------------ | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S2           | Extract `_parse_claude_response()`            | Duplicated JSON parsing in `annotate_trial_claude_code` and `_annotate_one_claude_code` replaced by shared helper; net line count reduction in `llm_annotator.py` |
| M2 (partial) | Test pure helpers                             | Tests for `_read_text`, `_load_json`, `_resolve_model_alias`, `_resolve_model_api` pass; these do not depend on the refactor                                      |
| NEW          | Capture real CLI envelope fixtures            | `tests/fixtures/claude_envelope_*.json` committed with 5+ variants (structured_output, raw fallback, is_error, code fences, empty result)                         |
| NEW          | Write `_parse_claude_response` contract tests | Contract tests pass against current code BEFORE extraction; same tests verify refactored helper                                                                   |
| NEW          | Replace hardcoded `taxonomy_v1.yaml`          | `_taxonomy_yaml()` uses `load_taxonomy()` or module constant instead of hardcoded filename                                                                        |
| NEW          | Reset taxonomy cache in test fixture          | Fixture resets `_cached_taxonomy = None` after each test class to prevent cross-test pollution                                                                    |

> **RISK (premortem)**: Phase 1 is the highest-risk phase. Three failure lenses (technical,
> integration, process) identified that refactoring without contract tests against realistic
> envelope fixtures would introduce silent regressions. Write contract tests BEFORE extracting
> the helper, not after. (Risk score: 12/12)

> **RISK (premortem)**: `_taxonomy_yaml()` hardcodes `taxonomy_v1.yaml` (line 90). When taxonomy
> evolves to v2, LLM annotator silently injects stale categories. Fix during this phase.
> (Integration failure, score 12/12)

**Why before Phase 2**: Writing LLM annotator backend tests against duplicated code means testing the same parsing logic twice (structured_output, raw fallback, is_error, bad JSON -- each tested in both functions). Extracting the helper first means tests are written once against clean code. The pure helper tests run in parallel with the extraction, raising the safety net from 35% before the refactor lands. Debate resolved: "refactoring earns the right to precede testing only when duplication is in the exact module being tested and the refactor is scoped enough to be safe."

> **PHASE GATE (premortem)**: Phase 1 MUST merge to main with all 278+ tests green and
> conftest.py with shared taxonomy fixtures present BEFORE any Phase 2 agent starts.
> Phase 2 agents must branch from post-Phase-1 main. Each agent owns exactly one test file.
> (Process failure, score 12/12)

### Phase 2: Coverage Sprint (3-4 hrs)

The bulk of the work. These are independent and can be parallelized with strict file ownership.

| Req | What                        | Acceptance                                                                                                                                |
| --- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| M1  | CLI tests >= 70%            | `pytest tests/test_cli.py --cov=agent_diagnostics.cli` shows >= 70%                                                                       |
| M2  | LLM annotator tests >= 60%  | `pytest tests/test_llm_annotator.py --cov=agent_diagnostics.llm_annotator` shows >= 60% (now against deduplicated code)                   |
| M3  | Report tests >= 85%         | `pytest tests/test_report.py --cov=agent_diagnostics.report` shows >= 85%                                                                 |
| M4  | Classifier tests >= 85%     | `pytest tests/test_classifier.py --cov=agent_diagnostics.classifier` shows >= 85%                                                         |
| M5  | Overall >= 80%              | `pytest tests/ --cov=agent_diagnostics` shows TOTAL >= 80%; all 278 existing tests still pass                                             |
| NEW | End-to-end golden-path test | One test runs extract -> annotate -> report against a checked-in fixture directory; asserts non-empty report with expected category names |
| NEW | CLI warns on 0 results      | `extract_all` / CLI exits non-zero when 0 trials found in non-empty directory                                                             |

> **RISK (premortem)**: Silent-failure pattern is the #1 cross-cutting risk (all 5 lenses).
> Functions return `[]` or `None` on failure, indistinguishable from legitimate empty results.
> The golden-path test and zero-result warning are the primary mitigations.

> **RISK (premortem)**: Coverage % alone incentivizes shallow mocking. Supplement M1-M4 with
> specific behavioral scenarios: `cmd_llm_annotate` with reward=None, `extract_all` on 0 trials,
> `_train_binary_lr` with all-same labels, `_paired_comparison` with >=20 shared tasks.
> (Scope failure, score 9/12)

> **File ownership for parallel agents**: CLI agent = test_cli.py only. LLM agent = test_llm_annotator.py only.
> Report agent = test_report.py only. Classifier agent = test_classifier.py only.
> No agent touches conftest.py (frozen from Phase 1).

### Phase 3: Enforcement & Polish (30 min)

Enable the coverage gate after reaching the target (not before -- `fail_under=80` while at 69% would fail every commit during the test-writing phase).

| Req | What                      | Acceptance                                                                                                                                         |
| --- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1  | Coverage enforcement      | `[tool.coverage.report] fail_under = 80` in pyproject.toml; `pytest --cov --cov-fail-under=80` exits 0                                             |
| S3  | Dynamic taxonomy fixtures | `conftest.py` fixture provides valid category names from `valid_category_names()`; at least `test_llm_annotator.py` and `test_annotator.py` use it |
| N1  | Fix stale repo URL        | `[project.urls] Repository` points to actual GitHub repo                                                                                           |

## Requirements Detail

### Must-Have

- **M1: CLI test coverage >= 70%** (Phase 2)
  - Test each of the 8 `cmd_*` functions via direct call with `argparse.Namespace` + `tmp_path`
  - Test error paths (missing file, invalid JSON) with `pytest.raises(SystemExit)`
  - Test `cmd_llm_annotate` sampling/assembly logic with mocked `annotate_trial_llm`

- **M2: LLM annotator test coverage >= 60%** (Phase 1 helpers + Phase 2 backends)
  - Test `annotate_trial_claude_code` with mocked `subprocess.run` for: success (structured_output), success (raw JSON fallback), subprocess failure, timeout, bad JSON, `is_error` response
  - Test `annotate_trial_api` with mocked `anthropic.Anthropic` for: success, API error, non-list response
  - Test pure helpers: `_read_text`, `_load_json`, `_resolve_model_alias`, `_resolve_model_api`

- **M3: Report test coverage >= 85%** (Phase 2)
  - Test `_core_task_name` with parametrized inputs covering all prefix patterns
  - Test `_paired_comparison` with fixture of 25+ annotations across 2 configs sharing 20+ tasks
  - Test `_top_categories_with_examples` and `_category_by_suite` as unit functions
  - Test "no success annotations" branch in `_render_markdown`

- **M4: Classifier test coverage >= 85%** (Phase 2)
  - Test `format_eval_markdown` output format
  - Test `predict_all` with a trained model
  - Test `_to_float` edge cases (unconvertible string)
  - Test `_train_binary_lr` with degenerate inputs (all-same labels, zero-variance features)

- **M5: Overall coverage >= 80%** (Phase 2 gate)
  - All existing 278 tests continue to pass

- **M6: Purge incorrectly tracked files from git index** (Phase 0)
  - `git rm -r --cached .venv/ dist/ '*/__pycache__/' '**/__pycache__/'`

- **M7: Complete .gitignore** (Phase 0)
  - Entries for `.venv/`, `dist/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `.coverage`, `.pytest_cache/`, `htmlcov/`

### Should-Have

- **S1: Coverage enforcement in pyproject.toml** (Phase 3)
  - `[tool.coverage.report]` with `fail_under = 80`
  - `[tool.coverage.run]` with `source = ["agent_diagnostics"]`

- **S2: Refactor duplicated LLM annotator parsing** (Phase 1)
  - Extract `_parse_claude_response(raw_json)` shared helper
  - Net reduction in `llm_annotator.py` line count

- **S3: Dynamic taxonomy references in tests** (Phase 3)
  - `conftest.py` fixture from `valid_category_names()`

- **S4: Add LICENSE file** (Phase 0)
  - Apache License 2.0 full text at repo root

### Nice-to-Have

- **N1: Fix stale pyproject.toml repository URL** (Phase 3)
- **N2: Add py.typed marker**
- **N3: Test async batch functions in llm_annotator**

## Design Considerations

### Test Philosophy: Mocks vs. Integration

The LLM annotator tests require mocking external services (subprocess for claude CLI, anthropic SDK). The risk is "testing mocks" -- tests that pass regardless of real behavior. Mitigate by:

- Using realistic response fixtures that match actual API/CLI output shapes
- Testing the parsing/validation layer, not the I/O layer
- Keeping mocks scoped to the outermost boundary (mock `subprocess.run`, not internal helpers)

### Coverage Target Rationale

80% overall is achievable by focusing on the 4 under-covered modules. Math:

- Current: 1184/1709 statements covered (69%)
- Needed: 1367/1709 (80%) -> 183 more statements
- Available in CLI alone: 202 missed statements; at 70% -> +137 covered
- Remaining ~46 from report + classifier improvements -> achievable

### Duplicated Code Risk (resolved)

`annotate_trial_claude_code` and `_annotate_one_claude_code` share ~80 lines of JSON parsing logic. Debate resolved: extract `_parse_claude_response()` in Phase 1 before writing backend tests in Phase 2. The 35% existing coverage plus new pure-helper tests provide the safety net for the extraction. Tests are then written once against clean code.

### Sequencing Rationale (from convergence debate)

Three advocates (tests-first, refactor-first, hygiene-first) debated and converged:

- **Hygiene is non-negotiable Phase 0** -- all three agreed. The repo is ~94% build artifacts by file count.
- **Refactor-before-test applies narrowly** -- only to `llm_annotator.py` where duplication exists. CLI, report, and classifier tests are written against existing code (tests-first wins there).
- **`fail_under=80` is a capstone, not a foundation** -- you can't gate at 80% while building toward it. Enable after reaching the target.

## Open Questions

1. ~~Should hygiene precede testing?~~ **Resolved**: Yes, unanimously.
2. ~~Should LLM annotator be refactored before testing?~~ **Resolved**: Yes, for the parsing duplication only; pure helpers are tested independently.
3. ~~When should `fail_under=80` be enabled?~~ **Resolved**: After reaching 80% (Phase 3), not before.
4. What is the exact shape of the `claude -p --output-format json` envelope? Mock fixtures need to match reality.
5. Does `asyncio.run()` in batch functions work inside existing event loops (e.g., Jupyter)?
6. Should `cmd_validate` tests require `jsonschema` as a test dependency, or should it be mocked?
7. Is the repository actually at `sourcegraph/agent-observatory` or has it moved?

## Research Provenance

### Divergent Research (3 independent agents)

1. **Test Architecture & Strategy** -- identified specific untested code paths in all 4 modules, recommended direct function calls over subprocess for CLI tests, and found that CLI coverage is "deceptively easy" (uniform `cmd_*` pattern).

2. **Repo Hygiene & Production Readiness** -- discovered 2,174 tracked `.venv` files (repo is ~94% venv by file count), incomplete package rename artifacts, and missing production files.

3. **Failure Modes & Risks** -- found duplicated parsing logic in LLM annotator (~80 lines copy-pasted), fragile taxonomy name coupling across 59 test references in 8 files, and silent failure modes (broad `except Exception` returning empty lists).

### Convergence Debate (3 advocates, 2 rounds)

**Positions**: Tests-first, Refactor-first, Hygiene-first.

**Key resolution**: All three independently proposed the same 3-phase sequence in Round 2 synthesis offers. The only substantive disagreement was whether 35% coverage is "enough safety net" for refactoring -- resolved by writing pure helper tests in parallel with the extraction.

**Strongest preserved arguments**:

- Tests-first: "Refactoring earns the right to precede testing only when duplication is in the exact module being tested."
- Refactor-first: "Total work is strictly less -- test once against clean code."
- Hygiene-first: "The .gitignore is decorative -- it prevents nothing because everything was already committed."

### Premortem (5 failure lenses)

**Lenses**: Technical Architecture, Integration & Dependency, Operational, Scope & Requirements, Team & Process.

**Top 3 risks** (all scored 12/12 -- Critical severity, High likelihood):

1. Mock-reality drift: mocked CLI envelopes and SDK responses diverge from production without any test detecting it
2. No phase gate: parallel Phase 2 agents build on stale pre-Phase-1 code, producing merge conflicts and duplicated work
3. Silent failures: functions return `[]`/`None` on error, indistinguishable from legitimate empty results (identified by all 5 lenses)

**Key design modifications added**: contract tests before refactor, real CLI envelope fixtures, hard phase gate, file ownership for parallel agents, golden-path end-to-end test, zero-result warnings.

Full premortem: [premortem_agent_diagnostics_testing_quality.md](premortem_agent_diagnostics_testing_quality.md)
