# PRD Build Log: Behavioral Taxonomy & Annotation Improvements

## 2026-04-11T00:00:00Z — Decomposition complete — 9 units across 5 layers

| Layer | Units                                                                       |
| ----- | --------------------------------------------------------------------------- |
| 0     | unit-redaction-constants, unit-fake-backend-fixtures                        |
| 1     | unit-signal-redaction, unit-taxonomy-v3                                     |
| 2     | unit-classifier-blend-ensemble, unit-prompt-quarantine, unit-new-categories |
| 3     | unit-annotation-cache                                                       |
| 4     | unit-annotation-result-type                                                 |

Baseline: 377 tests, 91% coverage.

## Layer 0 landed: unit-redaction-constants (b453f7d), unit-fake-backend-fixtures (3a3b8ab)

- constants.py with REDACTED_SIGNAL_FIELDS, contract tests, build_prompt redaction
- FakeLLMBackend, 14 per-category fixture dirs, determinism tests
- 468 tests passing

## Layer 1 landed: unit-signal-redaction (e659f9b), unit-taxonomy-v3 (ae996c9)

- judge_safe_signals() filter, signal leakage tests
- validate_categories bug fix, taxonomy_v3.yaml (11 dimensions, 37 categories)
- 514 tests passing

## Layer 2 landed: unit-classifier-blend-ensemble (5fbd3c5), unit-prompt-quarantine (014d677), unit-new-categories (3237728)

- Derived category exclusion from classifier training, metadata-driven blend trust
- Nonce-delimited untrusted_trajectory quarantine with breakout tests
- 6 new categories (reward_hacking, fabricated_success, hallucinated_api, tool_argument_error, premature_termination, verification_skipped) with 12 fixtures + heuristic checkers
- 543 tests passing

## Layer 3 landed: unit-annotation-cache (c4a699b + 6131ae4)

- SHA-256 content-hash cache, API backend tool-use structured output, model-ID alignment guard
- 562 tests passing

## Layer 4 landed: unit-annotation-result-type (a4e4a73)

- AnnotationResult discriminated union (Ok|NoCategoriesFound|Error), mypy clean
- Error results never cached, list[dict] unwrap at module boundary
- 585 tests passing

## PRD build complete

- **9/9 units landed**, 0 evictions, 1 pass
- **585 tests** (was 377), all passing
- Integration branch: `prd-build/behavioral-taxonomy-v3`

---

# PRD Build Log (previous): Testing & Quality Improvement

## 2026-04-05T00:00:00Z — Decomposition complete — 8 units across 4 layers

| Layer | Units                                                                                     |
| ----- | ----------------------------------------------------------------------------------------- |
| 0     | unit-hygiene                                                                              |
| 1     | unit-llm-refactor                                                                         |
| 2     | unit-cli-tests, unit-llm-tests, unit-report-tests, unit-classifier-tests, unit-e2e-golden |
| 3     | unit-enforcement                                                                          |

Baseline: 278 tests, 69% coverage. Target: 80%+.

## 2026-04-05T00:01:00Z — Layer 0 landed: unit-hygiene (commit 83ab723)

- Purged 3,232 tracked artifacts, completed .gitignore, added LICENSE, pinned anthropic<1.0

## 2026-04-05T00:05:00Z — Layer 1 landed: unit-llm-refactor (commit c2bce8c)

- Extracted \_parse_claude_response(), 6 fixtures, 21 new tests (278→299)

## 2026-04-05T00:06:00Z — Layer 2 dispatched: 5 parallel agents

- cli-agent, llm-agent, report-agent, classifier-agent, e2e-agent

## Layer 2 landed: all 5 units

- unit-cli-tests: 99% coverage (target 70%), 25 tests added (commit cbec994)
- unit-llm-tests: 66% coverage (target 60%), 9 tests added (commit 7b4cb82)
- unit-report-tests: 99% coverage (target 85%), 32 tests added (commit bc810be)
- unit-classifier-tests: 97% coverage (target 85%), 17 tests added (commit 690e48e)
- unit-e2e-golden: golden path test added (commit 6ea139e)

## Layer 3 landed: unit-enforcement (commit c82d985)

- Coverage gate: fail_under=80 in pyproject.toml
- Repository URL fixed: agent-observatory → agent-diagnostics

## PRD build complete

- **8/8 units landed**, 0 evictions, 1 pass
- **377 tests** (was 278), **91% coverage** (was 69%)
- Integration branch: `prd-build/testing-quality`
