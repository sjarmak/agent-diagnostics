# Research: unit-fake-backend-fixtures

## Taxonomy v2 Categories (14 from acceptance criteria)

From `src/agent_diagnostics/taxonomy_v2.yaml`, the spec lists these 14:

1. retrieval_failure
2. query_churn
3. wrong_tool_choice
4. over_exploration
5. edit_verify_loop_failure
6. incomplete_solution
7. near_miss
8. minimal_progress
9. rate_limited_run
10. exception_crash
11. stale_context
12. decomposition_failure
13. premature_commit
14. clean_success

Note: taxonomy*v2.yaml actually has MORE categories (missing_code_navigation, multi_repo_scope_failure, local_remote_mismatch, verifier_mismatch, success_via*\*, insufficient_provenance, task_ambiguity). The spec says 14 minimum with those specific names. "premature_commit" and "clean_success" are NOT in taxonomy_v2.yaml but ARE in the acceptance criteria list — they must be fixture directories regardless.

## Existing Patterns

- `tests/conftest.py` has FIXTURES_DIR = Path(**file**).parent / "fixtures"
- `tests/fixtures/golden_trial/` has trajectory.json and result.json (not expected.json)
- No `tests/__init__.py` exists — need to create
- No `tests/integration/` directory — need to create
- `llm_annotator.py` returns `list[dict]` with keys: name, confidence, evidence
- `_ANNOTATION_SCHEMA` defines the JSON structure: {"categories": [{"name","confidence","evidence"}]}

## FakeLLMBackend Design

- Must accept prompts, return deterministic JSON
- Must have `call_count` attribute
- Key by category name found in prompt for fixture-based testing
- Return valid annotation JSON matching `_ANNOTATION_SCHEMA`
