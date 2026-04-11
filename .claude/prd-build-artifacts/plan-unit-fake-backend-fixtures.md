# Plan: unit-fake-backend-fixtures

## Steps

1. Create `tests/__init__.py` (empty, for package imports)
2. Create `tests/integration/__init__.py` (empty)
3. Create `tests/fake_llm_backend.py` — FakeLLMBackend class
4. Create 14 fixture directories under `tests/fixtures/trials/` with expected.json + trajectory.json
5. Create `tests/integration/test_determinism.py` — determinism tests
6. Run acceptance criteria checks
