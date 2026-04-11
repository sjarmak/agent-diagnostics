# Research: unit-annotation-cache

## Existing Codebase Patterns

- `llm_annotator.py`: Two backends (claude-code CLI, API SDK). Both build prompt via `build_prompt()`, call LLM, parse JSON, validate via `validate_categories()`.
- `_MODEL_ALIASES` and `_API_MODEL_MAP` are module-level dicts with matching keys: haiku, sonnet, opus.
- `annotation_schema.json` is valid JSON, 153 lines. No changes needed.
- `FakeLLMBackend` in `tests/fake_llm_backend.py` has `call_count`, `call_log`, and `annotate(prompt)` returning `{"categories": [...]}`.
- Tests use pytest, `Path(__file__).parent / "fixtures"` pattern, `from __future__ import annotations`.

## Key Integration Points

- Cache insertion point: after `build_prompt()` and model resolution, before LLM call.
- API backend (lines 539-551) does raw text parsing with markdown fence stripping — replace with tool-use structured output.
- `_resolve_model_api(model)` returns model_id — needed for cache key.

## Files to Create/Modify

1. **NEW** `src/agent_diagnostics/annotation_cache.py` — cache_key, get_cached, put_cached
2. **MODIFY** `src/agent_diagnostics/llm_annotator.py` — integrate cache, add tool-use to API backend
3. **NEW** `tests/test_annotation_cache.py` — cache determinism, zero-call on hit, directory creation
4. **NEW** `tests/test_backend_parity.py` — same prompt through FakeLLMBackend produces identical output
5. **NEW** `tests/test_backend_model_parity.py` — \_API_MODEL_MAP and \_MODEL_ALIASES key alignment
