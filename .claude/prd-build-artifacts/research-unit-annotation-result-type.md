# Research: unit-annotation-result-type

## Files in scope

- `src/agent_diagnostics/types.py` — frozen dataclasses, TypedDict, Protocol. Uses `from __future__ import annotations`. Imports: dataclass, field, Optional, Protocol, Sequence, TypedDict, runtime_checkable. Union not yet imported.
- `src/agent_diagnostics/llm_annotator.py` — returns `list[dict]` from public functions. Error paths return `[]`. Success paths call `put_cached()` then return validated list. Uses `from __future__ import annotations`.
- `src/agent_diagnostics/annotation_cache.py` — `put_cached()` accepts `list[dict[str, Any]]`. No error guard currently.
- `pyproject.toml` — dev deps: pytest, pytest-cov, ruff. No mypy.
- `tests/test_llm_annotator.py` — comprehensive tests for all backends, helpers, dispatch.

## Key patterns

- All dataclasses in types.py are `frozen=True`
- `from __future__ import annotations` used everywhere
- Public annotator functions return `list[dict]` at boundary
- Error cases return `[]`, success cases call `put_cached` then return validated list
- Cache is only written on success paths already in `annotate_trial_claude_code` and `annotate_trial_api`

## Integration points

- `put_cached` is called only after successful validation in both backends
- The `annotate_trial_claude_code` function has multiple error exit points (rc!=0, parse failure, timeout, exception) all returning `[]`
- The `annotate_trial_api` function similarly returns `[]` on errors

## Observations

- Cache is already only written on success paths — the guard in annotation_cache.py is for defense-in-depth
- Adding `is_error` param to `put_cached` is cleanest approach per spec
- AnnotationResult union needs a docstring — can use module-level assignment + `__doc__`
