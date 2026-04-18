"""Tests for the AnnotationResult discriminated union type.

Acceptance criteria:
- AnnotationResult is importable with a docstring
- Ok, NoCategoriesFound, and Error variants are constructable and frozen
- Internal helpers _unwrap_result / _to_annotation_result / _is_error behave correctly
- put_cached skips caching when is_error=True
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_diagnostics.types import (
    AnnotationError,
    AnnotationNoCategoriesFound,
    AnnotationOk,
    AnnotationResult,
)

# ---------------------------------------------------------------------------
# AnnotationResult type exists and has docstring
# ---------------------------------------------------------------------------


class TestAnnotationResultType:
    """Verify AnnotationResult is importable and documented."""

    def test_importable(self) -> None:
        assert AnnotationResult is not None

    def test_has_docstring(self) -> None:
        assert AnnotationResult.__doc__ is not None
        assert len(AnnotationResult.__doc__) > 0

    def test_docstring_mentions_variants(self) -> None:
        doc = AnnotationResult.__doc__ or ""
        assert "AnnotationOk" in doc
        assert "AnnotationNoCategoriesFound" in doc
        assert "AnnotationError" in doc


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------


class TestAnnotationOk:
    """Test AnnotationOk variant."""

    def test_construction(self) -> None:
        cats = ({"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"},)
        result = AnnotationOk(categories=cats)
        assert result.categories == cats

    def test_frozen(self) -> None:
        result = AnnotationOk(categories=())
        with pytest.raises(AttributeError):
            result.categories = ()  # type: ignore[misc]

    def test_empty_categories_tuple(self) -> None:
        result = AnnotationOk(categories=())
        assert result.categories == ()

    def test_multiple_categories(self) -> None:
        cats = (
            {"name": "retrieval_failure", "confidence": 0.9, "evidence": "e1"},
            {"name": "query_churn", "confidence": 0.7, "evidence": "e2"},
        )
        result = AnnotationOk(categories=cats)
        assert len(result.categories) == 2


class TestAnnotationNoCategoriesFound:
    """Test AnnotationNoCategoriesFound variant."""

    def test_construction(self) -> None:
        result = AnnotationNoCategoriesFound()
        assert isinstance(result, AnnotationNoCategoriesFound)

    def test_frozen(self) -> None:
        result = AnnotationNoCategoriesFound()
        with pytest.raises(AttributeError):
            result.x = 1  # type: ignore[attr-defined]


class TestAnnotationError:
    """Test AnnotationError variant."""

    def test_construction(self) -> None:
        result = AnnotationError(reason="API rate limit exceeded")
        assert result.reason == "API rate limit exceeded"

    def test_frozen(self) -> None:
        result = AnnotationError(reason="fail")
        with pytest.raises(AttributeError):
            result.reason = "other"  # type: ignore[misc]

    def test_reason_required(self) -> None:
        with pytest.raises(TypeError):
            AnnotationError()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Internal helpers from llm_annotator
# ---------------------------------------------------------------------------


class TestUnwrapResult:
    """Test _unwrap_result converts AnnotationResult to list[dict]."""

    def test_unwrap_ok(self) -> None:
        from agent_diagnostics.llm_annotator import _unwrap_result

        cats = ({"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"},)
        result = _unwrap_result(AnnotationOk(categories=cats))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "retrieval_failure"

    def test_unwrap_no_categories(self) -> None:
        from agent_diagnostics.llm_annotator import _unwrap_result

        result = _unwrap_result(AnnotationNoCategoriesFound())
        assert result == []

    def test_unwrap_error(self) -> None:
        from agent_diagnostics.llm_annotator import _unwrap_result

        result = _unwrap_result(AnnotationError(reason="timeout"))
        assert result == []


class TestToAnnotationResult:
    """Test _to_annotation_result wraps list[dict] into the right variant."""

    def test_non_empty_becomes_ok(self) -> None:
        from agent_diagnostics.llm_annotator import _to_annotation_result

        cats = [{"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"}]
        result = _to_annotation_result(cats)
        assert isinstance(result, AnnotationOk)
        assert len(result.categories) == 1

    def test_empty_becomes_no_categories(self) -> None:
        from agent_diagnostics.llm_annotator import _to_annotation_result

        result = _to_annotation_result([])
        assert isinstance(result, AnnotationNoCategoriesFound)


class TestIsError:
    """Test _is_error predicate."""

    def test_ok_is_not_error(self) -> None:
        from agent_diagnostics.llm_annotator import _is_error

        assert not _is_error(AnnotationOk(categories=()))

    def test_no_categories_is_not_error(self) -> None:
        from agent_diagnostics.llm_annotator import _is_error

        assert not _is_error(AnnotationNoCategoriesFound())

    def test_error_is_error(self) -> None:
        from agent_diagnostics.llm_annotator import _is_error

        assert _is_error(AnnotationError(reason="fail"))


# ---------------------------------------------------------------------------
# Cache guard: put_cached skips on is_error=True
# ---------------------------------------------------------------------------


class TestCacheErrorGuard:
    """Verify put_cached does not write when is_error=True."""

    def test_put_cached_skips_on_error(self, tmp_path: Path) -> None:
        from agent_diagnostics.annotation_cache import get_cached, put_cached

        key = "test_error_key"
        put_cached(tmp_path, key, [], is_error=True)
        assert get_cached(tmp_path, key) is None
        # Verify file was not created
        assert not (tmp_path / f"{key}.json").exists()

    def test_put_cached_writes_on_success(self, tmp_path: Path) -> None:
        from agent_diagnostics.annotation_cache import get_cached, put_cached

        key = "test_success_key"
        categories = [{"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"}]
        put_cached(tmp_path, key, categories, is_error=False)
        cached = get_cached(tmp_path, key)
        assert cached is not None
        assert len(cached) == 1

    def test_put_cached_default_is_not_error(self, tmp_path: Path) -> None:
        from agent_diagnostics.annotation_cache import get_cached, put_cached

        key = "test_default_key"
        categories = [{"name": "query_churn", "confidence": 0.7, "evidence": "e"}]
        put_cached(tmp_path, key, categories)  # no is_error kwarg
        cached = get_cached(tmp_path, key)
        assert cached is not None
