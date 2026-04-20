"""Tests verifying validate_categories works with v1 and v2 taxonomy formats.

Ensures the MH-1 bug fix (replacing direct ['categories'] access with
valid_category_names()) works correctly across all taxonomy versions.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from agent_diagnostics.llm_annotator import validate_categories
from agent_diagnostics.taxonomy import (
    valid_category_names,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

V3_YAML_PATH = Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v3.yaml"


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Reset taxonomy cache before and after each test."""
    from agent_diagnostics import taxonomy as _mod

    _mod._cached_taxonomy = None
    _mod._cached_path = None
    yield  # type: ignore[misc]
    _mod._cached_taxonomy = None
    _mod._cached_path = None


# ---------------------------------------------------------------------------
# v1 compatibility
# ---------------------------------------------------------------------------


class TestValidateCategoriesV1:
    """validate_categories works when the default taxonomy is v1 format."""

    def test_valid_v1_category_accepted(self) -> None:
        """A known v1 category name is accepted by validate_categories."""
        # Use the default packaged taxonomy (v1)
        cats = [{"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"}]
        result = validate_categories(cats, "/tmp/trial")
        assert len(result) == 1
        assert result[0]["name"] == "retrieval_failure"

    def test_invalid_category_rejected_v1(self) -> None:
        cats = [{"name": "totally_fake_category", "confidence": 0.5, "evidence": "e"}]
        result = validate_categories(cats, "/tmp/trial")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# v2 compatibility
# ---------------------------------------------------------------------------


class TestValidateCategoriesV2:
    """validate_categories works when taxonomy is loaded from v2 format."""

    def test_valid_v2_category_accepted(self) -> None:
        """A category from v2 taxonomy is accepted via valid_category_names."""
        v2_path = Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v2.yaml"
        # Patch valid_category_names to use v2 taxonomy
        v2_names = valid_category_names(v2_path)
        with mock.patch(
            "agent_diagnostics.llm_annotator.valid_category_names",
            return_value=v2_names,
        ):
            cats = [{"name": "decomposition_failure", "confidence": 0.8, "evidence": "e"}]
            result = validate_categories(cats, "/tmp/trial")
            assert len(result) == 1
            assert result[0]["name"] == "decomposition_failure"

    def test_v2_dimension_category_accepted(self) -> None:
        """Categories nested under dimensions in v2 are accessible."""
        v2_path = Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v2.yaml"
        v2_names = valid_category_names(v2_path)
        # stale_context is under Reasoning dimension in v2
        assert "stale_context" in v2_names

    def test_invalid_category_rejected_v2(self) -> None:
        """An unknown category is rejected even with v2 taxonomy."""
        v2_path = Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v2.yaml"
        v2_names = valid_category_names(v2_path)
        with mock.patch(
            "agent_diagnostics.llm_annotator.valid_category_names",
            return_value=v2_names,
        ):
            cats = [{"name": "nonexistent_xyz", "confidence": 0.5, "evidence": "e"}]
            result = validate_categories(cats, "/tmp/trial")
            assert len(result) == 0


# ---------------------------------------------------------------------------
# v3 compatibility
# ---------------------------------------------------------------------------


class TestValidateCategoriesV3:
    """validate_categories works with v3 taxonomy (same dimensions structure)."""

    def test_v3_category_accepted(self) -> None:
        """A v3-only category (from new dimensions) is accepted."""
        v3_names = valid_category_names(V3_YAML_PATH)
        with mock.patch(
            "agent_diagnostics.llm_annotator.valid_category_names",
            return_value=v3_names,
        ):
            cats = [{"name": "hallucinated_api", "confidence": 0.9, "evidence": "e"}]
            result = validate_categories(cats, "/tmp/trial")
            assert len(result) == 1
            assert result[0]["name"] == "hallucinated_api"

    def test_v3_existing_category_still_works(self) -> None:
        """Existing v2 categories are still valid in v3."""
        v3_names = valid_category_names(V3_YAML_PATH)
        with mock.patch(
            "agent_diagnostics.llm_annotator.valid_category_names",
            return_value=v3_names,
        ):
            cats = [{"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"}]
            result = validate_categories(cats, "/tmp/trial")
            assert len(result) == 1


# ---------------------------------------------------------------------------
# Bug regression: no direct ["categories"] access
# ---------------------------------------------------------------------------


class TestNoCategoriesKeyAccess:
    """Regression test: validate_categories must not access ['categories'] directly."""

    def test_no_direct_categories_key_in_source(self) -> None:
        """The bug was load_taxonomy()['categories'] — verify it's gone."""
        import inspect

        from agent_diagnostics import llm_annotator

        source = inspect.getsource(llm_annotator.validate_categories)
        assert '["categories"]' not in source
        assert "['categories']" not in source

    def test_uses_valid_category_names(self) -> None:
        """validate_categories must call valid_category_names, not load_taxonomy."""
        import inspect

        from agent_diagnostics import llm_annotator

        source = inspect.getsource(llm_annotator.validate_categories)
        assert "valid_category_names" in source


# ---------------------------------------------------------------------------
# Cross-version: valid_category_names extracts from all formats
# ---------------------------------------------------------------------------


class TestValidCategoryNamesAllFormats:
    """valid_category_names() works correctly with v1, v2, and v3."""

    def test_v1_names(self) -> None:
        v1_path = Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v1.yaml"
        names = valid_category_names(v1_path)
        assert "retrieval_failure" in names
        assert "query_churn" in names
        assert len(names) > 5

    def test_v2_names(self) -> None:
        v2_path = Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v2.yaml"
        names = valid_category_names(v2_path)
        assert "retrieval_failure" in names
        assert "decomposition_failure" in names
        assert len(names) > 10

    def test_v3_names(self) -> None:
        names = valid_category_names(V3_YAML_PATH)
        assert "retrieval_failure" in names
        assert "hallucinated_api" in names
        assert "reward_hacking" in names
        assert len(names) > 20
