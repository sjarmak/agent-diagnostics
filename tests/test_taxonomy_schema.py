"""Tests validating the taxonomy v3 schema structure.

Every category in taxonomy_v3.yaml must have:
- severity: one of "blocker", "major", "minor"
- derived_from_signal: bool
- signal_dependencies: list of strings

Also validates dimension structure and category completeness.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_diagnostics.taxonomy import (
    _extract_categories,
    _is_v2,
    _is_v3,
    load_taxonomy,
    valid_category_names,
)

V3_YAML_PATH = (
    Path(__file__).parent.parent / "src" / "agent_diagnostics" / "taxonomy_v3.yaml"
)

VALID_SEVERITIES = {"blocker", "major", "minor"}

# Categories that existed in v2 and must still be present in v3
V2_CATEGORIES = {
    "retrieval_failure",
    "query_churn",
    "wrong_tool_choice",
    "missing_code_navigation",
    "decomposition_failure",
    "stale_context",
    "multi_repo_scope_failure",
    "edit_verify_loop_failure",
    "incomplete_solution",
    "near_miss",
    "minimal_progress",
    "local_remote_mismatch",
    "verifier_mismatch",
    "exception_crash",
    "rate_limited_run",
    "success_via_code_nav",
    "success_via_semantic_search",
    "success_via_local_exec",
    "success_via_commit_context",
    "success_via_decomposition",
    "insufficient_provenance",
    "task_ambiguity",
    "over_exploration",
}

# New dimensions expected in v3
NEW_V3_DIMENSIONS = {"ToolUse", "Faithfulness", "Metacognition", "Integrity", "Safety"}

# Categories that are pure functions of the reward scalar
DERIVED_FROM_SIGNAL_CATEGORIES = {
    "incomplete_solution",
    "near_miss",
    "minimal_progress",
}


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Reset taxonomy cache before and after each test."""
    from agent_diagnostics import taxonomy as _mod

    _mod._cached_taxonomy = None
    _mod._cached_path = None
    yield  # type: ignore[misc]
    _mod._cached_taxonomy = None
    _mod._cached_path = None


@pytest.fixture()
def v3_taxonomy() -> dict:
    """Load and return the v3 taxonomy dict."""
    return load_taxonomy(V3_YAML_PATH)


@pytest.fixture()
def v3_categories(v3_taxonomy: dict) -> list[dict]:
    """Return all categories from the v3 taxonomy."""
    return _extract_categories(v3_taxonomy)


# ---------------------------------------------------------------------------
# Version and format detection
# ---------------------------------------------------------------------------


class TestV3Format:
    """Verify v3 taxonomy is correctly identified."""

    def test_version_is_3(self, v3_taxonomy: dict) -> None:
        assert v3_taxonomy["version"] == "3.0.0"

    def test_is_v2_true(self, v3_taxonomy: dict) -> None:
        """v3 uses dimensions format, so _is_v2 returns True."""
        assert _is_v2(v3_taxonomy) is True

    def test_is_v3_true(self, v3_taxonomy: dict) -> None:
        assert _is_v3(v3_taxonomy) is True

    def test_has_dimensions(self, v3_taxonomy: dict) -> None:
        assert "dimensions" in v3_taxonomy
        assert isinstance(v3_taxonomy["dimensions"], list)
        assert len(v3_taxonomy["dimensions"]) > 0


# ---------------------------------------------------------------------------
# Dimension completeness
# ---------------------------------------------------------------------------


class TestV3Dimensions:
    """Verify all expected dimensions are present."""

    def test_new_dimensions_present(self, v3_taxonomy: dict) -> None:
        dim_names = {d["name"] for d in v3_taxonomy["dimensions"]}
        for dim in NEW_V3_DIMENSIONS:
            assert dim in dim_names, f"Missing new dimension: {dim}"

    def test_tooluse_dimension_exists(self, v3_taxonomy: dict) -> None:
        dim_names = {d["name"] for d in v3_taxonomy["dimensions"]}
        assert "ToolUse" in dim_names

    def test_integrity_dimension_exists(self, v3_taxonomy: dict) -> None:
        dim_names = {d["name"] for d in v3_taxonomy["dimensions"]}
        assert "Integrity" in dim_names


# ---------------------------------------------------------------------------
# Category completeness — all v2 categories preserved
# ---------------------------------------------------------------------------


class TestV2CategoriesPreserved:
    """All v2 categories must still exist in v3."""

    def test_all_v2_categories_present(self, v3_categories: list[dict]) -> None:
        v3_names = {cat["name"] for cat in v3_categories}
        missing = V2_CATEGORIES - v3_names
        assert missing == set(), f"v2 categories missing from v3: {missing}"


# ---------------------------------------------------------------------------
# Per-category schema validation
# ---------------------------------------------------------------------------


class TestCategorySeverity:
    """Every category must have a valid severity field."""

    def test_all_categories_have_severity(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert (
                "severity" in cat
            ), f"Category '{cat['name']}' missing 'severity' field"

    def test_severity_values_valid(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert (
                cat["severity"] in VALID_SEVERITIES
            ), f"Category '{cat['name']}' has invalid severity: {cat['severity']}"


class TestCategoryDerivedFromSignal:
    """Every category must have a derived_from_signal boolean field."""

    def test_all_categories_have_derived_from_signal(
        self, v3_categories: list[dict]
    ) -> None:
        for cat in v3_categories:
            assert (
                "derived_from_signal" in cat
            ), f"Category '{cat['name']}' missing 'derived_from_signal' field"

    def test_derived_from_signal_is_bool(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert isinstance(cat["derived_from_signal"], bool), (
                f"Category '{cat['name']}' derived_from_signal is not bool: "
                f"{type(cat['derived_from_signal'])}"
            )

    def test_expected_derived_categories(self, v3_categories: list[dict]) -> None:
        """incomplete_solution, near_miss, minimal_progress must be derived."""
        for cat in v3_categories:
            if cat["name"] in DERIVED_FROM_SIGNAL_CATEGORIES:
                assert (
                    cat["derived_from_signal"] is True
                ), f"Category '{cat['name']}' should be derived_from_signal=true"

    def test_non_derived_categories(self, v3_categories: list[dict]) -> None:
        """Categories NOT in the derived set should be false."""
        for cat in v3_categories:
            if cat["name"] not in DERIVED_FROM_SIGNAL_CATEGORIES:
                assert (
                    cat["derived_from_signal"] is False
                ), f"Category '{cat['name']}' should be derived_from_signal=false"


class TestCategorySignalDependencies:
    """Every category must have a signal_dependencies list."""

    def test_all_categories_have_signal_dependencies(
        self, v3_categories: list[dict]
    ) -> None:
        for cat in v3_categories:
            assert (
                "signal_dependencies" in cat
            ), f"Category '{cat['name']}' missing 'signal_dependencies' field"

    def test_signal_dependencies_is_list(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert isinstance(cat["signal_dependencies"], list), (
                f"Category '{cat['name']}' signal_dependencies is not list: "
                f"{type(cat['signal_dependencies'])}"
            )

    def test_signal_dependencies_contain_strings(
        self, v3_categories: list[dict]
    ) -> None:
        for cat in v3_categories:
            for dep in cat["signal_dependencies"]:
                assert isinstance(
                    dep, str
                ), f"Category '{cat['name']}' has non-string signal dependency: {dep}"

    def test_derived_categories_have_reward_dependency(
        self, v3_categories: list[dict]
    ) -> None:
        """Derived-from-signal categories should list 'reward' as a dependency."""
        for cat in v3_categories:
            if cat["name"] in DERIVED_FROM_SIGNAL_CATEGORIES:
                assert "reward" in cat["signal_dependencies"], (
                    f"Category '{cat['name']}' is derived but missing "
                    f"'reward' in signal_dependencies"
                )

    def test_non_derived_categories_valid_dependencies(
        self, v3_categories: list[dict]
    ) -> None:
        """Non-derived categories may have signal_dependencies as hints but must be lists."""
        for cat in v3_categories:
            if cat["name"] not in DERIVED_FROM_SIGNAL_CATEGORIES:
                assert isinstance(cat["signal_dependencies"], list), (
                    f"Category '{cat['name']}' signal_dependencies must be a list, "
                    f"got {type(cat['signal_dependencies'])}"
                )


# ---------------------------------------------------------------------------
# Basic structural validation
# ---------------------------------------------------------------------------


class TestCategoryStructure:
    """Basic structural checks for all categories."""

    def test_all_have_name(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert "name" in cat
            assert isinstance(cat["name"], str)
            assert len(cat["name"]) > 0

    def test_all_have_dimension(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert "dimension" in cat
            assert isinstance(cat["dimension"], str)

    def test_all_have_description(self, v3_categories: list[dict]) -> None:
        for cat in v3_categories:
            assert "description" in cat
            assert isinstance(cat["description"], str)
            assert len(cat["description"]) > 10

    def test_all_have_polarity(self, v3_categories: list[dict]) -> None:
        valid_polarities = {"failure", "success", "neutral"}
        for cat in v3_categories:
            assert "polarity" in cat
            assert cat["polarity"] in valid_polarities

    def test_no_duplicate_names(self, v3_categories: list[dict]) -> None:
        names = [cat["name"] for cat in v3_categories]
        assert len(names) == len(set(names)), (
            f"Duplicate category names: " f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_wrong_tool_choice_in_tooluse(self, v3_taxonomy: dict) -> None:
        """wrong_tool_choice should be in ToolUse dimension (moved from Retrieval)."""
        for dim in v3_taxonomy["dimensions"]:
            if dim["name"] == "ToolUse":
                cat_names = [c["name"] for c in dim["categories"]]
                assert "wrong_tool_choice" in cat_names
                return
        pytest.fail("ToolUse dimension not found")


# ---------------------------------------------------------------------------
# Load via load_taxonomy
# ---------------------------------------------------------------------------


class TestLoadTaxonomyV3:
    """Verify load_taxonomy can load v3 and extract categories."""

    def test_load_returns_dict(self) -> None:
        t = load_taxonomy(V3_YAML_PATH)
        assert isinstance(t, dict)

    def test_extract_categories_returns_list(self) -> None:
        t = load_taxonomy(V3_YAML_PATH)
        cats = _extract_categories(t)
        assert isinstance(cats, list)
        assert len(cats) > 20

    def test_valid_category_names_returns_set(self) -> None:
        names = valid_category_names(V3_YAML_PATH)
        assert isinstance(names, set)
        assert "retrieval_failure" in names
        assert "hallucinated_api" in names
