"""Shared test fixtures for agent-diagnostics tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_diagnostics import taxonomy as _taxonomy_mod

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Taxonomy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_taxonomy_cache() -> None:
    """Reset the taxonomy module cache so each test starts fresh."""
    _taxonomy_mod._cached_taxonomy = None
    _taxonomy_mod._cached_path = None
    yield  # type: ignore[misc]
    _taxonomy_mod._cached_taxonomy = None
    _taxonomy_mod._cached_path = None


@pytest.fixture()
def taxonomy_data() -> dict:
    """Return a loaded taxonomy dict (uses default packaged YAML)."""
    # Reset cache first to ensure clean state
    _taxonomy_mod._cached_taxonomy = None
    _taxonomy_mod._cached_path = None
    data = _taxonomy_mod.load_taxonomy()
    yield data  # type: ignore[misc]
    _taxonomy_mod._cached_taxonomy = None
    _taxonomy_mod._cached_path = None


@pytest.fixture()
def taxonomy_category_names(taxonomy_data: dict) -> set[str]:
    """Return the set of valid category name strings."""
    return {cat["name"] for cat in _taxonomy_mod._extract_categories(taxonomy_data)}
