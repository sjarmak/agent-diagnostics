"""Tests for model-ID alignment between API and CLI backends.

Verifies that _API_MODEL_MAP and _MODEL_ALIASES have matching keys,
ensuring every alias resolves consistently in both backends.
"""

from __future__ import annotations


from agent_diagnostics.llm_annotator import _API_MODEL_MAP, _MODEL_ALIASES


class TestModelAlignmentGuard:
    """Verify API model map and CLI alias map are aligned."""

    def test_api_map_keys_match_alias_keys(self) -> None:
        """Every key in _API_MODEL_MAP must exist in _MODEL_ALIASES."""
        assert set(_API_MODEL_MAP.keys()) == set(_MODEL_ALIASES.keys())

    def test_alias_keys_match_api_map_keys(self) -> None:
        """Every key in _MODEL_ALIASES must exist in _API_MODEL_MAP."""
        for alias_key in _MODEL_ALIASES:
            assert alias_key in _API_MODEL_MAP, (
                f"Alias '{alias_key}' present in _MODEL_ALIASES "
                f"but missing from _API_MODEL_MAP"
            )

    def test_api_map_keys_in_aliases(self) -> None:
        """Every key in _API_MODEL_MAP must exist in _MODEL_ALIASES."""
        for api_key in _API_MODEL_MAP:
            assert api_key in _MODEL_ALIASES, (
                f"Key '{api_key}' present in _API_MODEL_MAP "
                f"but missing from _MODEL_ALIASES"
            )

    def test_maps_are_nonempty(self) -> None:
        """Both maps must contain at least one entry."""
        assert len(_API_MODEL_MAP) > 0
        assert len(_MODEL_ALIASES) > 0

    def test_api_model_values_are_strings(self) -> None:
        """All _API_MODEL_MAP values should be non-empty strings."""
        for key, value in _API_MODEL_MAP.items():
            assert (
                isinstance(value, str) and len(value) > 0
            ), f"_API_MODEL_MAP['{key}'] must be a non-empty string"
