"""Tests for package version resolution via importlib.metadata."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

import agent_diagnostics


class TestVersionResolution:
    """`agent_diagnostics.__version__` is resolved from installed metadata."""

    def test_version_matches_importlib_metadata(self) -> None:
        # In any environment where the package is installed (including editable
        # installs from pyproject.toml), __version__ must equal the value
        # reported by importlib.metadata — this is the whole point of the
        # refactor: a single source of truth (pyproject.toml) for releases.
        assert agent_diagnostics.__version__ == _pkg_version("agent-diagnostics")

    def test_version_is_non_empty_string(self) -> None:
        version = agent_diagnostics.__version__
        assert isinstance(version, str)
        assert version  # non-empty
