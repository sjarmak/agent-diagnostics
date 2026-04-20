"""Tests for package version resolution via importlib.metadata."""

from __future__ import annotations

import importlib
import importlib.metadata
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import NoReturn

import pytest

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


class TestVersionFallback:
    """When package metadata is missing, `__version__` falls back to a sentinel."""

    def test_version_fallback_on_package_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the package not being installed (e.g. a source checkout
        # without `pip install -e .` having registered metadata). The fallback
        # branch in `agent_diagnostics.__init__` must set __version__ to the
        # '0.0.0+unknown' sentinel instead of raising.
        def _raise_not_found(name: str) -> NoReturn:
            raise PackageNotFoundError(name)

        # Patch the attribute on the module object, not the already-bound alias
        # in agent_diagnostics.__init__ (`from importlib.metadata import version
        # as _pkg_version`). The patch only takes effect because reload()
        # re-executes the from-import, picking up the patched attribute.
        with monkeypatch.context() as mp:
            mp.setattr(importlib.metadata, "version", _raise_not_found)
            reloaded = importlib.reload(agent_diagnostics)
            assert reloaded.__version__ == "0.0.0+unknown"

        # Outside the context, the patch is undone. Reload once more so the
        # module's __version__ returns to the real installed value for any
        # subsequent test that imports it.
        importlib.reload(agent_diagnostics)
