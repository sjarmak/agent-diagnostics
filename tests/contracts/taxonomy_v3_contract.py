"""Contract tests for taxonomy v3 expectations.

These tests define the structural contracts that the taxonomy system,
signal dependencies, and fixture schema must satisfy. They serve as
living documentation and regression guards for downstream consumers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Set

import pytest

from agent_diagnostics.constants import REDACTED_SIGNAL_FIELDS
from agent_diagnostics.taxonomy import valid_category_names

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Contract: signal_dependencies format — flat signal names, not dotted paths
# ---------------------------------------------------------------------------


class TestSignalDependenciesFormat:
    """Signal dependency lists must use flat signal names (e.g. 'reward'),
    never dotted paths (e.g. 'signals.reward' or 'result.reward').
    """

    _DOTTED_PATH_RE = re.compile(r"\.")

    def test_redacted_fields_are_flat_strings(self) -> None:
        """Every entry in REDACTED_SIGNAL_FIELDS must be a plain identifier."""
        for field_name in REDACTED_SIGNAL_FIELDS:
            assert isinstance(field_name, str), f"Expected str, got {type(field_name)}"
            assert not self._DOTTED_PATH_RE.search(field_name), (
                f"Signal dependency '{field_name}' uses dotted path; "
                "must be a flat signal name"
            )

    def test_redacted_fields_are_non_empty(self) -> None:
        """No empty strings in signal dependency sets."""
        for field_name in REDACTED_SIGNAL_FIELDS:
            assert field_name.strip(), "Empty signal name found"

    def test_redacted_fields_is_frozenset(self) -> None:
        """Signal dependency collections must be immutable frozensets."""
        assert isinstance(REDACTED_SIGNAL_FIELDS, frozenset)

    def test_required_redacted_fields_present(self) -> None:
        """The redacted set must contain at minimum: reward, passed, exception_info."""
        required = {"reward", "passed", "exception_info"}
        assert required.issubset(
            REDACTED_SIGNAL_FIELDS
        ), f"Missing required redacted fields: {required - REDACTED_SIGNAL_FIELDS}"

    def test_signal_dependencies_example_format(self) -> None:
        """Demonstrate the expected format: a list of flat signal name strings."""
        example_deps: list[str] = ["reward", "passed", "exception_info"]
        for dep in example_deps:
            assert isinstance(dep, str)
            assert "." not in dep


# ---------------------------------------------------------------------------
# Contract: valid_category_names() return type
# ---------------------------------------------------------------------------


class TestValidCategoryNamesContract:
    """valid_category_names() must return Set[str]."""

    def test_returns_set(self) -> None:
        result = valid_category_names()
        assert isinstance(result, set), f"Expected set, got {type(result)}"

    def test_returns_set_of_strings(self) -> None:
        result = valid_category_names()
        assert len(result) > 0, "Taxonomy should have at least one category"
        for name in result:
            assert isinstance(
                name, str
            ), f"Category name should be str, got {type(name)}"

    def test_return_type_is_set_str(self) -> None:
        """Verify the return type annotation contract: Set[str]."""
        import inspect

        sig = inspect.signature(valid_category_names)
        ret = sig.return_annotation
        # Accept both Set[str] and set[str]
        assert ret in (
            Set[str],
            set[str],
        ), f"valid_category_names return annotation should be Set[str], got {ret}"


# ---------------------------------------------------------------------------
# Contract: fixture directory schema
# ---------------------------------------------------------------------------


class TestFixtureSchema:
    """Fixture directories used for golden tests must contain
    expected.json and trajectory.json at minimum.
    """

    def _golden_fixture_dirs(self) -> list[Path]:
        """Find all golden_trial fixture directories."""
        golden = FIXTURES_DIR / "golden_trial"
        if golden.is_dir():
            return [golden]
        return []

    def test_golden_fixtures_exist(self) -> None:
        """At least one golden fixture directory must exist."""
        dirs = self._golden_fixture_dirs()
        assert len(dirs) > 0, f"No golden_trial fixture found under {FIXTURES_DIR}"

    def test_golden_fixture_has_trajectory(self) -> None:
        """Each golden fixture must contain trajectory.json."""
        for d in self._golden_fixture_dirs():
            traj = d / "trajectory.json"
            assert traj.exists(), f"Missing trajectory.json in {d}"
            # Must be valid JSON
            data = json.loads(traj.read_text())
            assert isinstance(data, dict), "trajectory.json must be a JSON object"

    def test_golden_fixture_trajectory_has_steps(self) -> None:
        """trajectory.json must have a 'steps' key with a list value."""
        for d in self._golden_fixture_dirs():
            traj = d / "trajectory.json"
            data = json.loads(traj.read_text())
            assert "steps" in data, "trajectory.json must have 'steps' key"
            assert isinstance(data["steps"], list), "'steps' must be a list"
