"""Integration tests verifying FakeLLMBackend determinism.

Two consecutive runs with the same inputs must produce byte-identical outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fake_llm_backend import FakeLLMBackend

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
TRIALS_DIR = FIXTURES_DIR / "trials"

# All 14 fixture categories expected by the acceptance criteria
EXPECTED_CATEGORIES = sorted(
    [
        "retrieval_failure",
        "query_churn",
        "wrong_tool_choice",
        "over_exploration",
        "edit_verify_loop_failure",
        "incomplete_solution",
        "near_miss",
        "minimal_progress",
        "rate_limited_run",
        "exception_crash",
        "stale_context",
        "decomposition_failure",
        "premature_commit",
        "clean_success",
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_trial_fixture(category: str) -> tuple[dict, dict]:
    """Load trajectory.json and expected.json for a given category."""
    trial_dir = TRIALS_DIR / category
    with open(trial_dir / "trajectory.json") as f:
        trajectory = json.load(f)
    with open(trial_dir / "expected.json") as f:
        expected = json.load(f)
    return trajectory, expected


def _build_prompt_for_category(category: str, trajectory: dict) -> str:
    """Build a synthetic prompt string containing the category name and trajectory."""
    traj_json = json.dumps(trajectory, indent=1)
    return (
        f"Annotate this trial for category: {category}\n"
        f"Trajectory:\n{traj_json}\n"
        "Return annotation JSON."
    )


# ---------------------------------------------------------------------------
# Fixture directory structure tests
# ---------------------------------------------------------------------------


class TestFixtureStructure:
    """Verify per-category fixture directories exist with required files."""

    def test_at_least_14_trial_directories(self) -> None:
        trial_dirs = [d for d in TRIALS_DIR.iterdir() if d.is_dir()]
        assert len(trial_dirs) >= 14, f"Expected >= 14 trial directories, found {len(trial_dirs)}"

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_trial_directory_exists(self, category: str) -> None:
        trial_dir = TRIALS_DIR / category
        assert trial_dir.is_dir(), f"Missing trial directory: {trial_dir}"

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_expected_json_exists(self, category: str) -> None:
        path = TRIALS_DIR / category / "expected.json"
        assert path.is_file(), f"Missing expected.json: {path}"

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_trajectory_json_exists(self, category: str) -> None:
        path = TRIALS_DIR / category / "trajectory.json"
        assert path.is_file(), f"Missing trajectory.json: {path}"

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_expected_json_valid(self, category: str) -> None:
        path = TRIALS_DIR / category / "expected.json"
        with open(path) as f:
            data = json.load(f)
        assert "categories" in data, f"expected.json missing 'categories' key: {path}"
        assert isinstance(data["categories"], list)
        for cat in data["categories"]:
            assert "name" in cat
            assert "confidence" in cat
            assert "evidence" in cat

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_trajectory_json_valid(self, category: str) -> None:
        path = TRIALS_DIR / category / "trajectory.json"
        with open(path) as f:
            data = json.load(f)
        assert "steps" in data, f"trajectory.json missing 'steps' key: {path}"
        assert isinstance(data["steps"], list)


# ---------------------------------------------------------------------------
# FakeLLMBackend determinism tests
# ---------------------------------------------------------------------------


class TestFakeLLMBackendDeterminism:
    """Verify that identical inputs produce identical outputs across runs."""

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_single_category_deterministic(self, category: str) -> None:
        """Two calls with the same prompt return byte-identical JSON."""
        trajectory, _expected = _load_trial_fixture(category)
        prompt = _build_prompt_for_category(category, trajectory)

        backend_a = FakeLLMBackend()
        backend_b = FakeLLMBackend()

        result_a = backend_a.annotate_json(prompt)
        result_b = backend_b.annotate_json(prompt)

        assert result_a == result_b, (
            f"Non-deterministic output for {category}:\n  run 1: {result_a}\n  run 2: {result_b}"
        )

    def test_full_sweep_deterministic(self) -> None:
        """Run all 14 categories twice and verify identical artifact sets."""
        backend_1 = FakeLLMBackend()
        backend_2 = FakeLLMBackend()

        artifacts_1: dict[str, str] = {}
        artifacts_2: dict[str, str] = {}

        for category in EXPECTED_CATEGORIES:
            trajectory, _expected = _load_trial_fixture(category)
            prompt = _build_prompt_for_category(category, trajectory)
            artifacts_1[category] = backend_1.annotate_json(prompt)
            artifacts_2[category] = backend_2.annotate_json(prompt)

        assert artifacts_1 == artifacts_2, "Full sweep produced different artifacts"

    def test_call_count_tracks_invocations(self) -> None:
        """call_count increments exactly once per annotate call."""
        backend = FakeLLMBackend()
        assert backend.call_count == 0

        for i, category in enumerate(EXPECTED_CATEGORIES, start=1):
            trajectory, _ = _load_trial_fixture(category)
            prompt = _build_prompt_for_category(category, trajectory)
            backend.annotate(prompt)
            assert backend.call_count == i

    def test_call_log_hashes_are_deterministic(self) -> None:
        """The same prompt sequence produces identical call_log hashes."""
        backend_a = FakeLLMBackend()
        backend_b = FakeLLMBackend()

        for category in EXPECTED_CATEGORIES:
            trajectory, _ = _load_trial_fixture(category)
            prompt = _build_prompt_for_category(category, trajectory)
            backend_a.annotate(prompt)
            backend_b.annotate(prompt)

        assert backend_a.call_log == backend_b.call_log

    def test_response_matches_expected_category(self) -> None:
        """FakeLLMBackend returns the correct category for each fixture."""
        backend = FakeLLMBackend()

        for category in EXPECTED_CATEGORIES:
            trajectory, expected = _load_trial_fixture(category)
            prompt = _build_prompt_for_category(category, trajectory)
            result = backend.annotate(prompt)

            result_names = {c["name"] for c in result["categories"]}
            expected_names = {c["name"] for c in expected["categories"]}
            assert expected_names.issubset(result_names), (
                f"Category {category}: expected {expected_names} in {result_names}"
            )

    def test_output_schema_valid(self) -> None:
        """Every response conforms to the annotation JSON schema."""
        backend = FakeLLMBackend()

        for category in EXPECTED_CATEGORIES:
            trajectory, _ = _load_trial_fixture(category)
            prompt = _build_prompt_for_category(category, trajectory)
            result = backend.annotate(prompt)

            assert "categories" in result
            assert isinstance(result["categories"], list)
            for cat in result["categories"]:
                assert isinstance(cat["name"], str)
                assert isinstance(cat["confidence"], (int, float))
                assert 0 <= cat["confidence"] <= 1
                assert isinstance(cat["evidence"], str)
