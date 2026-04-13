"""Tests for agent_diagnostics.signals module."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from agent_diagnostics.signals import (
    _is_excluded_path,
    _is_valid_trial,
    _resolve_benchmark_from_directory,
    extract_all,
    extract_signals,
    load_manifest,
)
from agent_diagnostics.tool_registry import (
    DEFAULT_REGISTRY,
    OPENHANDS_REGISTRY,
    ToolRegistry,
)
from agent_diagnostics.types import TrialSignals

# ---------------------------------------------------------------------------
# Helpers — synthetic trial directory builders
# ---------------------------------------------------------------------------


def _write_result(trial_dir: Path, data: dict) -> None:
    """Write a result.json into the trial directory."""
    (trial_dir / "result.json").write_text(json.dumps(data))


def _write_trajectory(trial_dir: Path, data: dict) -> None:
    """Write a trajectory.json into the trial directory."""
    (trial_dir / "trajectory.json").write_text(json.dumps(data))


def _make_tool_call(fn: str, args: dict | None = None) -> dict:
    return {"function_name": fn, "arguments": args or {}}


def _make_step(tool_calls: list[dict], observation: dict | None = None) -> dict:
    return {
        "tool_calls": tool_calls,
        "observation": observation or {},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_trial(tmp_path: Path) -> Path:
    """Trial dir with result.json (reward=1.0) and trajectory with tool calls."""
    trial = tmp_path / "trial_01"
    trial.mkdir()
    _write_result(
        trial,
        {
            "task_name": "fix_bug_123",
            "verifier_result": {"rewards": {"reward": 1.0}},
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:05:00Z",
            "agent_info": {"model_info": {"name": "anthropic/claude-sonnet-4-6"}},
        },
    )
    _write_trajectory(
        trial,
        {
            "steps": [
                _make_step(
                    [
                        _make_tool_call("Grep", {"pattern": "bug"}),
                        _make_tool_call("Read", {"file_path": "/src/main.py"}),
                    ]
                ),
                _make_step(
                    [
                        _make_tool_call(
                            "Edit",
                            {
                                "file_path": "/src/main.py",
                                "new_string": "fixed\nline\n",
                            },
                        ),
                    ]
                ),
                _make_step(
                    [
                        _make_tool_call("Bash", {"command": "pytest"}),
                    ]
                ),
            ],
        },
    )
    return trial


@pytest.fixture()
def no_trajectory_trial(tmp_path: Path) -> Path:
    """Trial dir with result.json but no trajectory.json."""
    trial = tmp_path / "trial_no_traj"
    trial.mkdir()
    _write_result(
        trial,
        {
            "task_name": "task_abc",
            "verifier_result": {"rewards": {"reward": 0.0}},
        },
    )
    return trial


@pytest.fixture()
def exception_trial(tmp_path: Path) -> Path:
    """Trial dir where exception_info is present."""
    trial = tmp_path / "trial_exc"
    trial.mkdir()
    _write_result(
        trial,
        {
            "task_name": "task_crash",
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": {"type": "RuntimeError", "message": "something broke"},
        },
    )
    return trial


@pytest.fixture()
def rate_limited_trial(tmp_path: Path) -> Path:
    """Trial dir with rate_limit in exception_info."""
    trial = tmp_path / "trial_rl"
    trial.mkdir()
    _write_result(
        trial,
        {
            "task_name": "task_rate",
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": {"type": "APIError", "message": "rate_limit exceeded"},
        },
    )
    return trial


# ---------------------------------------------------------------------------
# Tests: import checks
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_extract_signals(self) -> None:
        from agent_diagnostics.signals import extract_signals as fn

        assert callable(fn)

    def test_import_extract_all(self) -> None:
        from agent_diagnostics.signals import extract_all as fn

        assert callable(fn)

    def test_package_level_import(self) -> None:
        from agent_diagnostics import extract_all, extract_signals

        assert callable(extract_signals)
        assert callable(extract_all)


# ---------------------------------------------------------------------------
# Tests: basic extraction
# ---------------------------------------------------------------------------


class TestBasicExtraction:
    def test_returns_trial_signals_with_27_keys(self, basic_trial: Path) -> None:
        signals = extract_signals(
            basic_trial,
            suite_mapping={"fix_": "test_bench"},
        )
        expected_keys = {
            "task_id",
            "model",
            "config_name",
            "benchmark",
            "benchmark_source",
            "reward",
            "passed",
            "total_turns",
            "tool_calls_total",
            "search_tool_calls",
            "edit_tool_calls",
            "code_nav_tool_calls",
            "semantic_search_tool_calls",
            "unique_files_read",
            "unique_files_edited",
            "files_read_list",
            "files_edited_list",
            "error_count",
            "retry_count",
            "trajectory_length",
            "has_result_json",
            "has_trajectory",
            "duration_seconds",
            "rate_limited",
            "exception_crashed",
            "patch_size_lines",
            "tool_call_sequence",
        }
        assert set(signals.keys()) == expected_keys

    def test_reward_extracted(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["reward"] == 1.0

    def test_passed_true_when_reward_positive(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["passed"] is True

    def test_task_id_from_result(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["task_id"] == "fix_bug_123"

    def test_model_from_result(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["model"] == "anthropic/claude-sonnet-4-6"

    def test_has_result_json(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["has_result_json"] is True

    def test_has_trajectory(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["has_trajectory"] is True

    def test_duration_seconds(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["duration_seconds"] == 300.0

    def test_tool_counts(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["tool_calls_total"] == 4
        assert signals["search_tool_calls"] == 1  # Grep
        assert signals["edit_tool_calls"] == 1  # Edit
        assert signals["trajectory_length"] == 4

    def test_tool_call_sequence(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["tool_call_sequence"] == ["Grep", "Read", "Edit", "Bash"]

    def test_file_lists(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert "/src/main.py" in signals["files_read_list"]
        assert "/src/main.py" in signals["files_edited_list"]
        assert signals["unique_files_read"] >= 1
        assert signals["unique_files_edited"] == 1

    def test_total_turns(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["total_turns"] == 3

    def test_patch_size_lines(self, basic_trial: Path) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        # "fixed\nline\n" has 3 lines (2 newlines + 1)
        assert signals["patch_size_lines"] == 3


# ---------------------------------------------------------------------------
# Tests: missing trajectory
# ---------------------------------------------------------------------------


class TestMissingTrajectory:
    def test_has_trajectory_false(self, no_trajectory_trial: Path) -> None:
        signals = extract_signals(no_trajectory_trial, suite_mapping={})
        assert signals["has_trajectory"] is False

    def test_tool_counts_zero(self, no_trajectory_trial: Path) -> None:
        signals = extract_signals(no_trajectory_trial, suite_mapping={})
        assert signals["tool_calls_total"] == 0
        assert signals["trajectory_length"] == 0
        assert signals["tool_call_sequence"] == []

    def test_passed_false_when_reward_zero(self, no_trajectory_trial: Path) -> None:
        signals = extract_signals(no_trajectory_trial, suite_mapping={})
        assert signals["passed"] is False


# ---------------------------------------------------------------------------
# Tests: exception crash detection
# ---------------------------------------------------------------------------


class TestExceptionCrash:
    def test_exception_crashed_true(self, exception_trial: Path) -> None:
        signals = extract_signals(exception_trial, suite_mapping={})
        assert signals["exception_crashed"] is True

    def test_rate_limited_false_for_non_rate_exception(
        self, exception_trial: Path
    ) -> None:
        signals = extract_signals(exception_trial, suite_mapping={})
        assert signals["rate_limited"] is False


# ---------------------------------------------------------------------------
# Tests: rate limit detection
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_rate_limited_true(self, rate_limited_trial: Path) -> None:
        signals = extract_signals(rate_limited_trial, suite_mapping={})
        assert signals["rate_limited"] is True

    def test_exception_crashed_also_true(self, rate_limited_trial: Path) -> None:
        signals = extract_signals(rate_limited_trial, suite_mapping={})
        assert signals["exception_crashed"] is True


# ---------------------------------------------------------------------------
# Tests: suite mapping benchmark resolution
# ---------------------------------------------------------------------------


class TestSuiteMapping:
    def test_benchmark_from_suite_mapping(self, basic_trial: Path) -> None:
        mapping = {"fix_bug": "swebench_lite"}
        signals = extract_signals(basic_trial, suite_mapping=mapping)
        assert signals["benchmark"] == "swebench_lite"

    def test_longest_prefix_wins(self, basic_trial: Path) -> None:
        mapping = {
            "fix_": "generic_fix",
            "fix_bug": "swebench_lite",
        }
        signals = extract_signals(basic_trial, suite_mapping=mapping)
        assert signals["benchmark"] == "swebench_lite"

    def test_no_match_returns_empty(self, basic_trial: Path) -> None:
        mapping = {"unrelated_": "other_bench"}
        signals = extract_signals(basic_trial, suite_mapping=mapping)
        assert signals["benchmark"] == ""


# ---------------------------------------------------------------------------
# Tests: custom tool registry
# ---------------------------------------------------------------------------


class TestCustomToolRegistry:
    def test_custom_registry_categorization(self, tmp_path: Path) -> None:
        trial = tmp_path / "trial_custom"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "custom_task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call("my_search", {"query": "foo"}),
                            _make_tool_call(
                                "my_editor", {"file_path": "/a.py", "content": "x"}
                            ),
                        ]
                    ),
                ],
            },
        )

        custom_registry = ToolRegistry(
            search_tools=frozenset({"my_search"}),
            edit_tools=frozenset({"my_editor"}),
            code_nav_tools=frozenset(),
            semantic_search_tools=frozenset(),
        )
        signals = extract_signals(
            trial,
            tool_registry=custom_registry,
            suite_mapping={},
        )
        assert signals["search_tool_calls"] == 1
        assert signals["edit_tool_calls"] == 1
        assert signals["code_nav_tool_calls"] == 0


# ---------------------------------------------------------------------------
# Tests: warning emission
# ---------------------------------------------------------------------------


class TestWarningEmission:
    def test_warning_when_no_suite_mapping(self, basic_trial: Path) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extract_signals(basic_trial)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "suite_mapping" in str(user_warnings[0].message)

    def test_no_warning_with_suite_mapping(self, basic_trial: Path) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extract_signals(basic_trial, suite_mapping={})
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_no_warning_with_benchmark_resolver(self, basic_trial: Path) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extract_signals(basic_trial, benchmark_resolver=lambda p: "bench")
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0


# ---------------------------------------------------------------------------
# Tests: extract_all
# ---------------------------------------------------------------------------


class TestExtractAll:
    def test_finds_multiple_trial_dirs(self, tmp_path: Path) -> None:
        for i in range(3):
            trial = tmp_path / f"run_{i}" / "trial"
            trial.mkdir(parents=True)
            _write_result(
                trial,
                {
                    "task_name": f"task_{i}",
                    "agent_info": {"model_info": {"name": "test-model"}},
                    "verifier_result": {"rewards": {"reward": float(i)}},
                },
            )
        results = extract_all(tmp_path, suite_mapping={})
        assert len(results) == 3

    def test_returns_list_of_trial_signals(self, tmp_path: Path) -> None:
        trial = tmp_path / "single_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "only_task",
                "agent_info": {"model_info": {"name": "test-model"}},
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        results = extract_all(tmp_path, suite_mapping={})
        assert len(results) == 1
        assert results[0]["task_id"] == "only_task"

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        results = extract_all(tmp_path, suite_mapping={})
        assert results == []


# ---------------------------------------------------------------------------
# Tests: custom benchmark_resolver
# ---------------------------------------------------------------------------


class TestBenchmarkResolver:
    def test_custom_resolver_used(self, basic_trial: Path) -> None:
        def my_resolver(p: Path) -> str | None:
            return "custom_benchmark"

        signals = extract_signals(basic_trial, benchmark_resolver=my_resolver)
        assert signals["benchmark"] == "custom_benchmark"

    def test_resolver_returning_none_falls_through(self, basic_trial: Path) -> None:
        def null_resolver(p: Path) -> str | None:
            return None

        signals = extract_signals(
            basic_trial,
            benchmark_resolver=null_resolver,
            suite_mapping={"fix_": "fallback_bench"},
        )
        assert signals["benchmark"] == "fallback_bench"


# ---------------------------------------------------------------------------
# Tests: custom task_id_normalizer
# ---------------------------------------------------------------------------


class TestTaskIdNormalizer:
    def test_normalizer_applied(self, basic_trial: Path) -> None:
        signals = extract_signals(
            basic_trial,
            suite_mapping={},
            task_id_normalizer=lambda tid: tid.upper(),
        )
        assert signals["task_id"] == "FIX_BUG_123"

    def test_normalizer_affects_suite_mapping(self, basic_trial: Path) -> None:
        signals = extract_signals(
            basic_trial,
            suite_mapping={"FIX_": "normalized_bench"},
            task_id_normalizer=lambda tid: tid.upper(),
        )
        assert signals["benchmark"] == "normalized_bench"


# ---------------------------------------------------------------------------
# Tests: tool call sequence ordering
# ---------------------------------------------------------------------------


class TestToolCallSequence:
    def test_sequence_preserves_order(self, tmp_path: Path) -> None:
        trial = tmp_path / "seq_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "seq_task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step([_make_tool_call("Grep")]),
                    _make_step([_make_tool_call("Read")]),
                    _make_step([_make_tool_call("Edit")]),
                    _make_step([_make_tool_call("Bash")]),
                    _make_step([_make_tool_call("Grep")]),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["tool_call_sequence"] == ["Grep", "Read", "Edit", "Bash", "Grep"]


# ---------------------------------------------------------------------------
# Tests: file read/edit list extraction
# ---------------------------------------------------------------------------


class TestFileListExtraction:
    def test_unique_files_read(self, tmp_path: Path) -> None:
        trial = tmp_path / "file_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "file_task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call("Read", {"file_path": "/a.py"}),
                            _make_tool_call("Read", {"file_path": "/b.py"}),
                            _make_tool_call(
                                "Read", {"file_path": "/a.py"}
                            ),  # duplicate
                        ]
                    ),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["unique_files_read"] == 2
        assert sorted(signals["files_read_list"]) == ["/a.py", "/b.py"]

    def test_unique_files_edited(self, tmp_path: Path) -> None:
        trial = tmp_path / "edit_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "edit_task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call(
                                "Edit", {"file_path": "/x.py", "new_string": "a"}
                            ),
                            _make_tool_call(
                                "Write", {"file_path": "/y.py", "content": "b\nc"}
                            ),
                            _make_tool_call(
                                "Edit", {"file_path": "/x.py", "new_string": "d"}
                            ),
                        ]
                    ),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["unique_files_edited"] == 2
        assert sorted(signals["files_edited_list"]) == ["/x.py", "/y.py"]

    def test_mcp_read_file_path_extraction(self, tmp_path: Path) -> None:
        trial = tmp_path / "mcp_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "mcp_task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call(
                                "mcp__sourcegraph__read_file", {"path": "/remote.py"}
                            ),
                        ]
                    ),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert "/remote.py" in signals["files_read_list"]


# ---------------------------------------------------------------------------
# Tests: retry detection
# ---------------------------------------------------------------------------


class TestRetryDetection:
    def test_consecutive_same_calls_counted_as_retries(self, tmp_path: Path) -> None:
        trial = tmp_path / "retry_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "retry_task",
                "verifier_result": {"rewards": {"reward": 0.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step([_make_tool_call("Grep", {"pattern": "foo"})]),
                    _make_step([_make_tool_call("Grep", {"pattern": "foo"})]),
                    _make_step([_make_tool_call("Grep", {"pattern": "foo"})]),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["retry_count"] == 2


# ---------------------------------------------------------------------------
# Tests: error count
# ---------------------------------------------------------------------------


class TestErrorCount:
    def test_error_in_observation(self, tmp_path: Path) -> None:
        trial = tmp_path / "error_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "error_task",
                "verifier_result": {"rewards": {"reward": 0.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [_make_tool_call("Bash", {"command": "make"})],
                        {"results": [{"content": "Error: compilation failed"}]},
                    ),
                    _make_step(
                        [_make_tool_call("Bash", {"command": "make"})],
                        {"results": [{"content": "Success"}]},
                    ),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["error_count"] == 1


# ---------------------------------------------------------------------------
# Tests: reward via score fallback
# ---------------------------------------------------------------------------


class TestRewardScoreFallback:
    def test_score_field_used_when_reward_missing(self, tmp_path: Path) -> None:
        trial = tmp_path / "score_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "score_task",
                "verifier_result": {"rewards": {"score": 0.5}},
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["reward"] == 0.5


# ---------------------------------------------------------------------------
# Tests: no CSB imports or paths
# ---------------------------------------------------------------------------


class TestNoCsbDependencies:
    def test_no_csb_imports(self) -> None:
        import inspect
        import agent_diagnostics.signals as mod

        source = inspect.getsource(mod)
        assert "csb_metrics" not in source
        assert "CodeScaleBench" not in source
        assert "Path(__file__).parent.parent" not in source
        assert "sys.path" not in source


# ---------------------------------------------------------------------------
# Tests: _is_valid_trial
# ---------------------------------------------------------------------------


class TestIsValidTrial:
    def test_rejects_harness_summary_without_agent_info(self) -> None:
        """Harness summaries lack agent_info and should be rejected."""
        data = {
            "task_name": "summary",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }
        assert _is_valid_trial(data) is False

    def test_accepts_trial_with_agent_info(self) -> None:
        data = {
            "task_name": "real_trial",
            "agent_info": {"model_info": {"name": "anthropic/claude-sonnet-4-6"}},
            "verifier_result": {"rewards": {"reward": 1.0}},
        }
        assert _is_valid_trial(data) is True

    def test_rejects_empty_dict(self) -> None:
        assert _is_valid_trial({}) is False

    def test_accepts_minimal_agent_info(self) -> None:
        assert _is_valid_trial({"agent_info": {}}) is True


# ---------------------------------------------------------------------------
# Tests: _is_excluded_path
# ---------------------------------------------------------------------------


class TestIsExcludedPath:
    def test_excludes_archived_invalid(self) -> None:
        p = Path("/data/__archived_invalid/trial_01")
        assert _is_excluded_path(p) is True

    def test_excludes_incomplete(self) -> None:
        p = Path("/data/__incomplete/trial_01")
        assert _is_excluded_path(p) is True

    def test_excludes_pre_sgenv_fix(self) -> None:
        p = Path("/data/__pre_sgenv_fix/trial_01")
        assert _is_excluded_path(p) is True

    def test_excludes_verifier_path_bug(self) -> None:
        p = Path("/data/__verifier_path_bug/trial_01")
        assert _is_excluded_path(p) is True

    def test_excludes_doubled_prefix(self) -> None:
        p = Path("/data/__doubled_prefix/trial_01")
        assert _is_excluded_path(p) is True

    def test_does_not_exclude_normal_path(self) -> None:
        p = Path("/data/runs/trial_01")
        assert _is_excluded_path(p) is False

    def test_matches_pattern_anywhere_in_path(self) -> None:
        p = Path("/data/experiments/__incomplete/run_1/trial_01")
        assert _is_excluded_path(p) is True


# ---------------------------------------------------------------------------
# Tests: extract_all filtering
# ---------------------------------------------------------------------------


class TestExtractAllFiltering:
    def test_skips_trials_without_agent_info(self, tmp_path: Path) -> None:
        """Trials missing agent_info (harness summaries) are excluded."""
        # Valid trial
        valid = tmp_path / "valid_trial"
        valid.mkdir()
        _write_result(
            valid,
            {
                "task_name": "real_task",
                "agent_info": {"model_info": {"name": "test-model"}},
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )

        # Harness summary (no agent_info)
        invalid = tmp_path / "harness_summary"
        invalid.mkdir()
        _write_result(
            invalid,
            {
                "task_name": "summary",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )

        results = extract_all(tmp_path, suite_mapping={})
        assert len(results) == 1
        assert results[0]["task_id"] == "real_task"

    def test_skips_excluded_dir_patterns(self, tmp_path: Path) -> None:
        """Directories matching excluded patterns are skipped."""
        # Valid trial in excluded dir
        excluded = tmp_path / "__archived_invalid" / "trial_01"
        excluded.mkdir(parents=True)
        _write_result(
            excluded,
            {
                "task_name": "archived_task",
                "agent_info": {"model_info": {"name": "test-model"}},
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )

        # Valid trial in normal dir
        normal = tmp_path / "good_run" / "trial_01"
        normal.mkdir(parents=True)
        _write_result(
            normal,
            {
                "task_name": "good_task",
                "agent_info": {"model_info": {"name": "test-model"}},
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )

        results = extract_all(tmp_path, suite_mapping={})
        assert len(results) == 1
        assert results[0]["task_id"] == "good_task"

    def test_skips_all_excluded_patterns(self, tmp_path: Path) -> None:
        """All five excluded patterns are filtered out."""
        patterns = [
            "__archived_invalid",
            "__incomplete",
            "__pre_sgenv_fix",
            "__verifier_path_bug",
            "__doubled_prefix",
        ]
        for i, pattern in enumerate(patterns):
            trial = tmp_path / pattern / f"trial_{i}"
            trial.mkdir(parents=True)
            _write_result(
                trial,
                {
                    "task_name": f"task_{i}",
                    "agent_info": {"model_info": {"name": "test-model"}},
                    "verifier_result": {"rewards": {"reward": 1.0}},
                },
            )

        results = extract_all(tmp_path, suite_mapping={})
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: OpenHands auto-detection via agent_info.name
# ---------------------------------------------------------------------------


class TestOpenHandsAutoDetection:
    """Verify extract_signals auto-selects OPENHANDS_REGISTRY from agent_info."""

    def test_openhands_agent_uses_openhands_registry(self, tmp_path: Path) -> None:
        """When agent_info.name contains 'openhands', tool calls are categorized
        using the OpenHands registry."""
        trial = tmp_path / "oh_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "oh_task",
                "agent_info": {
                    "name": "openhands-agent-v1",
                    "model_info": {"name": "gpt-4"},
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call("execute_bash", {"command": "grep foo"}),
                            _make_tool_call(
                                "str_replace_editor",
                                {"file_path": "/src/app.py", "new_string": "fixed\n"},
                            ),
                            _make_tool_call("browser", {"url": "http://localhost"}),
                        ]
                    ),
                ],
            },
        )
        # Do NOT pass tool_registry — should auto-detect from agent_info.name
        signals = extract_signals(trial, suite_mapping={})
        assert signals["search_tool_calls"] == 1  # execute_bash
        assert signals["edit_tool_calls"] == 1  # str_replace_editor
        assert signals["code_nav_tool_calls"] == 1  # browser

    def test_non_openhands_agent_uses_default_registry(self, tmp_path: Path) -> None:
        """When agent_info.name does not contain 'openhands', default registry
        is used."""
        trial = tmp_path / "cc_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "cc_task",
                "agent_info": {
                    "name": "claude-code",
                    "model_info": {"name": "anthropic/claude-sonnet-4-6"},
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call("Grep", {"pattern": "bug"}),
                            _make_tool_call(
                                "Edit",
                                {"file_path": "/src/app.py", "new_string": "fixed\n"},
                            ),
                        ]
                    ),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["search_tool_calls"] == 1  # Grep via DEFAULT_REGISTRY
        assert signals["edit_tool_calls"] == 1  # Edit via DEFAULT_REGISTRY

    def test_explicit_registry_overrides_auto_detection(self, tmp_path: Path) -> None:
        """When tool_registry is explicitly passed, agent_info.name is ignored."""
        trial = tmp_path / "override_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "override_task",
                "agent_info": {
                    "name": "openhands-agent",
                    "model_info": {"name": "gpt-4"},
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call("Grep", {"pattern": "foo"}),
                        ]
                    ),
                ],
            },
        )
        # Explicitly pass DEFAULT_REGISTRY even though agent is openhands
        signals = extract_signals(
            trial, tool_registry=DEFAULT_REGISTRY, suite_mapping={}
        )
        assert signals["search_tool_calls"] == 1  # Grep categorized by DEFAULT

    def test_missing_agent_info_falls_back_to_default(self, tmp_path: Path) -> None:
        """When agent_info.name is absent, DEFAULT_REGISTRY is used."""
        trial = tmp_path / "no_agent_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "no_agent_task",
                "agent_info": {},
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        _write_trajectory(
            trial,
            {
                "steps": [
                    _make_step(
                        [
                            _make_tool_call("Grep", {"pattern": "test"}),
                        ]
                    ),
                ],
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["search_tool_calls"] == 1  # Grep via DEFAULT_REGISTRY
