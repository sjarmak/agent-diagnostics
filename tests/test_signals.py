"""Tests for agent_diagnostics.signals module."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from agent_diagnostics.signals import (
    _is_excluded_path,
    _is_valid_trial,
    _parse_benchmark_from_run_name,
    _resolve_benchmark_from_directory,
    extract_all,
    extract_signals,
    load_manifest,
)
from agent_diagnostics.tool_registry import (
    DEFAULT_REGISTRY,
    ToolRegistry,
)

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
    def test_returns_trial_signals_with_29_keys(self, basic_trial: Path) -> None:
        signals = extract_signals(
            basic_trial,
            suite_mapping={"fix_": "test_bench"},
        )
        expected_keys = {
            "trial_id",
            "trial_id_full",
            "task_id",
            "model",
            "agent_name",
            "config_name",
            "benchmark",
            "benchmark_source",
            "reward",
            "passed",
            "has_verifier_result",
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
# Tests: nullable reward and has_verifier_result
# ---------------------------------------------------------------------------


class TestNullableReward:
    def test_reward_none_when_no_verifier_result(self, tmp_path: Path) -> None:
        """When result.json has no verifier_result, reward should be None."""
        trial = tmp_path / "no_verifier_trial"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "no_verifier_task",
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["reward"] is None

    def test_has_verifier_result_false_when_no_verifier(self, tmp_path: Path) -> None:
        trial = tmp_path / "no_verifier_trial2"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "no_verifier_task",
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["has_verifier_result"] is False

    def test_has_verifier_result_true_when_verifier_present(
        self, basic_trial: Path
    ) -> None:
        signals = extract_signals(basic_trial, suite_mapping={})
        assert signals["has_verifier_result"] is True

    def test_passed_false_when_reward_none(self, tmp_path: Path) -> None:
        trial = tmp_path / "no_verifier_trial3"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "no_verifier_task",
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["passed"] is False

    def test_reward_none_when_no_result_json(self, tmp_path: Path) -> None:
        """When there's no result.json at all, reward should be None."""
        trial = tmp_path / "empty_trial"
        trial.mkdir()
        signals = extract_signals(trial, suite_mapping={})
        assert signals["reward"] is None
        assert signals["has_verifier_result"] is False

    def test_reward_zero_preserved_as_zero(self, no_trajectory_trial: Path) -> None:
        """reward=0.0 from verifier should remain 0.0, not become None."""
        signals = extract_signals(no_trajectory_trial, suite_mapping={})
        assert signals["reward"] == 0.0
        assert signals["has_verifier_result"] is True


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


# ---------------------------------------------------------------------------
# Tests: load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_parses_valid_manifest(self, tmp_path: Path) -> None:
        manifest = {
            "csb_crossrepo_run1": "crossrepo",
            "openhands_eval_42": "openhands",
            "swe_bench_lite": "swe-bench",
        }
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest))
        result = load_manifest(manifest_path)
        assert result == manifest

    def test_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        result = load_manifest(tmp_path / "nonexistent.json")
        assert result == {}

    def test_returns_empty_dict_for_malformed_json(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text("not valid json{{{")
        result = load_manifest(manifest_path)
        assert result == {}

    def test_returns_empty_dict_for_non_dict_json(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps(["a", "b"]))
        result = load_manifest(manifest_path)
        assert result == {}

    def test_filters_non_string_values(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps({"valid": "bench", "bad_key": 123}))
        result = load_manifest(manifest_path)
        assert "valid" in result
        assert result["valid"] == "bench"

    def test_usable_as_suite_mapping(self, tmp_path: Path) -> None:
        """Manifest output can be passed directly as suite_mapping."""
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps({"fix_bug": "swebench_lite"}))
        mapping = load_manifest(manifest_path)

        trial = tmp_path / "trial_01"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "fix_bug_123",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_info": {"model_info": {"name": "test-model"}},
            },
        )
        signals = extract_signals(trial, suite_mapping=mapping)
        assert signals["benchmark"] == "swebench_lite"
        assert signals["benchmark_source"] == "manifest"


# ---------------------------------------------------------------------------
# Tests: directory-name benchmark resolution
# ---------------------------------------------------------------------------


class TestDirectoryBenchmarkResolution:
    def test_crossrepo_from_dir_name(self, tmp_path: Path) -> None:
        trial = tmp_path / "csb_crossrepo_run1" / "trial"
        trial.mkdir(parents=True)
        assert _resolve_benchmark_from_directory(trial) == "crossrepo"

    def test_openhands_from_dir_name(self, tmp_path: Path) -> None:
        trial = tmp_path / "openhands_eval_42" / "trial"
        trial.mkdir(parents=True)
        assert _resolve_benchmark_from_directory(trial) == "openhands"

    def test_swe_bench_hyphen_from_dir_name(self, tmp_path: Path) -> None:
        trial = tmp_path / "swe-bench-lite" / "trial"
        trial.mkdir(parents=True)
        assert _resolve_benchmark_from_directory(trial) == "swe-bench"

    def test_swe_bench_underscore_from_dir_name(self, tmp_path: Path) -> None:
        trial = tmp_path / "swe_bench_verified" / "trial"
        trial.mkdir(parents=True)
        assert _resolve_benchmark_from_directory(trial) == "swe-bench"

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        trial = tmp_path / "unknown_benchmark" / "trial"
        trial.mkdir(parents=True)
        assert _resolve_benchmark_from_directory(trial) is None

    def test_case_insensitive(self, tmp_path: Path) -> None:
        trial = tmp_path / "CSB_CrossRepo_Run1" / "trial"
        trial.mkdir(parents=True)
        assert _resolve_benchmark_from_directory(trial) == "crossrepo"

    def test_directory_fallback_in_extract_signals(self, tmp_path: Path) -> None:
        """When no suite_mapping matches, directory convention is used."""
        trial = tmp_path / "csb_crossrepo_run1" / "trial_01"
        trial.mkdir(parents=True)
        _write_result(
            trial,
            {
                "task_name": "unmatched_task_id",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_info": {"model_info": {"name": "test-model"}},
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["benchmark"] == "crossrepo"
        assert signals["benchmark_source"] == "directory"


# ---------------------------------------------------------------------------
# Tests: benchmark_source provenance
# ---------------------------------------------------------------------------


class TestBenchmarkSource:
    def test_source_manifest_from_suite_mapping(self, basic_trial: Path) -> None:
        signals = extract_signals(
            basic_trial, suite_mapping={"fix_bug": "swebench_lite"}
        )
        assert signals["benchmark_source"] == "manifest"

    def test_source_manifest_from_benchmark_resolver(self, basic_trial: Path) -> None:
        signals = extract_signals(
            basic_trial, benchmark_resolver=lambda p: "custom_bench"
        )
        assert signals["benchmark_source"] == "manifest"

    def test_source_directory_fallback(self, tmp_path: Path) -> None:
        trial = tmp_path / "openhands_run" / "trial_01"
        trial.mkdir(parents=True)
        _write_result(
            trial,
            {
                "task_name": "some_task",
                "verifier_result": {"rewards": {"reward": 0.5}},
                "agent_info": {"model_info": {"name": "test-model"}},
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["benchmark"] == "openhands"
        assert signals["benchmark_source"] == "directory"

    def test_source_empty_when_unresolved(self, tmp_path: Path) -> None:
        trial = tmp_path / "unknown_dir" / "trial_01"
        trial.mkdir(parents=True)
        _write_result(
            trial,
            {
                "task_name": "no_match_task",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_info": {"model_info": {"name": "test-model"}},
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["benchmark"] == ""
        assert signals["benchmark_source"] == ""

    def test_suite_mapping_takes_precedence_over_directory(
        self, tmp_path: Path
    ) -> None:
        """When suite_mapping matches, directory convention is not used."""
        trial = tmp_path / "openhands_run" / "trial_01"
        trial.mkdir(parents=True)
        _write_result(
            trial,
            {
                "task_name": "oh_task_1",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_info": {"model_info": {"name": "test-model"}},
            },
        )
        signals = extract_signals(trial, suite_mapping={"oh_task": "custom_openhands"})
        assert signals["benchmark"] == "custom_openhands"
        assert signals["benchmark_source"] == "manifest"


# ---------------------------------------------------------------------------
# JSONL IO helpers
# ---------------------------------------------------------------------------


class TestWriteJsonl:
    """Tests for write_jsonl."""

    def test_writes_one_object_per_line(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import write_jsonl

        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        out = tmp_path / "out.jsonl"
        write_jsonl(data, out)

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # each line is valid JSON

    def test_creates_meta_sidecar(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import write_jsonl

        out = tmp_path / "signals.jsonl"
        meta_path = write_jsonl(
            [{"x": 1}],
            out,
            schema_version="test-v1",
            taxonomy_version="3.0",
        )

        assert meta_path.exists()
        assert meta_path.name == "signals.meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == "test-v1"
        assert meta["taxonomy_version"] == "3.0"
        assert "generated_at" in meta

    def test_roundtrip(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import load_jsonl, write_jsonl

        data = [{"task_id": "t1", "reward": 1.0}, {"task_id": "t2", "reward": 0.0}]
        out = tmp_path / "rt.jsonl"
        write_jsonl(data, out)
        loaded = load_jsonl(out)
        assert loaded == data


class TestLoadSignals:
    """Tests for load_signals (transparent format detection)."""

    def test_load_json(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import load_signals

        data = [{"task_id": "t1"}, {"task_id": "t2"}]
        p = tmp_path / "signals.json"
        p.write_text(json.dumps(data))
        loaded = load_signals(p)
        assert loaded == data

    def test_load_jsonl(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import load_signals, write_jsonl

        data = [{"task_id": "t1"}, {"task_id": "t2"}]
        p = tmp_path / "signals.jsonl"
        write_jsonl(data, p)
        loaded = load_signals(p)
        assert loaded == data

    def test_json_non_array_raises(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import load_signals

        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="Expected a JSON array"):
            load_signals(p)


class TestLoadAnnotations:
    """Tests for load_annotations (transparent format detection)."""

    def test_load_json_envelope(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import load_annotations

        envelope = {
            "schema_version": "observatory-annotation-v1",
            "taxonomy_version": "3.0",
            "annotations": [{"task_id": "t1", "categories": []}],
        }
        p = tmp_path / "ann.json"
        p.write_text(json.dumps(envelope))
        loaded = load_annotations(p)
        assert loaded == envelope

    def test_load_jsonl_reconstructs_envelope(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import load_annotations, write_jsonl

        items = [
            {"task_id": "t1", "categories": []},
            {"task_id": "t2", "categories": []},
        ]
        p = tmp_path / "ann.jsonl"
        write_jsonl(
            items,
            p,
            schema_version="observatory-annotation-v1",
            taxonomy_version="3.0",
        )
        loaded = load_annotations(p)
        assert loaded["schema_version"] == "observatory-annotation-v1"
        assert loaded["taxonomy_version"] == "3.0"
        assert loaded["annotations"] == items
        assert "generated_at" in loaded


class TestWriteOutput:
    """Tests for write_output (format detection by extension)."""

    def test_json_extension_writes_json(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import write_output

        data = [{"a": 1}, {"b": 2}]
        p = tmp_path / "out.json"
        write_output(data, p)
        loaded = json.loads(p.read_text())
        assert loaded == data

    def test_jsonl_extension_writes_jsonl(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import write_output

        data = [{"a": 1}, {"b": 2}]
        p = tmp_path / "out.jsonl"
        write_output(data, p)

        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

        meta_path = p.with_suffix(".meta.json")
        assert meta_path.exists()

    def test_jsonl_with_annotation_envelope(self, tmp_path: Path) -> None:
        from agent_diagnostics.signals import write_output

        envelope = {
            "schema_version": "observatory-annotation-v1",
            "taxonomy_version": "3.0",
            "annotations": [{"task_id": "t1"}, {"task_id": "t2"}],
        }
        p = tmp_path / "ann.jsonl"
        write_output(envelope, p)

        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2

        meta = json.loads(p.with_suffix(".meta.json").read_text())
        assert meta["schema_version"] == "observatory-annotation-v1"
        assert meta["taxonomy_version"] == "3.0"


class TestIsJsonlPath:
    """Tests for is_jsonl_path."""

    def test_jsonl(self) -> None:
        from agent_diagnostics.signals import is_jsonl_path

        assert is_jsonl_path("foo.jsonl") is True
        assert is_jsonl_path(Path("bar/baz.jsonl")) is True

    def test_json(self) -> None:
        from agent_diagnostics.signals import is_jsonl_path

        assert is_jsonl_path("foo.json") is False
        assert is_jsonl_path(Path("bar.json")) is False


# ---------------------------------------------------------------------------
# Tests: _parse_benchmark_from_run_name (structural run-dir parser)
# ---------------------------------------------------------------------------


class TestParseBenchmarkFromRunName:
    @pytest.mark.parametrize(
        "dir_name,expected",
        [
            ("csb_sdlc_fix_haiku_20260228_123456", "sdlc_fix"),
            ("csb_org_crossrepo_tracing_sonnet46_20260315", "org_crossrepo_tracing"),
            ("openhands_haiku45_20260301", "openhands"),
            ("openhands_sonnet46", "openhands"),
            ("crossrepo_opus_20260202_204730", "crossrepo"),
            ("_browse_cc_sonnet46", "browse"),
            ("bigcode_mcp_opus_20260204_023501", "bigcode_mcp"),
            ("ccb_build_haiku_022326", "build"),
            ("csb_sdlc_fix_haiku_20260228_123456__promoted", "sdlc_fix"),
        ],
    )
    def test_valid_run_names(self, dir_name: str, expected: str) -> None:
        assert _parse_benchmark_from_run_name(dir_name) == expected

    @pytest.mark.parametrize(
        "dir_name",
        [
            "",
            "home",
            "projects",
            "CodeScaleBench",
            "_raw",
            "baseline",
            "trial_01",
            "task-001",
        ],
    )
    def test_names_without_marker_return_none(self, dir_name: str) -> None:
        assert _parse_benchmark_from_run_name(dir_name) is None


# ---------------------------------------------------------------------------
# Tests: load_manifest with nested run_history shape
# ---------------------------------------------------------------------------


class TestLoadManifestRunHistory:
    def test_parses_nested_run_history(self, tmp_path: Path) -> None:
        manifest = {
            "description": "test",
            "generated": "2026-01-01",
            "run_history": {
                "ccb_crossrepo/baseline": {
                    "cr-aspnet-001": {"n_runs": 3, "mean_reward": 1.0},
                    "cr-envoy-001": {"n_runs": 3, "mean_reward": 0.0},
                },
                "ccb_dependeval/sourcegraph_full": {
                    "dep-pytorch-001": {"n_runs": 2, "mean_reward": 0.5},
                },
            },
        }
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest))
        result = load_manifest(manifest_path)
        assert result["cr-aspnet-001"] == "crossrepo"
        assert result["cr-envoy-001"] == "crossrepo"
        assert result["dep-pytorch-001"] == "dependeval"

    def test_run_history_takes_precedence_over_flat(self, tmp_path: Path) -> None:
        """When run_history is present, the flat shape is ignored."""
        manifest = {
            "run_history": {
                "ccb_swe_bench/default": {"task-1": {}},
            },
            "description": "should-not-be-used",
        }
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest))
        result = load_manifest(manifest_path)
        assert result == {"task-1": "swe_bench"}

    def test_empty_run_history_falls_through_to_flat(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps({"run_history": {}, "foo": "bar"}))
        result = load_manifest(manifest_path)
        assert result == {"foo": "bar"}

    def test_ignores_non_ccb_csb_prefix(self, tmp_path: Path) -> None:
        """Benchmark keys that do not start with csb_/ccb_ are kept intact."""
        manifest = {
            "run_history": {
                "other_bench/default": {"task-a": {}},
            },
        }
        manifest_path = tmp_path / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest))
        result = load_manifest(manifest_path)
        assert result == {"task-a": "other_bench"}


# ---------------------------------------------------------------------------
# Tests: agent_name field in extracted signals
# ---------------------------------------------------------------------------


class TestAgentNameField:
    def test_agent_name_populated_from_agent_info(self, tmp_path: Path) -> None:
        trial = tmp_path / "trial_01"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "t1",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_info": {
                    "name": "openhands",
                    "model_info": {"name": "anthropic/claude"},
                },
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["agent_name"] == "openhands"

    def test_agent_name_empty_when_missing(self, tmp_path: Path) -> None:
        trial = tmp_path / "trial_02"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "t2",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_info": {"model_info": {"name": "anthropic/claude"}},
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["agent_name"] == ""

    def test_agent_name_claude_code(self, tmp_path: Path) -> None:
        trial = tmp_path / "trial_03"
        trial.mkdir()
        _write_result(
            trial,
            {
                "task_name": "t3",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_info": {
                    "name": "claude-code",
                    "model_info": {"name": "anthropic/claude"},
                },
            },
        )
        signals = extract_signals(trial, suite_mapping={})
        assert signals["agent_name"] == "claude-code"


# ---------------------------------------------------------------------------
# Tests: token-bounded prefix matching against task_id
# ---------------------------------------------------------------------------


class TestTokenBoundedPrefixMatch:
    def _trial_with_task(self, tmp_path: Path, task_id: str) -> Path:
        trial = tmp_path / "run_dir" / "trial_01"
        trial.mkdir(parents=True)
        _write_result(
            trial,
            {
                "task_name": task_id,
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_info": {"model_info": {"name": "m"}},
            },
        )
        return trial

    def test_exact_task_id_match(self, tmp_path: Path) -> None:
        trial = self._trial_with_task(tmp_path, "task-10")
        signals = extract_signals(
            trial, suite_mapping={"task-10": "bench_ten", "task-1": "bench_one"}
        )
        assert signals["benchmark"] == "bench_ten"

    def test_prefix_does_not_spill_across_token_boundary(self, tmp_path: Path) -> None:
        """'task-10' must not match prefix 'task-1' (non-boundary)."""
        trial = self._trial_with_task(tmp_path, "task-10")
        signals = extract_signals(trial, suite_mapping={"task-1": "bench_one"})
        assert signals["benchmark"] != "bench_one"

    def test_prefix_with_trailing_underscore_is_boundary(self, tmp_path: Path) -> None:
        """Legacy prefix like 'fix_' must still match 'fix_bug_123'."""
        trial = self._trial_with_task(tmp_path, "fix_bug_123")
        signals = extract_signals(trial, suite_mapping={"fix_": "bench_fix"})
        assert signals["benchmark"] == "bench_fix"

    def test_prefix_match_at_dash_boundary(self, tmp_path: Path) -> None:
        """'cr-aspnet' prefix matches 'cr-aspnet-001' at the dash boundary."""
        trial = self._trial_with_task(tmp_path, "cr-aspnet-001")
        signals = extract_signals(trial, suite_mapping={"cr-aspnet": "crossrepo"})
        assert signals["benchmark"] == "crossrepo"
