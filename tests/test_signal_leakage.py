"""Tests for signal redaction — ensures judge-facing input excludes reward-leaking fields."""

from __future__ import annotations

from typing import Any

from agent_diagnostics.signals import judge_safe_signals


def _make_signals(**overrides: Any) -> dict[str, Any]:
    """Return a realistic signals dict with sensible defaults."""
    base: dict[str, Any] = {
        "task_id": "task-001",
        "model": "anthropic/claude-sonnet-4-6",
        "config_name": "default",
        "benchmark": "swe-bench",
        "reward": 1.0,
        "passed": True,
        "total_turns": 5,
        "tool_calls_total": 12,
        "search_tool_calls": 3,
        "edit_tool_calls": 4,
        "code_nav_tool_calls": 2,
        "semantic_search_tool_calls": 1,
        "unique_files_read": 6,
        "unique_files_edited": 3,
        "files_read_list": ["a.py", "b.py"],
        "files_edited_list": ["a.py"],
        "error_count": 0,
        "retry_count": 0,
        "trajectory_length": 12,
        "has_result_json": True,
        "has_trajectory": True,
        "duration_seconds": 120.0,
        "rate_limited": False,
        "exception_crashed": False,
        "exception_info": None,
        "patch_size_lines": 25,
        "tool_call_sequence": ["Read", "Edit", "Read"],
    }
    base.update(overrides)
    return base


class TestJudgeInputExcludesReward:
    """Judge-safe signals must not contain reward or reward-derived fields."""

    def test_judge_input_excludes_reward(self) -> None:
        signals = _make_signals(reward=1.0, passed=True, exception_info="boom")
        safe = judge_safe_signals(signals)

        assert "reward" not in safe
        assert "passed" not in safe
        assert "exception_info" not in safe

    def test_judge_input_excludes_exception_crashed(self) -> None:
        signals = _make_signals(exception_crashed=True)
        safe = judge_safe_signals(signals)

        assert "exception_crashed" not in safe

    def test_safe_signals_preserves_non_redacted_fields(self) -> None:
        signals = _make_signals()
        safe = judge_safe_signals(signals)

        assert safe["task_id"] == "task-001"
        assert safe["model"] == "anthropic/claude-sonnet-4-6"
        assert safe["total_turns"] == 5
        assert safe["tool_calls_total"] == 12
        assert safe["duration_seconds"] == 120.0

    def test_safe_signals_does_not_mutate_original(self) -> None:
        signals = _make_signals(reward=0.5)
        _ = judge_safe_signals(signals)

        assert "reward" in signals
        assert signals["reward"] == 0.5

    def test_empty_signals(self) -> None:
        safe = judge_safe_signals({})
        assert safe == {}

    def test_signals_with_only_redacted_keys(self) -> None:
        signals = {
            "reward": 1.0,
            "passed": True,
            "exception_info": "err",
            "exception_crashed": True,
        }
        safe = judge_safe_signals(signals)
        assert safe == {}
