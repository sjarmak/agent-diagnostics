"""Tests for the agent-observatory LLM annotator module."""

from __future__ import annotations

import builtins
import inspect
import json
from pathlib import Path
from unittest import mock

import pytest

from agent_diagnostics import llm_annotator
from agent_diagnostics.llm_annotator import (
    _load_json,
    _read_text,
    _resolve_model_alias,
    _resolve_model_api,
    annotate_batch,
    annotate_trial_api,
    annotate_trial_claude_code,
    annotate_trial_llm,
    build_prompt,
    summarise_step,
    truncate_trajectory,
    validate_categories,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Public API import tests
# ---------------------------------------------------------------------------


class TestPublicImports:
    """Verify all required public names are importable."""

    def test_annotate_trial_llm_importable(self) -> None:
        assert callable(annotate_trial_llm)

    def test_annotate_batch_importable(self) -> None:
        assert callable(annotate_batch)

    def test_annotate_trial_claude_code_importable(self) -> None:
        assert callable(annotate_trial_claude_code)

    def test_annotate_trial_api_importable(self) -> None:
        assert callable(annotate_trial_api)

    def test_build_prompt_importable(self) -> None:
        assert callable(build_prompt)

    def test_validate_categories_importable(self) -> None:
        assert callable(validate_categories)

    def test_truncate_trajectory_importable(self) -> None:
        assert callable(truncate_trajectory)

    def test_summarise_step_importable(self) -> None:
        assert callable(summarise_step)


# ---------------------------------------------------------------------------
# judge_trial must NOT exist
# ---------------------------------------------------------------------------


class TestNoJudge:
    """Ensure CSB-specific judge functions were removed."""

    def test_no_judge_trial(self) -> None:
        assert not hasattr(llm_annotator, "judge_trial")

    def test_no_build_judge_input(self) -> None:
        assert not hasattr(llm_annotator, "_build_judge_input")

    def test_no_extract_code_changes(self) -> None:
        assert not hasattr(llm_annotator, "_extract_code_changes")


# ---------------------------------------------------------------------------
# No sys.path manipulation in source
# ---------------------------------------------------------------------------


class TestNoSysPathManipulation:
    """Verify no sys.path hacks in the module source."""

    def test_no_sys_path_in_source(self) -> None:
        source = inspect.getsource(llm_annotator)
        assert "sys.path" not in source

    def test_no_csb_metrics_import(self) -> None:
        source = inspect.getsource(llm_annotator)
        assert "csb_metrics" not in source


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Test prompt construction."""

    def test_contains_taxonomy_section(self) -> None:
        prompt = build_prompt("do X", [], {"reward": 1.0}, "categories:\n  - foo")
        assert "## Taxonomy" in prompt
        assert "categories:" in prompt

    def test_contains_instruction_section(self) -> None:
        prompt = build_prompt("solve the bug", [], {}, "tax: y")
        assert "## Task instruction" in prompt
        assert "solve the bug" in prompt

    def test_contains_trajectory_section(self) -> None:
        steps = [{"tool_calls": [{"function_name": "Read", "arguments": {}}]}]
        prompt = build_prompt("task", steps, {}, "tax: y")
        assert "## Trajectory" in prompt
        assert "Read" in prompt

    def test_contains_signals_section(self) -> None:
        prompt = build_prompt("task", [], {"reward": 0.5}, "tax: y")
        assert "## Extracted signals" in prompt
        assert "0.5" in prompt

    def test_contains_instructions_section(self) -> None:
        prompt = build_prompt("task", [], {}, "tax: y")
        assert "## Instructions" in prompt

    def test_no_instruction_shows_not_available(self) -> None:
        prompt = build_prompt(None, [], {}, "tax: y")
        assert "(not available)" in prompt

    def test_no_trajectory_shows_placeholder(self) -> None:
        prompt = build_prompt("task", [], {}, "tax: y")
        assert "(no trajectory available)" in prompt

    def test_filters_tool_calls_by_name(self) -> None:
        prompt = build_prompt(
            "task",
            [],
            {"tool_calls_by_name": {"Read": 5}, "reward": 1.0},
            "tax: y",
        )
        assert "tool_calls_by_name" not in prompt

    def test_filters_trial_path(self) -> None:
        prompt = build_prompt(
            "task",
            [],
            {"trial_path": "/foo", "reward": 1.0},
            "tax: y",
        )
        assert "trial_path" not in prompt


# ---------------------------------------------------------------------------
# validate_categories
# ---------------------------------------------------------------------------


class TestValidateCategories:
    """Test category validation against taxonomy."""

    def test_valid_category_kept(self) -> None:
        cats = [{"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"}]
        result = validate_categories(cats, "/tmp/trial")
        assert len(result) == 1
        assert result[0]["name"] == "retrieval_failure"

    def test_invalid_category_filtered(self) -> None:
        cats = [{"name": "not_real_category_xyz", "confidence": 0.5, "evidence": "e"}]
        result = validate_categories(cats, "/tmp/trial")
        assert len(result) == 0

    def test_mixed_valid_invalid(self) -> None:
        cats = [
            {"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"},
            {"name": "bogus_category", "confidence": 0.5, "evidence": "e"},
            {"name": "query_churn", "confidence": 0.7, "evidence": "e2"},
        ]
        result = validate_categories(cats, "/tmp/trial")
        assert len(result) == 2
        names = {c["name"] for c in result}
        assert names == {"retrieval_failure", "query_churn"}

    def test_non_dict_entries_skipped(self) -> None:
        cats = ["not a dict", 42, None]
        result = validate_categories(cats, "/tmp/trial")
        assert len(result) == 0

    def test_default_confidence(self) -> None:
        cats = [{"name": "retrieval_failure", "evidence": "e"}]
        result = validate_categories(cats, "/tmp/trial")
        assert result[0]["confidence"] == 0.6

    def test_empty_list(self) -> None:
        result = validate_categories([], "/tmp/trial")
        assert result == []


# ---------------------------------------------------------------------------
# truncate_trajectory
# ---------------------------------------------------------------------------


class TestTruncateTrajectory:
    """Test trajectory truncation."""

    def test_none_input(self) -> None:
        assert truncate_trajectory(None) == []

    def test_empty_steps(self) -> None:
        assert truncate_trajectory({"steps": []}) == []

    def test_short_trajectory_kept_intact(self) -> None:
        steps = [{"id": i} for i in range(10)]
        result = truncate_trajectory({"steps": steps}, first_n=30, last_n=10)
        assert len(result) == 10

    def test_long_trajectory_truncated(self) -> None:
        steps = [{"id": i} for i in range(100)]
        result = truncate_trajectory({"steps": steps}, first_n=5, last_n=3)
        # 5 first + 1 marker + 3 last = 9
        assert len(result) == 9
        assert result[0]["id"] == 0
        assert result[4]["id"] == 4
        assert "_marker" in result[5]
        assert "92 steps omitted" in result[5]["_marker"]
        assert result[6]["id"] == 97
        assert result[8]["id"] == 99

    def test_exact_boundary(self) -> None:
        steps = [{"id": i} for i in range(40)]
        result = truncate_trajectory({"steps": steps}, first_n=30, last_n=10)
        # Exactly first_n + last_n, so no truncation
        assert len(result) == 40

    def test_missing_steps_key(self) -> None:
        assert truncate_trajectory({}) == []


# ---------------------------------------------------------------------------
# summarise_step
# ---------------------------------------------------------------------------


class TestSummariseStep:
    """Test step summarisation."""

    def test_tool_call_summarised(self) -> None:
        step = {
            "tool_calls": [
                {"function_name": "Read", "arguments": {"path": "/foo.py"}},
            ],
        }
        result = summarise_step(step)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["tool"] == "Read"

    def test_long_args_truncated(self) -> None:
        step = {
            "tool_calls": [
                {"function_name": "Write", "arguments": {"content": "x" * 500}},
            ],
        }
        result = summarise_step(step)
        content_val = result["tool_calls"][0]["args"]["content"]
        assert content_val.endswith("...")
        assert len(content_val) <= 204  # 200 + "..."

    def test_observation_summarised(self) -> None:
        step = {
            "observation": {
                "results": [{"content": "hello world"}],
            },
        }
        result = summarise_step(step)
        assert "observation" in result
        assert "hello world" in result["observation"][0]

    def test_marker_preserved(self) -> None:
        step = {"_marker": "... (50 steps omitted) ..."}
        result = summarise_step(step)
        assert result["_marker"] == "... (50 steps omitted) ..."

    def test_empty_step(self) -> None:
        result = summarise_step({})
        assert result == {}

    def test_string_args_truncated(self) -> None:
        step = {
            "tool_calls": [
                {"function_name": "Bash", "arguments": "a" * 500},
            ],
        }
        result = summarise_step(step)
        args_val = result["tool_calls"][0]["args"]
        assert args_val.endswith("...")


# ---------------------------------------------------------------------------
# annotate_trial_llm dispatch
# ---------------------------------------------------------------------------


class TestAnnotateTrialLlmDispatch:
    """Test that annotate_trial_llm dispatches to the correct backend."""

    def test_dispatches_to_claude_code(self) -> None:
        with mock.patch.object(
            llm_annotator,
            "annotate_trial_claude_code",
            return_value=[],
        ) as m:
            annotate_trial_llm("/tmp/trial", {}, backend="claude-code")
            m.assert_called_once_with("/tmp/trial", {}, "haiku")

    def test_dispatches_to_api(self) -> None:
        with mock.patch.object(
            llm_annotator,
            "annotate_trial_api",
            return_value=[],
        ) as m:
            annotate_trial_llm("/tmp/trial", {}, backend="api")
            m.assert_called_once_with("/tmp/trial", {}, "haiku")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            annotate_trial_llm("/tmp/trial", {}, backend="openai")

    def test_model_forwarded(self) -> None:
        with mock.patch.object(
            llm_annotator,
            "annotate_trial_claude_code",
            return_value=[],
        ) as m:
            annotate_trial_llm("/tmp/trial", {}, model="opus", backend="claude-code")
            m.assert_called_once_with("/tmp/trial", {}, "opus")


# ---------------------------------------------------------------------------
# annotate_trial_api ImportError
# ---------------------------------------------------------------------------


class TestAnnotateTrialApiImportError:
    """Test that missing anthropic raises ImportError (not logger.error)."""

    def test_raises_import_error_when_anthropic_missing(self) -> None:
        real_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "anthropic":
                raise ImportError("no anthropic")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="anthropic SDK required"):
                annotate_trial_api("/tmp/trial", {})


# ---------------------------------------------------------------------------
# No Path(__file__).parent for taxonomy
# ---------------------------------------------------------------------------


class TestTaxonomyResolution:
    """Verify taxonomy loading uses _package_data_path, not __file__."""

    def test_no_file_parent_in_taxonomy_yaml(self) -> None:
        source = inspect.getsource(llm_annotator._taxonomy_yaml)
        assert "__file__" not in source

    def test_no_hardcoded_taxonomy_filename(self) -> None:
        """_taxonomy_yaml must NOT hardcode 'taxonomy_v1.yaml' string literal."""
        source = inspect.getsource(llm_annotator._taxonomy_yaml)
        assert "taxonomy_v1.yaml" not in source


# ---------------------------------------------------------------------------
# _parse_claude_response contract tests
# ---------------------------------------------------------------------------


class TestParseClaudeResponse:
    """Contract tests for _parse_claude_response extracted helper."""

    def _load_fixture(self, name: str) -> dict:
        path = FIXTURES_DIR / name
        with open(path) as f:
            return json.load(f)

    def test_structured_output(self) -> None:
        envelope = self._load_fixture("claude_envelope_structured_output.json")
        result = llm_annotator._parse_claude_response(envelope)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "retrieval_failure"
        assert result[1]["name"] == "query_churn"

    def test_raw_fallback(self) -> None:
        envelope = self._load_fixture("claude_envelope_raw_fallback.json")
        result = llm_annotator._parse_claude_response(envelope)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "stale_context"

    def test_is_error_returns_none(self) -> None:
        envelope = self._load_fixture("claude_envelope_is_error.json")
        result = llm_annotator._parse_claude_response(envelope)
        assert result is None

    def test_empty_result_returns_empty(self) -> None:
        envelope = self._load_fixture("claude_envelope_empty_result.json")
        result = llm_annotator._parse_claude_response(envelope)
        assert result is None

    def test_code_fences_in_raw(self) -> None:
        envelope = self._load_fixture("claude_envelope_code_fences.json")
        result = llm_annotator._parse_claude_response(envelope)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "decomposition_failure"

    def test_raw_list_format(self) -> None:
        envelope = self._load_fixture("claude_envelope_raw_list.json")
        result = llm_annotator._parse_claude_response(envelope)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_bad_json_returns_none(self) -> None:
        envelope = {
            "is_error": False,
            "structured_output": None,
            "result": "this is not json at all",
        }
        result = llm_annotator._parse_claude_response(envelope)
        assert result is None

    def test_non_list_categories_returns_none(self) -> None:
        envelope = {
            "is_error": False,
            "structured_output": {"categories": "not a list"},
            "result": "",
        }
        result = llm_annotator._parse_claude_response(envelope)
        assert result is None


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestReadText:
    """Tests for _read_text helper."""

    def test_reads_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        assert _read_text(f) == "hello world"

    def test_truncates_at_max_chars(self, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        f.write_text("x" * 100)
        result = _read_text(f, max_chars=10)
        assert result is not None
        assert result.endswith("[truncated]")
        assert len(result.split("\n")[0]) == 10

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert _read_text(tmp_path / "nope.txt") is None

    def test_zero_max_chars_means_no_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("x" * 1000)
        result = _read_text(f, max_chars=0)
        assert result is not None
        assert len(result) == 1000


class TestLoadJson:
    """Tests for _load_json helper."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = _load_json(f)
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json")
        assert _load_json(f) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert _load_json(tmp_path / "nope.json") is None


class TestResolveModelAlias:
    """Tests for _resolve_model_alias."""

    def test_known_alias(self) -> None:
        assert _resolve_model_alias("haiku") == "haiku"
        assert _resolve_model_alias("sonnet") == "sonnet"
        assert _resolve_model_alias("opus") == "opus"

    def test_unknown_passthrough(self) -> None:
        assert _resolve_model_alias("claude-3-custom") == "claude-3-custom"


class TestResolveModelApi:
    """Tests for _resolve_model_api."""

    def test_known_alias(self) -> None:
        assert _resolve_model_api("haiku") == "claude-haiku-4-5-20251001"
        assert _resolve_model_api("sonnet") == "claude-sonnet-4-6"
        assert _resolve_model_api("opus") == "claude-opus-4-6"

    def test_unknown_passthrough(self) -> None:
        assert _resolve_model_api("claude-3-custom") == "claude-3-custom"


# ---------------------------------------------------------------------------
# Taxonomy cache reset fixture test
# ---------------------------------------------------------------------------


class TestTaxonomyCacheReset:
    """Verify taxonomy cache reset fixture works."""

    def test_cache_resets(self, reset_taxonomy_cache: None) -> None:
        from agent_diagnostics import taxonomy as _tax

        assert _tax._cached_taxonomy is None
        assert _tax._cached_path is None
