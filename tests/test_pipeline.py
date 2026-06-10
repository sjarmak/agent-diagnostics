"""Tests for the pipeline DAG runner (PRD NH-2)."""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from agent_diagnostics.pipeline import (
    PipelineError,
    Stage,
    format_summary,
    is_stale,
    load_pipeline,
    load_state,
    run_pipeline,
    stage_input_fingerprint,
)


def _write_pipeline(root: Path, body: str) -> Path:
    path = root / "pipeline.toml"
    path.write_text(textwrap.dedent(body).lstrip())
    return path


class TestLoadPipeline:
    def test_parses_minimal(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "extract"
            inputs = ["runs/"]
            outputs = ["data/signals.jsonl"]
            command = "echo hi"
            """,
        )
        stages = load_pipeline(path)
        assert len(stages) == 1
        assert stages[0].name == "extract"
        assert stages[0].inputs == ("runs/",)
        assert stages[0].outputs == ("data/signals.jsonl",)
        assert stages[0].command == "echo hi"

    def test_multiple_stages(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "extract"
            inputs = ["runs/"]
            outputs = ["data/signals.jsonl"]
            command = "echo a"

            [[stage]]
            name = "annotate"
            inputs = ["data/signals.jsonl"]
            outputs = ["data/annotations.jsonl"]
            command = "echo b"
            """,
        )
        stages = load_pipeline(path)
        assert [s.name for s in stages] == ["extract", "annotate"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PipelineError, match="not found"):
            load_pipeline(tmp_path / "nope.toml")

    def test_missing_required_field(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "extract"
            inputs = ["runs/"]
            command = "echo a"
            """,
        )
        with pytest.raises(PipelineError, match="outputs"):
            load_pipeline(path)

    def test_duplicate_stage_name(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "extract"
            inputs = ["a"]
            outputs = ["b"]
            command = "echo 1"

            [[stage]]
            name = "extract"
            inputs = ["b"]
            outputs = ["c"]
            command = "echo 2"
            """,
        )
        with pytest.raises(PipelineError, match="duplicate"):
            load_pipeline(path)

    def test_no_stages(self, tmp_path: Path) -> None:
        path = _write_pipeline(tmp_path, 'title = "empty"\n')
        with pytest.raises(PipelineError, match=r"\[\[stage\]\]"):
            load_pipeline(path)


class TestIsStale:
    def _recorded_state(self, stage: Stage, root: Path) -> dict:
        """Build a state dict as if *stage* had just run on current inputs."""
        return {stage.name: {"inputs_fingerprint": stage_input_fingerprint(stage, root)}}

    def test_missing_output_is_stale(self, tmp_path: Path) -> None:
        stage = Stage(name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo")
        (tmp_path / "in.txt").write_text("x")
        stale, reason = is_stale(stage, tmp_path)
        assert stale is True
        assert "missing" in reason

    def test_no_recorded_fingerprint_is_stale(self, tmp_path: Path) -> None:
        stage = Stage(name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo")
        (tmp_path / "in.txt").write_text("x")
        (tmp_path / "out.txt").write_text("y")
        stale, reason = is_stale(stage, tmp_path, state={})
        assert stale is True
        assert "no recorded" in reason

    def test_matching_fingerprint_is_fresh(self, tmp_path: Path) -> None:
        stage = Stage(name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo")
        (tmp_path / "in.txt").write_text("x")
        (tmp_path / "out.txt").write_text("y")
        state = self._recorded_state(stage, tmp_path)
        stale, reason = is_stale(stage, tmp_path, state=state)
        assert stale is False
        assert reason == "up-to-date"

    def test_changed_content_with_identical_mtime_is_stale(self, tmp_path: Path) -> None:
        """The regression mtime checks miss: same size, same mtime, new bytes
        (git checkout / cp -p / CI cache restore)."""
        stage = Stage(name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo")
        in_file = tmp_path / "in.txt"
        in_file.write_text("x")
        (tmp_path / "out.txt").write_text("y")
        state = self._recorded_state(stage, tmp_path)
        frozen = in_file.stat().st_mtime

        in_file.write_text("z")  # same size, different content
        os.utime(in_file, (frozen, frozen))  # restore the old mtime

        stale, reason = is_stale(stage, tmp_path, state=state)
        assert stale is True
        assert "content changed" in reason

    def test_mtime_churn_without_content_change_stays_fresh(self, tmp_path: Path) -> None:
        stage = Stage(name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo")
        (tmp_path / "in.txt").write_text("x")
        (tmp_path / "out.txt").write_text("y")
        state = self._recorded_state(stage, tmp_path)

        later = time.time() + 100
        os.utime(tmp_path / "in.txt", (later, later))

        assert is_stale(stage, tmp_path, state=state)[0] is False

    def test_directory_input_fingerprints_recursively(self, tmp_path: Path) -> None:
        stage = Stage(name="s", inputs=("in/",), outputs=("out.txt",), command="echo")
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.json").write_text("1")
        (tmp_path / "out.txt").write_text("2")
        state = self._recorded_state(stage, tmp_path)
        assert is_stale(stage, tmp_path, state=state)[0] is False

        # Edit a file inside the input directory → stage becomes stale
        (tmp_path / "in" / "a.json").write_text("2")
        assert is_stale(stage, tmp_path, state=state)[0] is True

        # Adding a new file also changes the fingerprint
        (tmp_path / "in" / "a.json").write_text("1")
        assert is_stale(stage, tmp_path, state=state)[0] is False
        (tmp_path / "in" / "b.json").write_text("3")
        assert is_stale(stage, tmp_path, state=state)[0] is True

    def test_no_inputs_present_is_fresh(self, tmp_path: Path) -> None:
        stage = Stage(name="s", inputs=("absent.txt",), outputs=("out.txt",), command="echo")
        (tmp_path / "out.txt").write_text("y")
        stale, reason = is_stale(stage, tmp_path)
        assert stale is False
        assert "no inputs" in reason


class TestRunPipeline:
    def test_all_stale_stages_run(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["out.txt"]
            command = "echo a"
            """,
        )
        (tmp_path / "in.txt").write_text("x")

        commands_run: list[str] = []

        def runner(cmd: str, cwd: Path) -> int:
            commands_run.append(cmd)
            (cwd / "out.txt").write_text("y")
            return 0

        results = run_pipeline(path, tmp_path, runner=runner)
        assert [r.status for r in results] == ["ran"]
        assert commands_run == ["echo a"]

    def test_fresh_stages_skip_on_second_run(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["out.txt"]
            command = "echo a"
            """,
        )
        (tmp_path / "in.txt").write_text("x")

        calls: list[str] = []

        def runner(cmd: str, cwd: Path) -> int:
            calls.append(cmd)
            (cwd / "out.txt").write_text("y")
            return 0

        first = run_pipeline(path, tmp_path, runner=runner)
        assert [r.status for r in first] == ["ran"]

        second = run_pipeline(path, tmp_path, runner=runner)
        assert [r.status for r in second] == ["skipped"]
        assert calls == ["echo a"]

    def test_changed_input_reruns_after_skip(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["out.txt"]
            command = "echo a"
            """,
        )
        (tmp_path / "in.txt").write_text("x")

        def runner(cmd: str, cwd: Path) -> int:
            (cwd / "out.txt").write_text("y")
            return 0

        run_pipeline(path, tmp_path, runner=runner)
        assert [r.status for r in run_pipeline(path, tmp_path, runner=runner)] == ["skipped"]

        (tmp_path / "in.txt").write_text("x2")
        results = run_pipeline(path, tmp_path, runner=runner)
        assert [r.status for r in results] == ["ran"]
        assert results[0].reason == "input content changed"

    def test_state_file_persisted(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["out.txt"]
            command = "echo a"
            """,
        )
        (tmp_path / "in.txt").write_text("x")

        def runner(cmd: str, cwd: Path) -> int:
            (cwd / "out.txt").write_text("y")
            return 0

        run_pipeline(path, tmp_path, runner=runner)

        state = load_state(tmp_path)
        assert "a" in state
        assert state["a"]["inputs_fingerprint"] == stage_input_fingerprint(
            Stage(name="a", inputs=("in.txt",), outputs=("out.txt",), command="echo a"),
            tmp_path,
        )

    def test_failed_stage_records_no_fingerprint(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["out.txt"]
            command = "echo a"
            """,
        )
        (tmp_path / "in.txt").write_text("x")
        (tmp_path / "out.txt").write_text("y")

        results = run_pipeline(path, tmp_path, runner=lambda c, w: 1)
        assert [r.status for r in results] == ["failed"]
        assert load_state(tmp_path) == {}

    def test_malformed_state_file_treated_as_empty(self, tmp_path: Path) -> None:
        (tmp_path / ".pipeline-state.json").write_text("{not json")
        assert load_state(tmp_path) == {}

    def test_all_up_to_date_message(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["out.txt"]
            command = "echo a"
            """,
        )
        (tmp_path / "in.txt").write_text("x")

        def runner(cmd: str, cwd: Path) -> int:
            (cwd / "out.txt").write_text("y")
            return 0

        run_pipeline(path, tmp_path, runner=runner)
        results = run_pipeline(path, tmp_path, runner=runner)
        assert "all stages up to date" in format_summary(results)

    def test_failure_aborts_downstream(self, tmp_path: Path) -> None:
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "a"
            inputs = ["in.txt"]
            outputs = ["mid.txt"]
            command = "cmd-a"

            [[stage]]
            name = "b"
            inputs = ["mid.txt"]
            outputs = ["out.txt"]
            command = "cmd-b"
            """,
        )
        (tmp_path / "in.txt").write_text("x")

        calls: list[str] = []

        def failing_runner(cmd: str, cwd: Path) -> int:
            calls.append(cmd)
            return 1

        results = run_pipeline(path, tmp_path, runner=failing_runner)
        assert [r.status for r in results] == ["failed"]
        assert calls == ["cmd-a"]

    def test_acceptance_fresh_pipeline_under_1s(self, tmp_path: Path) -> None:
        """NH-2 acceptance: a fully up-to-date pipeline completes in <1s."""
        path = _write_pipeline(
            tmp_path,
            """
            [[stage]]
            name = "extract"
            inputs = ["runs/"]
            outputs = ["data/signals.jsonl"]
            command = "echo extract"

            [[stage]]
            name = "annotate"
            inputs = ["data/signals.jsonl"]
            outputs = ["data/annotations.jsonl"]
            command = "echo annotate"

            [[stage]]
            name = "report"
            inputs = ["data/annotations.jsonl"]
            outputs = ["data/report/report.md"]
            command = "echo report"
            """,
        )
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "r.json").write_text("1")
        for rel in ("data/signals.jsonl", "data/annotations.jsonl", "data/report/report.md"):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("ok")

        # First run records fingerprints; the timed second run must skip all.
        run_pipeline(path, tmp_path, runner=lambda c, w: 0)

        start = time.perf_counter()
        results = run_pipeline(path, tmp_path, runner=lambda c, w: 0)
        elapsed = time.perf_counter() - start

        assert all(r.status == "skipped" for r in results)
        assert elapsed < 1.0, f"up-to-date pipeline took {elapsed:.3f}s"
