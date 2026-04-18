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
    StageResult,
    format_summary,
    is_stale,
    load_pipeline,
    run_pipeline,
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
        path = _write_pipeline(tmp_path, "title = \"empty\"\n")
        with pytest.raises(PipelineError, match=r"\[\[stage\]\]"):
            load_pipeline(path)


class TestIsStale:
    def _make_files(self, root: Path, inputs: list[str], outputs: list[str]) -> None:
        for p in inputs + outputs:
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_text("x")

    def test_missing_output_is_stale(self, tmp_path: Path) -> None:
        stage = Stage(
            name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo"
        )
        (tmp_path / "in.txt").write_text("x")
        stale, reason = is_stale(stage, tmp_path)
        assert stale is True
        assert "missing" in reason

    def test_input_newer_than_output_is_stale(self, tmp_path: Path) -> None:
        stage = Stage(
            name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo"
        )
        (tmp_path / "out.txt").write_text("x")
        time.sleep(0.01)
        (tmp_path / "in.txt").write_text("x")
        # Force mtime: input strictly newer
        os.utime(tmp_path / "in.txt", (time.time() + 10, time.time() + 10))
        stale, reason = is_stale(stage, tmp_path)
        assert stale is True
        assert "newer" in reason

    def test_output_newer_than_input_is_fresh(self, tmp_path: Path) -> None:
        stage = Stage(
            name="s", inputs=("in.txt",), outputs=("out.txt",), command="echo"
        )
        (tmp_path / "in.txt").write_text("x")
        time.sleep(0.01)
        (tmp_path / "out.txt").write_text("y")
        os.utime(tmp_path / "out.txt", (time.time() + 10, time.time() + 10))
        stale, reason = is_stale(stage, tmp_path)
        assert stale is False

    def test_directory_input_uses_recursive_mtime(self, tmp_path: Path) -> None:
        stage = Stage(
            name="s", inputs=("in/",), outputs=("out.txt",), command="echo"
        )
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.json").write_text("1")
        (tmp_path / "out.txt").write_text("2")
        os.utime(tmp_path / "out.txt", (time.time() + 10, time.time() + 10))
        assert is_stale(stage, tmp_path)[0] is False

        # Touch a file inside the input directory → stage becomes stale
        time.sleep(0.01)
        later = time.time() + 20
        os.utime(tmp_path / "in" / "a.json", (later, later))
        assert is_stale(stage, tmp_path)[0] is True


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

    def test_fresh_stages_skip(self, tmp_path: Path) -> None:
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
        later = time.time() + 10
        os.utime(tmp_path / "out.txt", (later, later))

        calls: list[str] = []

        def runner(cmd: str, cwd: Path) -> int:
            calls.append(cmd)
            return 0

        results = run_pipeline(path, tmp_path, runner=runner)
        assert [r.status for r in results] == ["skipped"]
        assert calls == []

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
        (tmp_path / "out.txt").write_text("y")
        later = time.time() + 10
        os.utime(tmp_path / "out.txt", (later, later))

        results = run_pipeline(path, tmp_path, runner=lambda c, w: 0)
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
            later = time.time() + 10
            os.utime(p, (later, later))

        start = time.perf_counter()
        results = run_pipeline(path, tmp_path, runner=lambda c, w: 0)
        elapsed = time.perf_counter() - start

        assert all(r.status == "skipped" for r in results)
        assert elapsed < 1.0, f"up-to-date pipeline took {elapsed:.3f}s"
