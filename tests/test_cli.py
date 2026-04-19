"""Tests for the agent-observatory CLI entrypoint."""

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Existing smoke tests
# ---------------------------------------------------------------------------


def test_main_importable():
    """from agent_diagnostics.cli import main succeeds."""
    from agent_diagnostics.cli import main

    assert callable(main)


def test_help_output_includes_all_subcommands():
    """python -m agent_diagnostics --help lists all 8 subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_diagnostics", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    help_text = result.stdout

    expected_subcommands = [
        "extract",
        "annotate",
        "report",
        "llm-annotate",
        "train",
        "predict",
        "ensemble",
        "ingest",
        "validate",
        "query",
        "export",
        "manifest",
        "db",
    ]
    for sub in expected_subcommands:
        assert sub in help_text, f"Subcommand '{sub}' not found in --help output"


def test_no_judge_flag_in_help():
    """--judge flag must not appear in llm-annotate help."""
    result = subprocess.run(
        [sys.executable, "-m", "agent_diagnostics", "llm-annotate", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert (
        "--judge" not in result.stdout
    ), "--judge flag should not exist on llm-annotate"


def test_cli_imports_use_agent_diagnostics():
    """All imports in cli.py use agent_diagnostics, not observatory."""
    cli_path = (
        Path(__file__).resolve().parent.parent / "src" / "agent_diagnostics" / "cli.py"
    )
    content = cli_path.read_text()

    # Should not have bare 'from observatory.' imports
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("from observatory.") or stripped.startswith(
            "import observatory."
        ):
            pytest.fail(f"Found bare observatory import: {stripped}")

    # Should have agent_diagnostics imports
    assert "from agent_diagnostics." in content


def test_dunder_main_imports_from_agent_diagnostics():
    """__main__.py imports from agent_diagnostics.cli."""
    main_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "agent_diagnostics"
        / "__main__.py"
    )
    content = main_path.read_text()
    assert "from agent_diagnostics.cli import main" in content


def test_pyproject_has_scripts_entry():
    """pyproject.toml has [project.scripts] observatory entry."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    assert "[project.scripts]" in content
    assert 'observatory = "agent_diagnostics.cli:main"' in content


# ---------------------------------------------------------------------------
# cmd_extract tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: create a minimal valid trial directory
# ---------------------------------------------------------------------------


def _make_trial(parent: Path, name: str = "trial1", **result_overrides) -> Path:
    """Create a trial directory with a valid result.json."""
    trial_dir = parent / name
    trial_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_name": "test_task",
        "agent_info": {"name": "test-agent"},
        "verifier_result": {"rewards": {"reward": 1.0}},
        **result_overrides,
    }
    (trial_dir / "result.json").write_text(json.dumps(result))
    return trial_dir


# ---------------------------------------------------------------------------
# cmd_ingest tests
# ---------------------------------------------------------------------------


class TestCmdIngest:
    """Tests for cmd_ingest."""

    def test_missing_runs_dir_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_ingest

        args = argparse.Namespace(
            runs_dir=str(tmp_path / "nonexistent"),
            output=str(tmp_path / "out.jsonl"),
            manifest=None,
            state=None,
        )
        with pytest.raises(SystemExit):
            cmd_ingest(args)

    def test_basic_ingest(self, tmp_path):
        """Ingest writes JSONL output with one trial per line."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        _make_trial(runs_dir, "trial1")
        _make_trial(runs_dir, "trial2")

        output = tmp_path / "out.jsonl"
        args = argparse.Namespace(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=None,
            state=None,
        )
        cmd_ingest(args)

        assert output.exists()
        lines = [json.loads(line) for line in output.read_text().strip().splitlines()]
        assert len(lines) == 2
        # Each line should have trial_path set
        for line in lines:
            assert "trial_path" in line
            assert "task_id" in line

    def test_invalid_trial_filtered(self, tmp_path):
        """Harness summaries without agent_info are skipped."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        # Valid trial
        _make_trial(runs_dir, "valid_trial")
        # Invalid trial (no agent_info = harness summary)
        invalid_dir = runs_dir / "summary"
        invalid_dir.mkdir(parents=True)
        (invalid_dir / "result.json").write_text(json.dumps({"summary": True}))

        output = tmp_path / "out.jsonl"
        args = argparse.Namespace(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=None,
            state=None,
        )
        cmd_ingest(args)

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_excluded_path_filtered(self, tmp_path):
        """Trials in excluded directories are skipped."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        _make_trial(runs_dir, "good_trial")
        _make_trial(runs_dir / "__archived_invalid", "bad_trial")

        output = tmp_path / "out.jsonl"
        args = argparse.Namespace(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=None,
            state=None,
        )
        cmd_ingest(args)

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_manifest_flag(self, tmp_path):
        """--manifest loads suite_mapping and passes it to extract_signals."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        _make_trial(runs_dir, "trial1")

        manifest = tmp_path / "MANIFEST.json"
        manifest.write_text(json.dumps({"test_task": "my-benchmark"}))

        output = tmp_path / "out.jsonl"
        args = argparse.Namespace(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=str(manifest),
            state=None,
        )
        cmd_ingest(args)

        lines = [json.loads(line) for line in output.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["benchmark"] == "my-benchmark"

    def test_manifest_missing_exits(self, tmp_path):
        """--manifest with nonexistent file exits with error."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        args = argparse.Namespace(
            runs_dir=str(runs_dir),
            output=str(tmp_path / "out.jsonl"),
            manifest=str(tmp_path / "missing_manifest.json"),
            state=None,
        )
        with pytest.raises(SystemExit):
            cmd_ingest(args)

    def test_state_file_creation(self, tmp_path):
        """--state creates a state file with mtime/size entries."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        trial = _make_trial(runs_dir, "trial1")

        state_file = tmp_path / "state.json"
        output = tmp_path / "out.jsonl"
        args = argparse.Namespace(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=None,
            state=str(state_file),
        )
        cmd_ingest(args)

        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert str(trial) in state
        entry = state[str(trial)]
        assert "mtime" in entry
        assert "size" in entry

    def test_incremental_skip(self, tmp_path):
        """Running ingest twice with no changes skips unchanged trials."""
        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        _make_trial(runs_dir, "trial1")

        state_file = tmp_path / "state.json"
        output = tmp_path / "out.jsonl"

        base_args = dict(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=None,
            state=str(state_file),
        )

        # First run: extracts the trial
        cmd_ingest(argparse.Namespace(**base_args))
        lines_first = output.read_text().strip().splitlines()
        assert len(lines_first) == 1

        # Second run: trial unchanged, should be skipped
        cmd_ingest(argparse.Namespace(**base_args))
        lines_second = output.read_text().strip().splitlines()
        # Output should be empty (0 new trials extracted)
        assert len(lines_second) == 0

    def test_incremental_reextracts_changed(self, tmp_path):
        """Modified trial is re-extracted on second run."""
        import time

        from agent_diagnostics.cli import cmd_ingest

        runs_dir = tmp_path / "runs"
        trial = _make_trial(runs_dir, "trial1")

        state_file = tmp_path / "state.json"
        output = tmp_path / "out.jsonl"

        base_args = dict(
            runs_dir=str(runs_dir),
            output=str(output),
            manifest=None,
            state=str(state_file),
        )

        # First run
        cmd_ingest(argparse.Namespace(**base_args))
        assert len(output.read_text().strip().splitlines()) == 1

        # Modify the result.json (change content to change size, touch for mtime)
        time.sleep(0.05)  # ensure mtime differs
        result_path = trial / "result.json"
        data = json.loads(result_path.read_text())
        data["extra_field"] = "changed"
        result_path.write_text(json.dumps(data))

        # Second run: should re-extract
        cmd_ingest(argparse.Namespace(**base_args))
        assert len(output.read_text().strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# cmd_extract tests
# ---------------------------------------------------------------------------


class TestCmdExtract:
    """Tests for cmd_extract."""

    def test_missing_runs_dir_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_extract

        args = argparse.Namespace(
            runs_dir=str(tmp_path / "nonexistent"),
            output=str(tmp_path / "out.json"),
            cache_dir=None,
        )
        with pytest.raises(SystemExit):
            cmd_extract(args)

    @patch("agent_diagnostics.signals.extract_all")
    def test_success(self, mock_extract, tmp_path):
        from agent_diagnostics.cli import cmd_extract

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_extract.return_value = [{"trial_id": "t1"}, {"trial_id": "t2"}]

        output = tmp_path / "signals.json"
        args = argparse.Namespace(
            runs_dir=str(runs_dir), output=str(output), cache_dir=None
        )
        cmd_extract(args)

        mock_extract.assert_called_once_with(runs_dir, cache=None)
        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data) == 2


# ---------------------------------------------------------------------------
# cmd_annotate tests
# ---------------------------------------------------------------------------


class TestCmdAnnotate:
    """Tests for cmd_annotate."""

    def test_missing_signals_file_exits(self, tmp_path):
        import agent_diagnostics.annotator as _ann_mod
        from agent_diagnostics.cli import cmd_annotate

        _ann_mod.annotate_all = MagicMock()
        try:
            args = argparse.Namespace(
                signals=str(tmp_path / "missing.json"),
                output=str(tmp_path / "out.json"),
            )
            with pytest.raises(SystemExit):
                cmd_annotate(args)
        finally:
            if hasattr(_ann_mod, "annotate_all"):
                del _ann_mod.annotate_all

    @patch("agent_diagnostics.annotator.annotate_trial")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_success(self, mock_taxonomy, mock_annotate_trial, tmp_path):
        from agent_diagnostics.cli import cmd_annotate
        from agent_diagnostics.types import CategoryAssignment

        mock_taxonomy.return_value = {"version": "3.0"}
        mock_annotate_trial.return_value = [
            CategoryAssignment(name="cat1", confidence=0.9, evidence="test")
        ]

        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"task_id": "t1"}]))

        output = tmp_path / "annotations.json"
        args = argparse.Namespace(signals=str(signals_file), output=str(output))
        cmd_annotate(args)

        mock_annotate_trial.assert_called_once()
        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["categories"][0]["name"] == "cat1"


# ---------------------------------------------------------------------------
# cmd_report tests
# ---------------------------------------------------------------------------


class TestCmdReport:
    """Tests for cmd_report."""

    def test_missing_annotations_file_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_report

        args = argparse.Namespace(
            annotations=str(tmp_path / "missing.json"),
            output=str(tmp_path / "report"),
        )
        with pytest.raises(SystemExit):
            cmd_report(args)

    @patch("agent_diagnostics.report.generate_report")
    def test_success(self, mock_report, tmp_path):
        from agent_diagnostics.cli import cmd_report

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps({"annotations": []}))

        out_dir = tmp_path / "report"
        mock_report.return_value = (out_dir / "report.md", out_dir / "report.json")

        args = argparse.Namespace(annotations=str(ann_file), output=str(out_dir))
        cmd_report(args)

        mock_report.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_llm_annotate tests
# ---------------------------------------------------------------------------


class TestCmdLlmAnnotate:
    """Tests for cmd_llm_annotate."""

    def test_missing_signals_file_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_llm_annotate

        args = argparse.Namespace(
            signals=str(tmp_path / "missing.json"),
            output=str(tmp_path / "out.json"),
            sample_size=5,
            model="haiku",
            backend="claude-code",
        )
        with pytest.raises(SystemExit):
            cmd_llm_annotate(args)

    def test_no_trajectories_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_llm_annotate

        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"trial_path": str(tmp_path / "nope")}]))

        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(tmp_path / "out.json"),
            sample_size=5,
            model="haiku",
            backend="claude-code",
        )
        with pytest.raises(SystemExit):
            cmd_llm_annotate(args)

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_success_with_mock(self, mock_taxonomy, mock_annotate_batch, tmp_path):
        from agent_diagnostics.cli import cmd_llm_annotate

        # Create a trial directory with trajectory file
        trial_dir = tmp_path / "trial1" / "agent"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trajectory.json").write_text("[]")

        signals = [
            {
                "trial_path": str(tmp_path / "trial1"),
                "task_id": "task1",
                "reward": 1.0,
                "passed": True,
            }
        ]
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        from agent_diagnostics.types import AnnotationOk

        mock_taxonomy.return_value = {"version": "1.0", "categories": []}
        mock_annotate_batch.return_value = [
            AnnotationOk(
                categories=(
                    {"name": "cat1", "confidence": 0.9, "evidence": "test"},
                )
            )
        ]

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=5,
            model="haiku",
            backend="claude-code",
        )
        cmd_llm_annotate(args)

        mock_annotate_batch.assert_called_once()
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["schema_version"] == "observatory-annotation-v2"
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["task_id"] == "task1"
        assert data["annotations"][0]["annotation_result_status"] == "ok"
        assert data["annotation_summary"] == {
            "ok": 1,
            "no_categories": 0,
            "error": 0,
            "total": 1,
        }


# ---------------------------------------------------------------------------
# cmd_train tests
# ---------------------------------------------------------------------------


class TestCmdTrain:
    """Tests for cmd_train."""

    @patch("agent_diagnostics.classifier.save_model")
    @patch("agent_diagnostics.classifier.train")
    def test_success(self, mock_train, mock_save, tmp_path):
        from agent_diagnostics.cli import cmd_train

        mock_train.return_value = {
            "classifiers": {
                "cat1": {
                    "positive_count": 5,
                    "total_count": 10,
                    "train_accuracy": 0.90,
                }
            },
            "skipped_categories": [],
            "training_samples": 10,
            "min_positive": 3,
        }

        output = tmp_path / "model.json"
        args = argparse.Namespace(
            labels=str(tmp_path / "labels.json"),
            signals=str(tmp_path / "signals.json"),
            output=str(output),
            min_positive=3,
            lr=0.1,
            epochs=300,
            eval=False,
        )
        cmd_train(args)

        mock_train.assert_called_once_with(
            llm_file=args.labels,
            signals_file=args.signals,
            min_positive=3,
            lr=0.1,
            epochs=300,
        )
        mock_save.assert_called_once()

    @patch("agent_diagnostics.classifier.format_eval_markdown")
    @patch("agent_diagnostics.classifier.evaluate")
    @patch("agent_diagnostics.classifier.save_model")
    @patch("agent_diagnostics.classifier.train")
    def test_with_eval(self, mock_train, mock_save, mock_evaluate, mock_fmt, tmp_path):
        from agent_diagnostics.cli import cmd_train

        mock_train.return_value = {
            "classifiers": {},
            "skipped_categories": ["cat1"],
            "training_samples": 5,
            "min_positive": 3,
        }
        mock_evaluate.return_value = {"accuracy": 0.9}
        mock_fmt.return_value = "## Eval\nAccuracy: 0.9"

        output = tmp_path / "model.json"
        args = argparse.Namespace(
            labels=str(tmp_path / "labels.json"),
            signals=str(tmp_path / "signals.json"),
            output=str(output),
            min_positive=3,
            lr=0.1,
            epochs=300,
            eval=True,
        )
        cmd_train(args)

        mock_evaluate.assert_called_once()
        mock_fmt.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_predict tests
# ---------------------------------------------------------------------------


class TestCmdPredict:
    """Tests for cmd_predict."""

    @patch("agent_diagnostics.classifier.predict_all")
    @patch("agent_diagnostics.classifier.load_model")
    def test_success(self, mock_load, mock_predict, tmp_path):
        from agent_diagnostics.cli import cmd_predict

        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"trial_id": "t1"}]))

        mock_load.return_value = {"classifiers": {}}
        mock_predict.return_value = {
            "annotations": [{"task_id": "t1", "categories": [{"name": "cat1"}]}]
        }

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            model=str(tmp_path / "model.json"),
            signals=str(signals_file),
            output=str(output),
            threshold=0.5,
        )
        cmd_predict(args)

        mock_load.assert_called_once()
        mock_predict.assert_called_once()
        assert output.exists()


# ---------------------------------------------------------------------------
# cmd_ensemble tests
# ---------------------------------------------------------------------------


class TestCmdEnsemble:
    """Tests for cmd_ensemble."""

    @patch("agent_diagnostics.ensemble.ensemble_all")
    @patch("agent_diagnostics.classifier.load_model")
    def test_success(self, mock_load, mock_ensemble, tmp_path):
        from agent_diagnostics.cli import cmd_ensemble

        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"trial_id": "t1"}]))

        mock_load.return_value = {"classifiers": {}}
        mock_ensemble.return_value = {
            "annotations": [{"task_id": "t1", "categories": []}],
            "tier_counts": {"heuristic": 1, "classifier": 0},
        }

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            model=str(tmp_path / "model.json"),
            output=str(output),
            threshold=0.5,
            min_f1=0.7,
        )
        cmd_ensemble(args)

        mock_load.assert_called_once()
        mock_ensemble.assert_called_once()
        assert output.exists()


# ---------------------------------------------------------------------------
# cmd_validate tests
# ---------------------------------------------------------------------------


class TestCmdValidate:
    """Tests for cmd_validate."""

    def test_missing_annotations_file_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_validate

        args = argparse.Namespace(annotations=str(tmp_path / "missing.json"))
        with pytest.raises(SystemExit):
            cmd_validate(args)

    def test_invalid_json_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_validate

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!!")

        args = argparse.Namespace(annotations=str(bad_file))
        with pytest.raises(SystemExit):
            cmd_validate(args)

    @patch("jsonschema.validate")
    @patch("agent_diagnostics.taxonomy.valid_category_names")
    def test_valid_file_exits_zero(self, mock_names, mock_validate, tmp_path):
        from agent_diagnostics.cli import cmd_validate

        ann_file = tmp_path / "ann.json"
        ann_file.write_text(
            json.dumps(
                {
                    "schema_version": "observatory-annotation-v1",
                    "annotations": [
                        {"task_id": "t1", "categories": [{"name": "cat1"}]}
                    ],
                }
            )
        )

        mock_validate.return_value = None
        mock_names.return_value = {"cat1", "cat2"}

        args = argparse.Namespace(annotations=str(ann_file))
        # cmd_validate calls sys.exit(0) on success
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)
        assert exc_info.value.code == 0

    @patch("agent_diagnostics.taxonomy.valid_category_names")
    def test_schema_error_exits(self, mock_names, tmp_path):
        import jsonschema as _js

        from agent_diagnostics.cli import cmd_validate

        ann_file = tmp_path / "ann.json"
        ann_file.write_text(json.dumps({"annotations": []}))

        mock_names.return_value = set()

        args = argparse.Namespace(annotations=str(ann_file))
        with patch(
            "jsonschema.validate",
            side_effect=_js.ValidationError("bad"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
        assert exc_info.value.code == 1

    @patch("jsonschema.validate")
    @patch("agent_diagnostics.taxonomy.valid_category_names")
    def test_unknown_category_exits(self, mock_names, mock_validate, tmp_path):
        from agent_diagnostics.cli import cmd_validate

        ann_file = tmp_path / "ann.json"
        ann_file.write_text(
            json.dumps(
                {
                    "annotations": [
                        {"task_id": "t1", "categories": [{"name": "unknown_cat"}]}
                    ],
                }
            )
        )

        mock_validate.return_value = None
        mock_names.return_value = {"cat1", "cat2"}

        args = argparse.Namespace(annotations=str(ann_file))
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() argument parsing."""

    def test_no_command_exits_zero(self):
        from agent_diagnostics.cli import main

        with patch("sys.argv", ["observatory"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# JSONL format support in CLI commands
# ---------------------------------------------------------------------------


class TestCmdExtractJsonl:
    """Tests for cmd_extract with .jsonl output."""

    @patch("agent_diagnostics.signals.extract_all")
    def test_jsonl_output(self, mock_extract, tmp_path):
        from agent_diagnostics.cli import cmd_extract

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_extract.return_value = [{"trial_id": "t1"}, {"trial_id": "t2"}]

        output = tmp_path / "signals.jsonl"
        args = argparse.Namespace(runs_dir=str(runs_dir), output=str(output))
        cmd_extract(args)

        assert output.exists()
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"trial_id": "t1"}
        assert json.loads(lines[1]) == {"trial_id": "t2"}

        # Check meta sidecar
        meta_path = output.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "schema_version" in meta
        assert "taxonomy_version" in meta
        assert "generated_at" in meta

    @patch("agent_diagnostics.signals.extract_all")
    def test_json_output_still_works(self, mock_extract, tmp_path):
        from agent_diagnostics.cli import cmd_extract

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_extract.return_value = [{"trial_id": "t1"}]

        output = tmp_path / "signals.json"
        args = argparse.Namespace(runs_dir=str(runs_dir), output=str(output))
        cmd_extract(args)

        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 1


class TestCmdAnnotateJsonl:
    """Tests for cmd_annotate with .jsonl input/output."""

    @patch("agent_diagnostics.annotator.annotate_trial")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_jsonl_input_and_output(self, mock_taxonomy, mock_annotate_trial, tmp_path):
        from agent_diagnostics.cli import cmd_annotate
        from agent_diagnostics.signals import write_jsonl
        from agent_diagnostics.types import CategoryAssignment

        mock_taxonomy.return_value = {"version": "3.0"}
        # Return categories for first call, empty for second
        mock_annotate_trial.side_effect = [
            [CategoryAssignment(name="cat1", confidence=0.9, evidence="test")],
            [],
        ]

        # Write signals as JSONL
        signals_data = [{"task_id": "t1"}, {"task_id": "t2"}]
        signals_file = tmp_path / "signals.jsonl"
        write_jsonl(signals_data, signals_file)

        output = tmp_path / "annotations.jsonl"
        args = argparse.Namespace(signals=str(signals_file), output=str(output))
        cmd_annotate(args)

        assert mock_annotate_trial.call_count == 2
        assert output.exists()

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 2

        meta_path = output.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == "observatory-annotation-v1"
        assert meta["taxonomy_version"] == "3.0"

    @patch("agent_diagnostics.annotator.annotate_trial")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_json_input_jsonl_output(
        self, mock_taxonomy, mock_annotate_trial, tmp_path
    ):
        from agent_diagnostics.cli import cmd_annotate
        from agent_diagnostics.types import CategoryAssignment

        mock_taxonomy.return_value = {"version": "3.0"}
        mock_annotate_trial.return_value = [
            CategoryAssignment(name="cat1", confidence=0.9, evidence="test")
        ]

        # Write signals as JSON (legacy)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"task_id": "t1"}]))

        output = tmp_path / "annotations.jsonl"
        args = argparse.Namespace(signals=str(signals_file), output=str(output))
        cmd_annotate(args)

        assert output.exists()
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# cmd_calibrate: permission contract for composed-reference temp dir + file
# ---------------------------------------------------------------------------


class TestCalibrateGoldenDirPermissions:
    """The --golden-dir path materialises a reference.json in an internal
    temp directory. The directory is always 0o700 and the file is always
    0o600, regardless of caller umask, because the composed document exists
    only for the lifetime of a single compare_annotations() call and must
    never be exposed to other users on shared hosts."""

    def test_golden_dir_reference_file_is_owner_only(self, tmp_path, monkeypatch):
        from agent_diagnostics import calibrate as calibrate_mod
        from agent_diagnostics.cli import cmd_calibrate

        # Minimal golden corpus: one trial with one category. The trial_path
        # key used by _load_annotations comes from the directory name, not
        # from any field inside the JSON.
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir()
        trial_dir = golden_dir / "trial_a"
        trial_dir.mkdir()
        (trial_dir / "expected_annotations.json").write_text(
            json.dumps(
                {
                    "categories": [
                        {"name": "cat_x", "confidence": 0.8, "evidence": "e"}
                    ],
                }
            )
        )

        # Predictor annotation file covering the same trial.
        predictor_path = tmp_path / "predictor.json"
        predictor_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "trial_path": "trial_a",
                            "annotation_result_status": "ok",
                            "categories": [
                                {"name": "cat_x", "confidence": 0.8, "evidence": "e"}
                            ],
                        }
                    ]
                }
            )
        )

        # Spy compare_annotations to capture mode while the temp file exists.
        captured: dict[str, int] = {}
        real_compare = calibrate_mod.compare_annotations

        def spy(predictor, reference):
            # reference is always a Path — cmd_calibrate constructs it as one
            # (cli.py: `reference_path: Path`), so no coercion is needed here.
            captured["file_mode"] = stat.S_IMODE(reference.stat().st_mode)
            captured["dir_mode"] = stat.S_IMODE(reference.parent.stat().st_mode)
            return real_compare(predictor, reference)

        # Patch the source-module attribute. This works here only because
        # cmd_calibrate imports compare_annotations inside the function body
        # (cli.py:750), so the `from ... import` runs AFTER monkeypatch.setattr
        # and picks up the spy. A module-level import would have captured the
        # original function before the patch fires and the spy would be missed.
        monkeypatch.setattr(calibrate_mod, "compare_annotations", spy)

        # Force a permissive process umask so default file-create mode would
        # be 0o644. The test then proves cmd_calibrate tightens to 0o600
        # regardless of caller umask, rather than silently relying on whatever
        # umask the pytest process happened to inherit. umask is process-global
        # and not thread-safe; this project does not use pytest-xdist so the
        # try/finally restore is sufficient.
        old_umask = os.umask(0o022)
        try:
            args = argparse.Namespace(
                predictor=str(predictor_path),
                reference=None,
                golden_dir=str(golden_dir),
                output_dir=str(tmp_path / "out"),
            )
            cmd_calibrate(args)
        finally:
            os.umask(old_umask)

        assert captured, (
            "spy was never invoked — monkeypatch did not intercept "
            "compare_annotations; the file-mode assertion below would have "
            "raised KeyError rather than AssertionError"
        )
        assert captured["file_mode"] == 0o600, (
            f"reference.json mode should be 0o600, got {oct(captured['file_mode'])}"
        )
        assert captured["dir_mode"] == 0o700, (
            f"temp dir mode should be 0o700, got {oct(captured['dir_mode'])}"
        )

    def test_golden_dir_missing_path_exits_cleanly(self, tmp_path, caplog):
        """--golden-dir pointing at a nonexistent path must log and exit(1),
        not raise an unhandled FileNotFoundError from _collect_golden_corpus."""
        import logging

        from agent_diagnostics.cli import cmd_calibrate

        predictor_path = tmp_path / "predictor.json"
        predictor_path.write_text(json.dumps({"annotations": []}))

        args = argparse.Namespace(
            predictor=str(predictor_path),
            reference=None,
            golden_dir=str(tmp_path / "does_not_exist"),
            output_dir=str(tmp_path / "out"),
        )
        with caplog.at_level(logging.ERROR, logger="agent_diagnostics.cli"):
            with pytest.raises(SystemExit) as exc:
                cmd_calibrate(args)

        assert exc.value.code == 1
        assert any(
            "golden" in rec.message.lower() and "not found" in rec.message.lower()
            for rec in caplog.records
        ), f"expected clean error log, got: {[r.message for r in caplog.records]}"


class TestCalibrateSharedTrialsWarning:
    """When predictor.trial_path values don't join against the golden
    corpus dir names, compare_annotations produces shared_trials=0.
    The CLI surfaces this via a WARNING so the empty report doesn't
    silently pass for a join-key mismatch. Still exits 0 — empty output
    is a valid result, just rarely the one the user meant."""

    def test_calibrate_warns_when_shared_trials_is_zero(
        self, tmp_path, caplog
    ):
        import logging

        from agent_diagnostics.cli import cmd_calibrate

        # Golden corpus with one trial named "trial_a".
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir()
        (golden_dir / "trial_a").mkdir()
        (golden_dir / "trial_a" / "expected_annotations.json").write_text(
            json.dumps({"categories": [{"name": "cat_x", "confidence": 0.8}]})
        )

        # Predictor whose trial_path does NOT match (filesystem-style path).
        predictor_path = tmp_path / "predictor.json"
        predictor_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "trial_path": "runs/totally_different_name",
                            "categories": [
                                {"name": "cat_x", "confidence": 0.8, "evidence": "e"}
                            ],
                        }
                    ]
                }
            )
        )

        args = argparse.Namespace(
            predictor=str(predictor_path),
            reference=None,
            golden_dir=str(golden_dir),
            output_dir=str(tmp_path / "out"),
        )
        with caplog.at_level(logging.WARNING, logger="agent_diagnostics.cli"):
            cmd_calibrate(args)

        warning_messages = [
            rec.message for rec in caplog.records if rec.levelno >= logging.WARNING
        ]
        assert any(
            "trial_path" in msg or "shared_trials" in msg or "no overlap" in msg.lower()
            for msg in warning_messages
        ), f"expected WARN about join mismatch, got: {warning_messages}"


# ---------------------------------------------------------------------------
# cmd_annotate: annotation_result_status propagation (0.8.1 bead 13t)
# ---------------------------------------------------------------------------


class TestAnnotateResultStatusField:
    """0.8.0 release notes claim annotation_result_status propagates through
    the v2 schema. Heuristic cmd_annotate was missing the field. This test
    pins the contract for both branches (ok / no_categories)."""

    def test_heuristic_annotate_emits_status_on_every_record(self, tmp_path):
        from agent_diagnostics.cli import cmd_annotate

        # Two trials hand-crafted to hit different branches:
        # - ok: premature_termination fires reliably when effective_length<3
        #   and reward>0, so a short passing trial always gets ≥1 category.
        # - no_categories: an all-zeros signal with no reward and no
        #   trajectory gives no heuristic checker anything to latch onto.
        signals_path = tmp_path / "signals.jsonl"
        signals_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "task_id": "t_ok",
                            "trial_path": "runs/t_ok",
                            "reward": 0.5,
                            "passed": True,
                            "agent_name": "test-agent",
                            "total_turns": 0,
                            "trajectory_length": 0,
                        }
                    ),
                    json.dumps(
                        {
                            "task_id": "t_empty",
                            "trial_path": "runs/t_empty",
                            "agent_name": "test-agent",
                        }
                    ),
                ]
            )
            + "\n"
        )
        out_path = tmp_path / "annotations.json"

        cmd_annotate(
            argparse.Namespace(signals=str(signals_path), output=str(out_path))
        )
        doc = json.loads(out_path.read_text())
        records = {r["trial_path"]: r for r in doc["annotations"]}

        # Field is present on every record regardless of branch.
        for trial_path, rec in records.items():
            assert "annotation_result_status" in rec, (
                f"{trial_path} missing annotation_result_status; keys="
                f"{sorted(rec.keys())}"
            )

        # ok branch: trial with categories
        ok_rec = records["runs/t_ok"]
        assert ok_rec["categories"], "expected fixture to fire ≥1 heuristic"
        assert ok_rec["annotation_result_status"] == "ok"

        # no_categories branch: trial with no categories
        empty_rec = records["runs/t_empty"]
        assert empty_rec["categories"] == []
        assert empty_rec["annotation_result_status"] == "no_categories"


# ---------------------------------------------------------------------------
# cmd_report: JSONL input support (0.8.1 bead xo1)
# ---------------------------------------------------------------------------


class TestReportAcceptsJSONL:
    """Report must consume annotate's JSONL output without hand-conversion.
    0.8.0 required a {annotations:[...]} document and crashed on JSONL with
    'json.decoder.JSONDecodeError: Extra data'."""

    def _annotate_then_report(self, tmp_path, ext: str):
        from agent_diagnostics.cli import cmd_annotate, cmd_report

        # Two minimal signals so the report has something non-trivial to render.
        signals_path = tmp_path / "signals.jsonl"
        signals_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "task_id": f"t{i}",
                        "trial_path": f"runs/t{i}",
                        "reward": 1.0,
                        "passed": True,
                        "agent_name": "test-agent",
                        "total_turns": 5,
                        "trajectory_length": 5,
                    }
                )
                for i in range(2)
            )
            + "\n"
        )
        ann_path = tmp_path / f"annotations.{ext}"
        cmd_annotate(
            argparse.Namespace(signals=str(signals_path), output=str(ann_path))
        )

        report_dir = tmp_path / f"report_{ext}"
        cmd_report(
            argparse.Namespace(annotations=str(ann_path), output=str(report_dir))
        )
        return report_dir

    def test_report_consumes_jsonl_output_of_annotate(self, tmp_path):
        report_dir = self._annotate_then_report(tmp_path, "jsonl")
        md = report_dir / "reliability_report.md"
        js = report_dir / "reliability_report.json"
        assert md.is_file() and js.is_file()
        assert "Total annotated trials" in md.read_text()

    def test_report_still_consumes_json_document(self, tmp_path):
        report_dir = self._annotate_then_report(tmp_path, "json")
        md = report_dir / "reliability_report.md"
        js = report_dir / "reliability_report.json"
        assert md.is_file() and js.is_file()
        assert "Total annotated trials" in md.read_text()
