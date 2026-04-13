"""Tests for the agent-observatory CLI entrypoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
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
        "validate",
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


class TestCmdExtract:
    """Tests for cmd_extract."""

    def test_missing_runs_dir_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_extract

        args = argparse.Namespace(
            runs_dir=str(tmp_path / "nonexistent"),
            output=str(tmp_path / "out.json"),
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
        args = argparse.Namespace(runs_dir=str(runs_dir), output=str(output))
        cmd_extract(args)

        mock_extract.assert_called_once_with(runs_dir)
        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data) == 2


# ---------------------------------------------------------------------------
# cmd_annotate tests
# ---------------------------------------------------------------------------


class TestCmdAnnotate:
    """Tests for cmd_annotate."""

    def test_missing_signals_file_exits(self, tmp_path):
        from agent_diagnostics.cli import cmd_annotate

        import agent_diagnostics.annotator as _ann_mod

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

    def test_success(self, tmp_path):
        from agent_diagnostics.cli import cmd_annotate

        import agent_diagnostics.annotator as _ann_mod

        mock_annotate = MagicMock(
            return_value={
                "annotations": [{"task_id": "t1", "categories": [{"name": "cat1"}]}]
            }
        )
        _ann_mod.annotate_all = mock_annotate
        try:
            signals_file = tmp_path / "signals.json"
            signals_file.write_text(json.dumps([{"trial_id": "t1"}]))

            output = tmp_path / "annotations.json"
            args = argparse.Namespace(signals=str(signals_file), output=str(output))
            cmd_annotate(args)

            mock_annotate.assert_called_once()
            assert output.exists()
            data = json.loads(output.read_text())
            assert len(data["annotations"]) == 1
        finally:
            if hasattr(_ann_mod, "annotate_all"):
                del _ann_mod.annotate_all


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

    @patch("agent_diagnostics.llm_annotator.annotate_trial_llm")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_success_with_mock(self, mock_taxonomy, mock_annotate, tmp_path):
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

        mock_taxonomy.return_value = {"version": "1.0", "categories": []}
        mock_annotate.return_value = [
            {"name": "cat1", "confidence": 0.9, "evidence": "test"}
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

        mock_annotate.assert_called_once()
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["schema_version"] == "observatory-annotation-v1"
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["task_id"] == "task1"


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

    def test_jsonl_input_and_output(self, tmp_path):
        from agent_diagnostics.cli import cmd_annotate
        from agent_diagnostics.signals import write_jsonl

        import agent_diagnostics.annotator as _ann_mod

        # Write signals as JSONL
        signals_data = [{"trial_id": "t1"}, {"trial_id": "t2"}]
        signals_file = tmp_path / "signals.jsonl"
        write_jsonl(signals_data, signals_file)

        mock_annotate = MagicMock(
            return_value={
                "schema_version": "observatory-annotation-v1",
                "taxonomy_version": "3.0",
                "annotations": [
                    {"task_id": "t1", "categories": [{"name": "cat1"}]},
                    {"task_id": "t2", "categories": []},
                ],
            }
        )
        _ann_mod.annotate_all = mock_annotate
        try:
            output = tmp_path / "annotations.jsonl"
            args = argparse.Namespace(signals=str(signals_file), output=str(output))
            cmd_annotate(args)

            mock_annotate.assert_called_once()
            assert output.exists()

            lines = output.read_text().strip().splitlines()
            assert len(lines) == 2

            meta_path = output.with_suffix(".meta.json")
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["schema_version"] == "observatory-annotation-v1"
            assert meta["taxonomy_version"] == "3.0"
        finally:
            if hasattr(_ann_mod, "annotate_all"):
                del _ann_mod.annotate_all

    def test_json_input_jsonl_output(self, tmp_path):
        from agent_diagnostics.cli import cmd_annotate

        import agent_diagnostics.annotator as _ann_mod

        # Write signals as JSON (legacy)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"trial_id": "t1"}]))

        mock_annotate = MagicMock(
            return_value={
                "annotations": [{"task_id": "t1", "categories": [{"name": "cat1"}]}]
            }
        )
        _ann_mod.annotate_all = mock_annotate
        try:
            output = tmp_path / "annotations.jsonl"
            args = argparse.Namespace(signals=str(signals_file), output=str(output))
            cmd_annotate(args)

            assert output.exists()
            lines = output.read_text().strip().splitlines()
            assert len(lines) == 1
        finally:
            if hasattr(_ann_mod, "annotate_all"):
                del _ann_mod.annotate_all
