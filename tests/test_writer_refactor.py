"""Integration tests for writer-refactor: all four annotation writers route
through AnnotationStore.upsert_annotations() in narrow-tall schema.

Acceptance criteria:
- cmd_annotate writes via AnnotationStore with annotator_type='heuristic'
- cmd_llm_annotate writes via AnnotationStore with annotator_type='llm'
- cmd_predict writes via AnnotationStore with annotator_type='classifier'
- cmd_ensemble/ensemble_all writes via AnnotationStore with annotator_type='ensemble'
- ensemble_all remains exported from __init__.py
- Running all 4 writers sequentially on test fixture produces zero duplicate PKs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch


from agent_diagnostics.annotation_store import PK_FIELDS, AnnotationStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signals_list(n: int = 20) -> list[dict]:
    """Generate *n* distinct synthetic signal dicts with unique trial_ids."""
    from agent_diagnostics.signals import compute_trial_id

    signals = []
    for i in range(n):
        task_id = f"task-{i:03d}"
        config_name = "default"
        started_at = f"2026-01-01T00:00:{i:02d}Z"
        model = "test-model"
        trial_id, trial_id_full = compute_trial_id(
            task_id, config_name, started_at, model
        )
        signals.append(
            {
                "trial_id": trial_id,
                "trial_id_full": trial_id_full,
                "task_id": task_id,
                "trial_path": f"/tmp/trials/trial-{i}",
                "config_name": config_name,
                "started_at": started_at,
                "model": model,
                "benchmark": "test-bench",
                "reward": float(i % 2),
                "passed": bool(i % 2),
                "exception_crashed": i % 5 == 0,
                "rate_limited": i % 7 == 0,
                "has_verifier_result": True,
                "total_turns": 10,
                "tool_calls_total": 10 + i,
                "search_tool_calls": 2,
                "edit_tool_calls": 3,
                "code_nav_tool_calls": 1,
                "semantic_search_tool_calls": 0,
                "unique_files_read": 5,
                "unique_files_edited": 2,
                "files_read_list": [],
                "files_edited_list": [],
                "error_count": i % 3,
                "retry_count": 0,
                "trajectory_length": 10 + i,
                "has_result_json": True,
                "has_trajectory": True,
                "duration_seconds": 120.0 + i,
                "patch_size_lines": 50 + i,
                "tool_call_sequence": [],
                "benchmark_source": "test",
                "agent_name": "test-agent",
            }
        )
    return signals


def _pk_tuple(row: dict) -> tuple:
    """Extract PK tuple from a narrow-tall row."""
    return tuple(str(row.get(f, "")) for f in PK_FIELDS)


# ---------------------------------------------------------------------------
# Test: cmd_annotate writes via AnnotationStore (heuristic)
# ---------------------------------------------------------------------------


class TestCmdAnnotateStore:
    """cmd_annotate writes narrow-tall rows via AnnotationStore."""

    @patch("agent_diagnostics.annotator.annotate_trial")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_writes_heuristic_rows(self, mock_taxonomy, mock_annotate, tmp_path):
        from agent_diagnostics.cli import cmd_annotate
        from agent_diagnostics.types import CategoryAssignment

        mock_taxonomy.return_value = {"version": "3.0"}
        mock_annotate.return_value = [
            CategoryAssignment(
                name="exception_crash", confidence=0.95, evidence="crashed"
            ),
        ]

        signals = _make_signals_list(3)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        output = tmp_path / "annotations.json"
        ann_out = tmp_path / "annotations.jsonl"

        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            annotations_out=str(ann_out),
        )
        cmd_annotate(args)

        # Legacy output should exist
        assert output.exists()

        # AnnotationStore output should exist
        assert ann_out.exists()
        store = AnnotationStore(ann_out)
        rows = store.read_annotations()

        assert len(rows) == 3  # 3 signals x 1 category each
        for row in rows:
            assert row["annotator_type"] == "heuristic"
            assert row["annotator_identity"] == "heuristic:rule-engine"
            assert row["taxonomy_version"] == "3.0"
            assert row["category_name"] == "exception_crash"

    @patch("agent_diagnostics.annotator.annotate_trial")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_no_store_write_when_flag_absent(
        self, mock_taxonomy, mock_annotate, tmp_path
    ):
        """When --annotations-out is not set, no JSONL file is created."""
        from agent_diagnostics.cli import cmd_annotate
        from agent_diagnostics.types import CategoryAssignment

        mock_taxonomy.return_value = {"version": "3.0"}
        mock_annotate.return_value = [
            CategoryAssignment(name="cat1", confidence=0.8, evidence="test"),
        ]

        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps([{"task_id": "t1"}]))

        output = tmp_path / "annotations.json"
        ann_out = tmp_path / "annotations.jsonl"

        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            annotations_out=None,
        )
        cmd_annotate(args)

        assert output.exists()
        assert not ann_out.exists()


# ---------------------------------------------------------------------------
# Test: cmd_llm_annotate writes via AnnotationStore (llm)
# ---------------------------------------------------------------------------


class TestCmdLlmAnnotateStore:
    """cmd_llm_annotate writes narrow-tall rows via AnnotationStore."""

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_writes_llm_rows(self, mock_taxonomy, mock_batch, tmp_path):
        from agent_diagnostics.cli import cmd_llm_annotate

        # Create trial dirs with trajectory files
        signals = _make_signals_list(2)
        for sig in signals:
            trial_dir = Path(sig["trial_path"])
            agent_dir = trial_dir / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "trajectory.json").write_text("[]")

        mock_taxonomy.return_value = {"version": "3.0"}
        mock_batch.return_value = [
            [{"name": "retrieval_failure", "confidence": 0.9, "evidence": "test"}],
            [{"name": "query_churn", "confidence": 0.8, "evidence": "test2"}],
        ]

        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        output = tmp_path / "llm_out.json"
        ann_out = tmp_path / "annotations.jsonl"

        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=2,
            model="haiku",
            backend="claude-code",
            annotations_out=str(ann_out),
        )
        cmd_llm_annotate(args)

        assert ann_out.exists()
        store = AnnotationStore(ann_out)
        rows = store.read_annotations()

        assert len(rows) == 2
        for row in rows:
            assert row["annotator_type"] == "llm"
            assert row["annotator_identity"] == "llm:haiku-4"
            assert row["taxonomy_version"] == "3.0"


# ---------------------------------------------------------------------------
# Test: cmd_predict writes via AnnotationStore (classifier)
# ---------------------------------------------------------------------------


class TestCmdPredictStore:
    """cmd_predict writes narrow-tall rows via AnnotationStore."""

    @patch("agent_diagnostics.classifier.predict_all")
    @patch("agent_diagnostics.classifier.load_model")
    def test_writes_classifier_rows(self, mock_load, mock_predict, tmp_path):
        from agent_diagnostics.cli import cmd_predict

        signals = _make_signals_list(3)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_load.return_value = {"classifiers": {}}
        mock_predict.return_value = {
            "taxonomy_version": "3.0",
            "annotations": [
                {
                    "task_id": sig["task_id"],
                    "categories": [
                        {
                            "name": "retrieval_failure",
                            "confidence": 0.85,
                            "evidence": "clf",
                        }
                    ],
                }
                for sig in signals
            ],
        }

        output = tmp_path / "predict_out.json"
        ann_out = tmp_path / "annotations.jsonl"

        args = argparse.Namespace(
            model=str(tmp_path / "model.json"),
            signals=str(signals_file),
            output=str(output),
            threshold=0.5,
            annotations_out=str(ann_out),
        )
        cmd_predict(args)

        assert ann_out.exists()
        store = AnnotationStore(ann_out)
        rows = store.read_annotations()

        assert len(rows) == 3
        for row in rows:
            assert row["annotator_type"] == "classifier"
            assert row["annotator_identity"] == "classifier:trained-model"
            assert row["taxonomy_version"] == "3.0"


# ---------------------------------------------------------------------------
# Test: cmd_ensemble / ensemble_all writes via AnnotationStore (ensemble)
# ---------------------------------------------------------------------------


class TestCmdEnsembleStore:
    """cmd_ensemble writes narrow-tall rows via AnnotationStore."""

    @patch("agent_diagnostics.ensemble.ensemble_all")
    @patch("agent_diagnostics.classifier.load_model")
    def test_passes_annotations_out_to_ensemble_all(
        self, mock_load, mock_ensemble, tmp_path
    ):
        from agent_diagnostics.cli import cmd_ensemble

        signals = _make_signals_list(2)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_load.return_value = {"classifiers": {}}
        mock_ensemble.return_value = {
            "annotations": [{"task_id": "t1", "categories": []}],
            "tier_counts": {"heuristic": 1, "classifier": 0},
        }

        output = tmp_path / "ensemble_out.json"
        ann_out = tmp_path / "annotations.jsonl"

        args = argparse.Namespace(
            signals=str(signals_file),
            model=str(tmp_path / "model.json"),
            output=str(output),
            threshold=0.5,
            min_f1=0.7,
            annotations_out=str(ann_out),
        )
        cmd_ensemble(args)

        # Verify ensemble_all was called with annotations_out
        mock_ensemble.assert_called_once()
        call_kwargs = mock_ensemble.call_args
        assert call_kwargs[1].get("annotations_out") == str(ann_out)


class TestEnsembleAllStore:
    """ensemble_all writes to AnnotationStore when annotations_out is given."""

    def test_writes_ensemble_rows(self, tmp_path):
        from agent_diagnostics.ensemble import ensemble_all
        from agent_diagnostics.types import CategoryAssignment

        signals = _make_signals_list(3)
        ann_out = tmp_path / "annotations.jsonl"

        mock_model = {
            "classifiers": {},
            "means": [0.0] * 21,
            "stds": [1.0] * 21,
            "training_samples": 100,
        }

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=[
                    CategoryAssignment(
                        name="exception_crash", confidence=0.95, evidence="crashed"
                    )
                ],
            ),
            patch("agent_diagnostics.ensemble.predict_trial", return_value=[]),
        ):
            result = ensemble_all(
                signals,
                mock_model,
                annotations_out=str(ann_out),
            )

        # Legacy return format still works
        assert isinstance(result, dict)
        assert "annotations" in result
        assert result["annotator"]["type"] == "ensemble"

        # AnnotationStore file was created
        assert ann_out.exists()
        store = AnnotationStore(ann_out)
        rows = store.read_annotations()

        assert len(rows) == 3
        for row in rows:
            assert row["annotator_type"] == "ensemble"
            assert row["annotator_identity"] == "ensemble:heuristic+classifier"

    def test_no_store_write_without_annotations_out(self, tmp_path):
        """When annotations_out is None, no JSONL file is created."""
        from agent_diagnostics.ensemble import ensemble_all
        from agent_diagnostics.types import CategoryAssignment

        signals = _make_signals_list(2)
        ann_out = tmp_path / "annotations.jsonl"

        mock_model = {
            "classifiers": {},
            "means": [0.0] * 21,
            "stds": [1.0] * 21,
            "training_samples": 100,
        }

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=[
                    CategoryAssignment(
                        name="exception_crash", confidence=0.95, evidence="crashed"
                    )
                ],
            ),
            patch("agent_diagnostics.ensemble.predict_trial", return_value=[]),
        ):
            result = ensemble_all(signals, mock_model, annotations_out=None)

        assert isinstance(result, dict)
        assert not ann_out.exists()


# ---------------------------------------------------------------------------
# Test: ensemble_all still exported in __init__.py
# ---------------------------------------------------------------------------


class TestEnsembleAllExport:
    """ensemble_all remains in __all__ for backwards compat (R9)."""

    def test_exported_in_all(self):
        import agent_diagnostics

        assert "ensemble_all" in agent_diagnostics.__all__

    def test_importable(self):
        from agent_diagnostics import ensemble_all

        assert callable(ensemble_all)


# ---------------------------------------------------------------------------
# Test: _resolve_llm_annotator_identity
# ---------------------------------------------------------------------------


class TestResolveLlmIdentity:
    """Test the model alias -> annotator identity resolver."""

    def test_haiku_alias(self):
        from agent_diagnostics.cli import _resolve_llm_annotator_identity

        assert _resolve_llm_annotator_identity("haiku") == "llm:haiku-4"

    def test_sonnet_alias(self):
        from agent_diagnostics.cli import _resolve_llm_annotator_identity

        assert _resolve_llm_annotator_identity("sonnet") == "llm:sonnet-4"

    def test_opus_alias(self):
        from agent_diagnostics.cli import _resolve_llm_annotator_identity

        assert _resolve_llm_annotator_identity("opus") == "llm:opus-4"

    def test_snapshot_id_resolved(self):
        from agent_diagnostics.cli import _resolve_llm_annotator_identity

        result = _resolve_llm_annotator_identity("claude-haiku-4-5-20251001")
        assert result == "llm:haiku-4"

    def test_unknown_model_fallback(self):
        from agent_diagnostics.cli import _resolve_llm_annotator_identity

        result = _resolve_llm_annotator_identity("gpt-4o")
        assert result == "llm:gpt-4o"


# ---------------------------------------------------------------------------
# Test: _annotations_to_narrow_rows
# ---------------------------------------------------------------------------


class TestAnnotationsToNarrowRows:
    """Test the nested-to-narrow-tall conversion helper."""

    def test_basic_conversion(self):
        from agent_diagnostics.cli import _annotations_to_narrow_rows

        annotations = [
            {
                "trial_id": "abc123",
                "task_id": "t1",
                "categories": [
                    {"name": "cat1", "confidence": 0.9, "evidence": "ev1"},
                    {"name": "cat2", "confidence": 0.7, "evidence": "ev2"},
                ],
                "annotated_at": "2026-01-01T00:00:00Z",
            }
        ]
        rows = _annotations_to_narrow_rows(
            annotations,
            annotator_type="heuristic",
            annotator_identity="heuristic:rule-engine",
            taxonomy_version="3.0",
        )

        assert len(rows) == 2
        assert rows[0]["trial_id"] == "abc123"
        assert rows[0]["category_name"] == "cat1"
        assert rows[0]["annotator_type"] == "heuristic"
        assert rows[1]["category_name"] == "cat2"

    def test_computes_trial_id_when_missing(self):
        from agent_diagnostics.cli import _annotations_to_narrow_rows

        annotations = [
            {
                "task_id": "t1",
                "config_name": "cfg",
                "started_at": "2026-01-01T00:00:00Z",
                "model": "test",
                "categories": [{"name": "cat1", "confidence": 0.8, "evidence": "ev"}],
            }
        ]
        rows = _annotations_to_narrow_rows(
            annotations,
            annotator_type="heuristic",
            annotator_identity="heuristic:rule-engine",
            taxonomy_version="3.0",
        )

        assert len(rows) == 1
        assert rows[0]["trial_id"] != ""  # Should have computed a trial_id

    def test_empty_categories_produces_no_rows(self):
        from agent_diagnostics.cli import _annotations_to_narrow_rows

        annotations = [
            {"trial_id": "abc", "categories": []},
        ]
        rows = _annotations_to_narrow_rows(
            annotations,
            annotator_type="heuristic",
            annotator_identity="heuristic:rule-engine",
            taxonomy_version="3.0",
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Integration test: all 4 writers produce zero duplicate PKs
# ---------------------------------------------------------------------------


class TestZeroDuplicatePKs:
    """Run all 4 writers sequentially on a 20-trial fixture, verify zero
    duplicate primary keys in the shared AnnotationStore output."""

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.classifier.predict_all")
    @patch("agent_diagnostics.classifier.load_model")
    @patch("agent_diagnostics.annotator.annotate_trial")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_sequential_writers_no_duplicate_pks(
        self,
        mock_taxonomy,
        mock_heur_annotate,
        mock_load_model,
        mock_predict_all,
        mock_llm_batch,
        tmp_path,
    ):
        from agent_diagnostics.cli import (
            cmd_annotate,
            cmd_llm_annotate,
            cmd_predict,
        )
        from agent_diagnostics.types import CategoryAssignment

        mock_taxonomy.return_value = {"version": "3.0", "categories": []}

        signals = _make_signals_list(20)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        ann_out = str(tmp_path / "shared_annotations.jsonl")

        # --- Writer 1: cmd_annotate (heuristic) ---
        mock_heur_annotate.return_value = [
            CategoryAssignment(name="exception_crash", confidence=0.9, evidence="heur"),
        ]

        args_annotate = argparse.Namespace(
            signals=str(signals_file),
            output=str(tmp_path / "heuristic.json"),
            annotations_out=ann_out,
        )
        cmd_annotate(args_annotate)

        # --- Writer 2: cmd_llm_annotate (llm) ---
        # Create trial dirs with trajectory
        for sig in signals:
            agent_dir = Path(sig["trial_path"]) / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "trajectory.json").write_text("[]")

        mock_llm_batch.return_value = [
            [{"name": "retrieval_failure", "confidence": 0.85, "evidence": "llm"}]
            for _ in signals
        ]

        args_llm = argparse.Namespace(
            signals=str(signals_file),
            output=str(tmp_path / "llm.json"),
            sample_size=20,
            model="haiku",
            backend="claude-code",
            annotations_out=ann_out,
        )
        cmd_llm_annotate(args_llm)

        # --- Writer 3: cmd_predict (classifier) ---
        mock_load_model.return_value = {"classifiers": {}}
        mock_predict_all.return_value = {
            "taxonomy_version": "3.0",
            "annotations": [
                {
                    "task_id": sig["task_id"],
                    "categories": [
                        {"name": "query_churn", "confidence": 0.75, "evidence": "clf"}
                    ],
                }
                for sig in signals
            ],
        }

        args_predict = argparse.Namespace(
            model=str(tmp_path / "model.json"),
            signals=str(signals_file),
            output=str(tmp_path / "predict.json"),
            threshold=0.5,
            annotations_out=ann_out,
        )
        cmd_predict(args_predict)

        # --- Writer 4: cmd_ensemble (ensemble) ---
        # Use ensemble_all directly (since cmd_ensemble delegates to it)
        from agent_diagnostics.ensemble import ensemble_all

        mock_model = {
            "classifiers": {},
            "means": [0.0] * 21,
            "stds": [1.0] * 21,
            "training_samples": 100,
        }

        with (
            patch(
                "agent_diagnostics.ensemble.heuristic_annotate",
                return_value=[
                    CategoryAssignment(
                        name="rate_limited_run", confidence=0.8, evidence="ens"
                    )
                ],
            ),
            patch("agent_diagnostics.ensemble.predict_trial", return_value=[]),
            patch(
                "agent_diagnostics.ensemble.load_taxonomy",
                return_value={"version": "3.0", "categories": []},
            ),
        ):
            ensemble_all(
                signals,
                mock_model,
                annotations_out=ann_out,
            )

        # --- Verify zero duplicate PKs ---
        store = AnnotationStore(Path(ann_out))
        all_rows = store.read_annotations()

        # Each writer produced 20 rows with different (trial_id, category, annotator_type, identity)
        # heuristic: 20 x exception_crash x heuristic x heuristic:rule-engine
        # llm: 20 x retrieval_failure x llm x llm:haiku-4
        # classifier: 20 x query_churn x classifier x classifier:trained-model
        # ensemble: 20 x rate_limited_run x ensemble x ensemble:heuristic+classifier
        assert len(all_rows) == 80, f"Expected 80 rows, got {len(all_rows)}"

        # Check no duplicate PKs
        pks = [_pk_tuple(row) for row in all_rows]
        pk_set = set(pks)
        assert len(pks) == len(
            pk_set
        ), f"Found {len(pks) - len(pk_set)} duplicate primary keys"

        # Verify all 4 annotator types are present
        annotator_types = {row["annotator_type"] for row in all_rows}
        assert annotator_types == {"heuristic", "llm", "classifier", "ensemble"}


# ---------------------------------------------------------------------------
# Test: --annotations-out arg available on all 4 commands
# ---------------------------------------------------------------------------


class TestAnnotationsOutArgParsing:
    """Verify --annotations-out is registered on annotate, llm-annotate,
    predict, and ensemble subcommands."""

    def test_annotate_has_annotations_out(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agent_diagnostics", "annotate", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "--annotations-out" in result.stdout

    def test_llm_annotate_has_annotations_out(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agent_diagnostics", "llm-annotate", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "--annotations-out" in result.stdout

    def test_predict_has_annotations_out(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agent_diagnostics", "predict", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "--annotations-out" in result.stdout

    def test_ensemble_has_annotations_out(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agent_diagnostics", "ensemble", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "--annotations-out" in result.stdout
