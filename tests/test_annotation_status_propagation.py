"""End-to-end tests for annotation_result_status propagation.

Covers:
- cli.cmd_llm_annotate emitting per-row ``annotation_result_status`` and
  top-level ``annotation_summary`` for each AnnotationResult variant.
- calibrate.compare_annotations excluding errored trials and reporting
  error rates.
- blend_labels.blend skipping trials whose LLM annotation carries
  ``annotation_result_status == "error"``.
- Round-trip of the v2 schema fields through annotation_schema.json
  validation.
- FakeLLMBackend.set_next_result queueing each variant for test-time
  deterministic dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import jsonschema
import pytest

from agent_diagnostics.types import (
    AnnotationError,
    AnnotationNoCategoriesFound,
    AnnotationOk,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "agent_diagnostics"
    / "annotation_schema.json"
)


def _make_signals_list(n: int, tmp_path: Path) -> list[dict]:
    signals = []
    for i in range(n):
        trial_dir = tmp_path / f"trial_{i}"
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "trajectory.json").write_text("[]")
        signals.append(
            {
                "trial_path": str(trial_dir),
                "task_id": f"task_{i}",
                "reward": 1.0 if i % 2 == 0 else 0.0,
                "passed": i % 2 == 0,
                "trial_id": f"trial_id_{i:03d}",
            }
        )
    return signals


# ---------------------------------------------------------------------------
# cli.cmd_llm_annotate variant handling
# ---------------------------------------------------------------------------


class TestCmdLlmAnnotateVariantHandling:
    """Verify each AnnotationResult variant is serialised into the v2 JSON."""

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_ok_variant_yields_status_ok(
        self, mock_tax, mock_batch, tmp_path: Path
    ) -> None:
        from agent_diagnostics.cli import cmd_llm_annotate

        signals = _make_signals_list(1, tmp_path)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_tax.return_value = {"version": "3.0", "categories": []}
        mock_batch.return_value = [
            AnnotationOk(
                categories=(
                    {"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"},
                )
            )
        ]

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=1,
            model="haiku",
            backend="claude-code",
        )
        cmd_llm_annotate(args)

        data = json.loads(output.read_text())
        assert data["schema_version"] == "observatory-annotation-v2"
        assert data["annotation_summary"] == {
            "ok": 1,
            "no_categories": 0,
            "error": 0,
            "total": 1,
        }
        ann = data["annotations"][0]
        assert ann["annotation_result_status"] == "ok"
        assert ann["categories"][0]["name"] == "retrieval_failure"
        assert "error_reason" not in ann

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_no_categories_variant_yields_status_no_categories(
        self, mock_tax, mock_batch, tmp_path: Path
    ) -> None:
        from agent_diagnostics.cli import cmd_llm_annotate

        signals = _make_signals_list(1, tmp_path)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_tax.return_value = {"version": "3.0", "categories": []}
        mock_batch.return_value = [AnnotationNoCategoriesFound()]

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=1,
            model="haiku",
            backend="claude-code",
        )
        cmd_llm_annotate(args)

        data = json.loads(output.read_text())
        assert data["annotation_summary"]["no_categories"] == 1
        ann = data["annotations"][0]
        assert ann["annotation_result_status"] == "no_categories"
        assert ann["categories"] == []
        assert "error_reason" not in ann

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_error_variant_yields_status_error_and_reason(
        self, mock_tax, mock_batch, tmp_path: Path
    ) -> None:
        from agent_diagnostics.cli import cmd_llm_annotate

        signals = _make_signals_list(1, tmp_path)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_tax.return_value = {"version": "3.0", "categories": []}
        mock_batch.return_value = [
            AnnotationError(reason="claude CLI rc=1: timeout")
        ]

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=1,
            model="haiku",
            backend="claude-code",
        )
        cmd_llm_annotate(args)

        data = json.loads(output.read_text())
        assert data["annotation_summary"] == {
            "ok": 0,
            "no_categories": 0,
            "error": 1,
            "total": 1,
        }
        ann = data["annotations"][0]
        assert ann["annotation_result_status"] == "error"
        assert ann["categories"] == []
        assert ann["error_reason"] == "claude CLI rc=1: timeout"

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_mixed_variants_yield_correct_summary_counts(
        self, mock_tax, mock_batch, tmp_path: Path
    ) -> None:
        from agent_diagnostics.cli import cmd_llm_annotate

        signals = _make_signals_list(3, tmp_path)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_tax.return_value = {"version": "3.0", "categories": []}
        mock_batch.return_value = [
            AnnotationOk(
                categories=(
                    {"name": "query_churn", "confidence": 0.8, "evidence": "e"},
                )
            ),
            AnnotationNoCategoriesFound(),
            AnnotationError(reason="API rate limited"),
        ]

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=3,
            model="haiku",
            backend="claude-code",
        )
        cmd_llm_annotate(args)

        data = json.loads(output.read_text())
        assert data["annotation_summary"] == {
            "ok": 1,
            "no_categories": 1,
            "error": 1,
            "total": 3,
        }
        statuses = {a["annotation_result_status"] for a in data["annotations"]}
        assert statuses == {"ok", "no_categories", "error"}

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_errored_trials_produce_zero_annotation_store_rows(
        self, mock_tax, mock_batch, tmp_path: Path
    ) -> None:
        """Errored trials must not emit narrow-tall rows to AnnotationStore.

        AnnotationStore's PK is (trial_id, category_name, ...); errored
        trials have zero categories, so emitting nothing preserves the
        invariant.  Status lives only in the document JSON.
        """
        from agent_diagnostics.annotation_store import AnnotationStore
        from agent_diagnostics.cli import cmd_llm_annotate

        signals = _make_signals_list(2, tmp_path)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_tax.return_value = {"version": "3.0", "categories": []}
        mock_batch.return_value = [
            AnnotationOk(
                categories=(
                    {"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"},
                )
            ),
            AnnotationError(reason="timed out"),
        ]

        ann_out = tmp_path / "annotations.jsonl"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(tmp_path / "doc.json"),
            sample_size=2,
            model="haiku",
            backend="claude-code",
            annotations_out=str(ann_out),
        )
        cmd_llm_annotate(args)

        store = AnnotationStore(ann_out)
        rows = store.read_annotations()
        # Only the Ok trial contributes a row; the errored trial is absent.
        assert len(rows) == 1
        assert rows[0]["category_name"] == "retrieval_failure"

    @patch("agent_diagnostics.llm_annotator.annotate_batch")
    @patch("agent_diagnostics.taxonomy.load_taxonomy")
    def test_v2_output_validates_against_schema(
        self, mock_tax, mock_batch, tmp_path: Path
    ) -> None:
        """v2 documents with status fields must still pass jsonschema validation."""
        from agent_diagnostics.cli import cmd_llm_annotate

        signals = _make_signals_list(2, tmp_path)
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(signals))

        mock_tax.return_value = {"version": "3.0", "categories": []}
        mock_batch.return_value = [
            AnnotationOk(
                categories=(
                    {"name": "retrieval_failure", "confidence": 0.9, "evidence": "e"},
                )
            ),
            AnnotationError(reason="boom"),
        ]

        output = tmp_path / "out.json"
        args = argparse.Namespace(
            signals=str(signals_file),
            output=str(output),
            sample_size=2,
            model="haiku",
            backend="claude-code",
        )
        cmd_llm_annotate(args)

        data = json.loads(output.read_text())
        schema = json.loads(_SCHEMA_PATH.read_text())
        jsonschema.validate(instance=data, schema=schema)


# ---------------------------------------------------------------------------
# calibrate error-row filtering
# ---------------------------------------------------------------------------


def _write_annotations_doc(
    path: Path, rows: list[dict], *, schema_version: str = "observatory-annotation-v2"
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "annotations": rows,
            },
            indent=2,
        )
    )


class TestCalibrateErrorRowFiltering:
    """Verify errored trials are excluded from compare_annotations."""

    def test_errored_llm_trials_excluded_from_shared(self, tmp_path: Path) -> None:
        from agent_diagnostics.calibrate import compare_annotations

        heur = tmp_path / "heur.json"
        llm = tmp_path / "llm.json"
        _write_annotations_doc(
            heur,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "retrieval_failure", "confidence": 0.8}],
                },
                {
                    "task_id": "t2",
                    "trial_path": "runs/t2",
                    "reward": 0.0,
                    "passed": False,
                    "categories": [{"name": "query_churn", "confidence": 0.7}],
                },
            ],
        )
        _write_annotations_doc(
            llm,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "retrieval_failure", "confidence": 0.9}],
                    "annotation_result_status": "ok",
                },
                {
                    "task_id": "t2",
                    "trial_path": "runs/t2",
                    "reward": 0.0,
                    "passed": False,
                    "categories": [],
                    "annotation_result_status": "error",
                    "error_reason": "timed out",
                },
            ],
        )

        result = compare_annotations(heur, llm)

        # Only t1 is in shared (t2 excluded due to LLM error)
        assert result["shared_trials"] == 1
        assert result["excluded_errored_trials"] == 1
        assert result["llm_error_count"] == 1
        assert result["heuristic_error_count"] == 0
        # retrieval_failure has TP=1, 0 FP/FN (both sides agree on t1)
        rf = result["categories"]["retrieval_failure"]
        assert rf["true_positive"] == 1
        assert rf["false_positive"] == 0
        assert rf["false_negative"] == 0
        # query_churn should not appear because it's only on t2 which is excluded
        assert "query_churn" not in result["categories"]

    def test_legacy_v1_files_default_to_ok_status(self, tmp_path: Path) -> None:
        """Legacy docs without annotation_result_status are treated as all-ok."""
        from agent_diagnostics.calibrate import compare_annotations

        heur = tmp_path / "heur.json"
        llm = tmp_path / "llm.json"
        # v1 schema — no annotation_result_status fields
        _write_annotations_doc(
            heur,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "cat_a", "confidence": 0.9}],
                }
            ],
            schema_version="observatory-annotation-v1",
        )
        _write_annotations_doc(
            llm,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "cat_a", "confidence": 0.85}],
                }
            ],
            schema_version="observatory-annotation-v1",
        )

        result = compare_annotations(heur, llm)
        assert result["shared_trials"] == 1
        assert result["excluded_errored_trials"] == 0
        assert result["llm_error_count"] == 0

    def test_cross_model_excludes_errored(self, tmp_path: Path) -> None:
        from agent_diagnostics.calibrate import compare_cross_model

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_annotations_doc(
            a,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "near_miss", "confidence": 0.8}],
                    "annotation_result_status": "ok",
                },
                {
                    "task_id": "t2",
                    "trial_path": "runs/t2",
                    "reward": 0.0,
                    "passed": False,
                    "categories": [],
                    "annotation_result_status": "error",
                    "error_reason": "boom",
                },
            ],
        )
        _write_annotations_doc(
            b,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "near_miss", "confidence": 0.9}],
                    "annotation_result_status": "ok",
                },
                {
                    "task_id": "t2",
                    "trial_path": "runs/t2",
                    "reward": 0.0,
                    "passed": False,
                    "categories": [{"name": "something_else", "confidence": 0.7}],
                    "annotation_result_status": "ok",
                },
            ],
        )

        result = compare_cross_model(a, b)
        assert result["shared_trials"] == 1
        assert result["excluded_errored_trials"] == 1
        assert result["a_error_count"] == 1

    def test_format_markdown_surfaces_excluded_count(self, tmp_path: Path) -> None:
        from agent_diagnostics.calibrate import compare_annotations, format_markdown

        heur = tmp_path / "heur.json"
        llm = tmp_path / "llm.json"
        _write_annotations_doc(
            heur,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "cat_a", "confidence": 0.9}],
                },
                {
                    "task_id": "t2",
                    "trial_path": "runs/t2",
                    "reward": 0.0,
                    "passed": False,
                    "categories": [{"name": "cat_b", "confidence": 0.7}],
                },
            ],
        )
        _write_annotations_doc(
            llm,
            [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [{"name": "cat_a", "confidence": 0.85}],
                    "annotation_result_status": "ok",
                },
                {
                    "task_id": "t2",
                    "trial_path": "runs/t2",
                    "reward": 0.0,
                    "passed": False,
                    "categories": [],
                    "annotation_result_status": "error",
                    "error_reason": "boom",
                },
            ],
        )
        summary = compare_annotations(heur, llm)
        md = format_markdown(summary)
        assert "Excluded errored trials: 1" in md


# ---------------------------------------------------------------------------
# blend_labels error-row skipping
# ---------------------------------------------------------------------------


class TestBlendErrorRowSkipping:
    """Verify blend() skips LLM-errored trials at the top of the loop."""

    def test_errored_llm_trial_not_blended(self, tmp_path: Path) -> None:
        from agent_diagnostics.blend_labels import blend

        heur = tmp_path / "heur.json"
        llm = tmp_path / "llm.json"
        heur.write_text(
            json.dumps(
                {
                    "schema_version": "observatory-annotation-v1",
                    "annotations": [
                        {
                            "task_id": "t1",
                            "trial_path": "runs/t1",
                            "reward": 1.0,
                            "passed": True,
                            "categories": [
                                {
                                    "name": "exception_crash",
                                    "confidence": 0.9,
                                    "evidence": "heur",
                                }
                            ],
                        },
                        {
                            "task_id": "t2",
                            "trial_path": "runs/t2",
                            "reward": 0.0,
                            "passed": False,
                            "categories": [
                                {
                                    "name": "exception_crash",
                                    "confidence": 0.9,
                                    "evidence": "heur",
                                }
                            ],
                        },
                    ],
                }
            )
        )
        llm.write_text(
            json.dumps(
                {
                    "schema_version": "observatory-annotation-v2",
                    "annotations": [
                        {
                            "task_id": "t1",
                            "trial_path": "runs/t1",
                            "reward": 1.0,
                            "passed": True,
                            "categories": [
                                {
                                    "name": "retrieval_failure",
                                    "confidence": 0.85,
                                    "evidence": "llm",
                                }
                            ],
                            "annotation_result_status": "ok",
                        },
                        {
                            "task_id": "t2",
                            "trial_path": "runs/t2",
                            "reward": 0.0,
                            "passed": False,
                            "categories": [],
                            "annotation_result_status": "error",
                            "error_reason": "timed out",
                        },
                    ],
                }
            )
        )

        result = blend(heur, llm)
        # Only t1 survives; t2's LLM-errored annotation suppresses the
        # whole trial (including the heuristic fallback path).
        assert len(result["annotations"]) == 1
        assert result["annotations"][0]["trial_path"] == "runs/t1"
        assert result["blend_metadata"]["skipped_errored_llm_trials"] == 1


# ---------------------------------------------------------------------------
# Report's Annotation Quality section
# ---------------------------------------------------------------------------


class TestReportAnnotationQualitySection:
    """Verify generate_report surfaces the annotation_summary when present."""

    def test_markdown_includes_annotation_quality_section(
        self, tmp_path: Path
    ) -> None:
        from agent_diagnostics.report import generate_report

        annotations = {
            "schema_version": "observatory-annotation-v2",
            "taxonomy_version": "3.0",
            "annotation_summary": {
                "ok": 8,
                "no_categories": 1,
                "error": 1,
                "total": 10,
            },
            "annotations": [
                {
                    "task_id": f"t{i}",
                    "trial_path": f"runs/t{i}",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [],
                }
                for i in range(10)
            ],
        }

        md_path, json_path = generate_report(annotations, tmp_path / "report")
        md = md_path.read_text()
        assert "## Annotation Quality" in md
        assert "| ok | 8 |" in md
        assert "| error | 1 |" in md
        payload = json.loads(json_path.read_text())
        assert payload["annotation_summary"]["error"] == 1

    def test_markdown_omits_section_when_no_summary_present(
        self, tmp_path: Path
    ) -> None:
        from agent_diagnostics.report import generate_report

        annotations = {
            "schema_version": "observatory-annotation-v1",
            "annotations": [
                {
                    "task_id": "t1",
                    "trial_path": "runs/t1",
                    "reward": 1.0,
                    "passed": True,
                    "categories": [],
                }
            ],
        }

        md_path, json_path = generate_report(annotations, tmp_path / "report")
        md = md_path.read_text()
        assert "## Annotation Quality" not in md
        payload = json.loads(json_path.read_text())
        assert "annotation_summary" not in payload


# ---------------------------------------------------------------------------
# FakeLLMBackend.set_next_result
# ---------------------------------------------------------------------------


class TestFakeLLMBackendSetNextResult:
    """Verify the queued-result mechanism exercises all three variants."""

    def test_queue_ok(self) -> None:
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        cat = {"name": "retrieval_failure", "confidence": 0.9, "evidence": "queued"}
        backend.set_next_result(AnnotationOk(categories=(cat,)))
        result = backend.annotate_as_result("irrelevant prompt")
        assert isinstance(result, AnnotationOk)
        assert result.categories == (cat,)

    def test_queue_no_categories(self) -> None:
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        backend.set_next_result(AnnotationNoCategoriesFound())
        result = backend.annotate_as_result("irrelevant")
        assert isinstance(result, AnnotationNoCategoriesFound)

    def test_queue_error(self) -> None:
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        backend.set_next_result(AnnotationError(reason="rate limit"))
        result = backend.annotate_as_result("irrelevant")
        assert isinstance(result, AnnotationError)
        assert result.reason == "rate limit"

    def test_queue_fifo_order(self) -> None:
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        backend.set_next_result(AnnotationNoCategoriesFound())
        backend.set_next_result(AnnotationError(reason="second"))
        r1 = backend.annotate_as_result("a")
        r2 = backend.annotate_as_result("b")
        assert isinstance(r1, AnnotationNoCategoriesFound)
        assert isinstance(r2, AnnotationError)

    def test_queue_empty_falls_back_to_keyword_match(self) -> None:
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        result = backend.annotate_as_result("prompt mentioning retrieval_failure here")
        assert isinstance(result, AnnotationOk)
        names = {c["name"] for c in result.categories}
        assert "retrieval_failure" in names

    def test_set_next_result_rejects_non_variant(self) -> None:
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        with pytest.raises(TypeError, match="AnnotationResult variant"):
            backend.set_next_result({"not": "an AnnotationResult"})  # type: ignore[arg-type]

    def test_annotate_dict_api_raises_on_queued_error(self) -> None:
        """The dict-returning annotate() raises when an error is queued,
        simulating a live backend failure for integration tests."""
        from tests.fake_llm_backend import FakeLLMBackend

        backend = FakeLLMBackend()
        backend.set_next_result(AnnotationError(reason="simulated failure"))
        with pytest.raises(RuntimeError, match="simulated failure"):
            backend.annotate("prompt")


# ---------------------------------------------------------------------------
# Batch error propagation: annotate_batch_claude_code & annotate_batch_api
# propagate AnnotationError(reason) through their async inner coroutines
# (the CRITICAL fix from the plan-review).
# ---------------------------------------------------------------------------


class TestBatchErrorReasonsPropagate:
    """Async batch paths must surface AnnotationError with a specific reason."""

    def test_claude_code_batch_subprocess_failure_has_reason(
        self, tmp_path: Path
    ) -> None:
        from agent_diagnostics.llm_annotator import annotate_batch_claude_code

        trial_dir = tmp_path / "trial"
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "agent" / "trajectory.json").write_text('{"steps": []}')
        (trial_dir / "agent" / "instruction.txt").write_text("task")

        async def _failing_subprocess(*args, **kwargs):
            proc = mock.MagicMock()
            proc.returncode = 2
            proc.communicate = mock.AsyncMock(return_value=(b"", b"boom"))
            return proc

        with (
            mock.patch(
                "agent_diagnostics.llm_annotator.shutil.which",
                return_value="/usr/bin/claude",
            ),
            mock.patch(
                "asyncio.create_subprocess_exec",
                side_effect=_failing_subprocess,
            ),
        ):
            results = annotate_batch_claude_code([trial_dir], [{}])

        assert len(results) == 1
        assert isinstance(results[0], AnnotationError)
        assert "rc=2" in results[0].reason

    def test_api_batch_runtime_error_has_reason(self, tmp_path: Path) -> None:
        from agent_diagnostics.llm_annotator import annotate_batch_api

        trial_dir = tmp_path / "trial"
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "agent" / "trajectory.json").write_text('{"steps": []}')
        (trial_dir / "agent" / "instruction.txt").write_text("task")

        mock_aclient = mock.MagicMock()
        mock_aclient.messages.create = mock.AsyncMock(
            side_effect=RuntimeError("specific api failure")
        )
        mock_anthropic = mock.MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_aclient

        with mock.patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            results = annotate_batch_api([trial_dir], [{}])

        assert len(results) == 1
        assert isinstance(results[0], AnnotationError)
        assert "specific api failure" in results[0].reason
