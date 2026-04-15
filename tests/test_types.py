"""Tests for agent_diagnostics.types — data contract types."""

from dataclasses import FrozenInstanceError
from typing import Protocol, get_type_hints

import pytest

from agent_diagnostics.types import (
    Annotation,
    AnnotationDocument,
    CategoryAssignment,
    TrialInput,
    TrialSignals,
)


class TestTrialSignalsImport:
    """TrialSignals TypedDict is importable and has 31 keys."""

    def test_import_succeeds(self) -> None:
        assert TrialSignals is not None

    def test_has_31_keys(self) -> None:
        hints = get_type_hints(TrialSignals)
        assert len(hints) == 31, f"Expected 31 keys, got {len(hints)}: {sorted(hints)}"

    def test_expected_keys_present(self) -> None:
        hints = get_type_hints(TrialSignals)
        expected_keys = {
            "trial_id",
            "trial_id_full",
            "task_id",
            "model",
            "agent_name",
            "config_name",
            "benchmark",
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
            "benchmark_source",
        }
        assert set(hints.keys()) == expected_keys

    def test_can_construct_partial(self) -> None:
        signals: TrialSignals = {"task_id": "t1", "reward": 0.5}  # type: ignore[typeddict-item]
        assert signals["task_id"] == "t1"


class TestTrialInput:
    """TrialInput is a Protocol."""

    def test_import_succeeds(self) -> None:
        assert TrialInput is not None

    def test_is_protocol(self) -> None:
        assert issubclass(TrialInput, Protocol)  # type: ignore[arg-type]

    def test_runtime_checkable(self) -> None:
        """A conforming object satisfies isinstance check."""

        class _Conforming:
            @property
            def task_id(self) -> str:
                return "t1"

            @property
            def trial_path(self) -> str:
                return "/path"

            @property
            def reward(self) -> float:
                return 1.0

            @property
            def passed(self) -> bool:
                return True

            @property
            def signals(self) -> TrialSignals:
                return TrialSignals(task_id="t1")  # type: ignore[typeddict-item]

        assert isinstance(_Conforming(), TrialInput)

    def test_non_conforming_fails(self) -> None:
        class _Missing:
            pass

        assert not isinstance(_Missing(), TrialInput)


class TestCategoryAssignment:
    """CategoryAssignment is a frozen dataclass."""

    def test_import_succeeds(self) -> None:
        assert CategoryAssignment is not None

    def test_construction(self) -> None:
        ca = CategoryAssignment(name="retrieval_failure", confidence=0.9)
        assert ca.name == "retrieval_failure"
        assert ca.confidence == 0.9
        assert ca.evidence is None

    def test_with_evidence(self) -> None:
        ca = CategoryAssignment(
            name="query_churn", confidence=0.8, evidence="step 3 repeated query"
        )
        assert ca.evidence == "step 3 repeated query"

    def test_frozen(self) -> None:
        ca = CategoryAssignment(name="x", confidence=0.5)
        with pytest.raises(FrozenInstanceError):
            ca.name = "y"  # type: ignore[misc]

    def test_fields(self) -> None:
        hints = get_type_hints(CategoryAssignment)
        assert "name" in hints
        assert "confidence" in hints
        assert "evidence" in hints


class TestAnnotation:
    """Annotation is a frozen dataclass with required and optional fields."""

    def test_import_succeeds(self) -> None:
        assert Annotation is not None

    def test_required_fields(self) -> None:
        ann = Annotation(
            task_id="t1",
            trial_path="/p",
            reward=1.0,
            passed=True,
            categories=(CategoryAssignment(name="c", confidence=0.9),),
        )
        assert ann.task_id == "t1"
        assert ann.trial_path == "/p"
        assert ann.reward == 1.0
        assert ann.passed is True
        assert len(ann.categories) == 1

    def test_optional_fields_default_none(self) -> None:
        ann = Annotation(
            task_id="t1",
            trial_path="/p",
            reward=0.0,
            passed=False,
            categories=(),
        )
        assert ann.run_id is None
        assert ann.model is None
        assert ann.config_name is None
        assert ann.benchmark is None
        assert ann.annotator_type is None
        assert ann.annotator_identity is None
        assert ann.notes is None
        assert ann.signals is None
        assert ann.annotated_at is None

    def test_frozen(self) -> None:
        ann = Annotation(
            task_id="t1",
            trial_path="/p",
            reward=0.0,
            passed=False,
            categories=(),
        )
        with pytest.raises(FrozenInstanceError):
            ann.task_id = "t2"  # type: ignore[misc]

    def test_all_optional_fields(self) -> None:
        signals: TrialSignals = {"task_id": "t1"}  # type: ignore[typeddict-item]
        ann = Annotation(
            task_id="t1",
            trial_path="/p",
            reward=1.0,
            passed=True,
            categories=(),
            run_id="run-1",
            model="claude-sonnet-4-6",
            config_name="baseline",
            benchmark="swebench_lite",
            annotator_type="llm",
            annotator_identity="claude-sonnet-4-6",
            notes="looks good",
            signals=signals,
            annotated_at="2026-04-04T00:00:00Z",
        )
        assert ann.run_id == "run-1"
        assert ann.model == "claude-sonnet-4-6"
        assert ann.annotated_at == "2026-04-04T00:00:00Z"


class TestAnnotationDocument:
    """AnnotationDocument is a frozen dataclass."""

    def test_import_succeeds(self) -> None:
        assert AnnotationDocument is not None

    def test_construction(self) -> None:
        doc = AnnotationDocument(
            schema_version="observatory-annotation-v1",
            annotations=(),
        )
        assert doc.schema_version == "observatory-annotation-v1"
        assert doc.annotations == ()
        assert doc.taxonomy_version is None
        assert doc.generated_at is None

    def test_frozen(self) -> None:
        doc = AnnotationDocument(
            schema_version="v1",
            annotations=(),
        )
        with pytest.raises(FrozenInstanceError):
            doc.schema_version = "v2"  # type: ignore[misc]

    def test_with_annotations(self) -> None:
        ann = Annotation(
            task_id="t1",
            trial_path="/p",
            reward=1.0,
            passed=True,
            categories=(),
        )
        doc = AnnotationDocument(
            schema_version="observatory-annotation-v1",
            annotations=(ann,),
            taxonomy_version="1.0",
            generated_at="2026-04-04T00:00:00Z",
        )
        assert len(doc.annotations) == 1
        assert doc.taxonomy_version == "1.0"
