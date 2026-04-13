"""Type definitions forming the data contract between signal extraction and annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, TypedDict, Union, runtime_checkable

# ---------------------------------------------------------------------------
# TrialSignals — typed dict of the 26 signal keys extracted from a trial
# ---------------------------------------------------------------------------


class TrialSignals(TypedDict, total=False):
    """Raw signal values extracted from a single benchmark trial.

    All 29 keys are declared; ``total=False`` allows partial construction
    during incremental extraction.
    """

    task_id: str
    model: str
    agent_name: str
    config_name: str
    benchmark: str
    reward: float | None
    passed: bool
    has_verifier_result: bool
    total_turns: int
    tool_calls_total: int
    search_tool_calls: int
    edit_tool_calls: int
    code_nav_tool_calls: int
    semantic_search_tool_calls: int
    unique_files_read: int
    unique_files_edited: int
    files_read_list: list[str]
    files_edited_list: list[str]
    error_count: int
    retry_count: int
    trajectory_length: int
    has_result_json: bool
    has_trajectory: bool
    duration_seconds: float
    rate_limited: bool
    exception_crashed: bool
    patch_size_lines: int
    tool_call_sequence: list[str]
    benchmark_source: str


# ---------------------------------------------------------------------------
# TrialInput — protocol describing what data the observatory needs to consume
# ---------------------------------------------------------------------------


@runtime_checkable
class TrialInput(Protocol):
    """Structural interface that any trial data source must satisfy.

    Implementations may be dataclasses, plain objects, or adapter wrappers —
    as long as they expose these attributes the observatory can consume them.
    """

    @property
    def task_id(self) -> str: ...

    @property
    def trial_path(self) -> str: ...

    @property
    def reward(self) -> float | None: ...

    @property
    def passed(self) -> bool: ...

    @property
    def signals(self) -> TrialSignals: ...


# ---------------------------------------------------------------------------
# CategoryAssignment — a single taxonomy category applied to a trial
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryAssignment:
    """A single taxonomy category assigned to a trial with confidence."""

    name: str
    confidence: float
    evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Annotation — fully resolved annotation for one trial
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Annotation:
    """Complete annotation record for a single benchmark trial."""

    # Required fields (mirror JSON schema required list)
    task_id: str
    trial_path: str
    reward: Optional[float]
    passed: bool
    categories: tuple[CategoryAssignment, ...]

    # Optional metadata
    run_id: Optional[str] = None
    model: Optional[str] = None
    config_name: Optional[str] = None
    benchmark: Optional[str] = None
    annotator_type: Optional[str] = None
    annotator_identity: Optional[str] = None
    notes: Optional[str] = None
    signals: Optional[TrialSignals] = None
    annotated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# AnnotationDocument — top-level wrapper for a collection of annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationDocument:
    """Top-level annotation document wrapping multiple trial annotations."""

    schema_version: str
    annotations: tuple[Annotation, ...]
    taxonomy_version: Optional[str] = None
    generated_at: Optional[str] = None
    annotator_type: Optional[str] = None
    annotator_identity: Optional[str] = None


# ---------------------------------------------------------------------------
# AnnotationResult — discriminated union for annotation outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationOk:
    """Successful annotation with one or more categories."""

    categories: tuple[dict, ...]


@dataclass(frozen=True)
class AnnotationNoCategoriesFound:
    """LLM returned no categories for this trial."""

    pass


@dataclass(frozen=True)
class AnnotationError:
    """Annotation failed with an error."""

    reason: str


AnnotationResult = Union[AnnotationOk, AnnotationNoCategoriesFound, AnnotationError]
AnnotationResult.__doc__ = (
    "Discriminated union representing the outcome of an annotation attempt. "
    "Variants: AnnotationOk (success), AnnotationNoCategoriesFound (empty), "
    "AnnotationError (failure with reason)."
)
