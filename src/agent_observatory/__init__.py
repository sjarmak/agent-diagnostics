"""Agent Reliability Observatory — behavioral taxonomy and annotation framework for coding agents."""

__version__ = "0.4.0"

from agent_observatory.annotator import annotate_trial
from agent_observatory.ensemble import ensemble_all, ensemble_annotate
from agent_observatory.signals import extract_all, extract_signals
from agent_observatory.taxonomy import (
    load_taxonomy,
    valid_category_names,
    validate_annotation_categories,
)
from agent_observatory.tool_registry import DEFAULT_REGISTRY, ToolRegistry
from agent_observatory.types import (
    Annotation,
    AnnotationDocument,
    CategoryAssignment,
    TrialInput,
    TrialSignals,
)

__all__ = [
    "Annotation",
    "AnnotationDocument",
    "CategoryAssignment",
    "DEFAULT_REGISTRY",
    "ToolRegistry",
    "TrialInput",
    "TrialSignals",
    "annotate_trial",
    "ensemble_all",
    "ensemble_annotate",
    "extract_all",
    "extract_signals",
    "load_taxonomy",
    "valid_category_names",
    "validate_annotation_categories",
]
