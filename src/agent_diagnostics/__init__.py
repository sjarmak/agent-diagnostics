"""Agent Reliability Observatory — behavioral taxonomy and annotation framework for coding agents."""

__version__ = "0.6.1"

from agent_diagnostics.annotator import annotate_trial
from agent_diagnostics.ensemble import ensemble_all, ensemble_annotate
from agent_diagnostics.llm_annotator import annotate_trial_llm
from agent_diagnostics.report import generate_report
from agent_diagnostics.signals import extract_all, extract_signals, load_manifest
from agent_diagnostics.taxonomy import (
    load_taxonomy,
    valid_category_names,
    validate_annotation_categories,
)
from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry
from agent_diagnostics.types import (
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
    "annotate_trial_llm",
    "ensemble_all",
    "ensemble_annotate",
    "extract_all",
    "extract_signals",
    "load_manifest",
    "generate_report",
    "load_taxonomy",
    "valid_category_names",
    "validate_annotation_categories",
]
