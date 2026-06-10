"""Shared annotation-store helpers for CLI subcommands."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Narrow-tall annotation helpers
# ---------------------------------------------------------------------------

# Maps CLI model aliases (haiku, sonnet, opus) to logical annotator identities
# used by AnnotationStore.  Falls back to "llm:<alias>" for unknown aliases.
_MODEL_ALIAS_TO_IDENTITY: dict[str, str] = {
    "haiku": "llm:haiku-4",
    "sonnet": "llm:sonnet-4",
    "opus": "llm:opus-4",
}


def _resolve_llm_annotator_identity(model_arg: str) -> str:
    """Resolve a CLI --model argument to a logical annotator identity.

    Tries the static alias map first, then falls back to
    ``model_identity.resolve_identity`` for full snapshot IDs,
    and finally returns ``"llm:<model_arg>"`` as a last resort.
    """
    if model_arg in _MODEL_ALIAS_TO_IDENTITY:
        return _MODEL_ALIAS_TO_IDENTITY[model_arg]
    try:
        from agent_diagnostics.model_identity import resolve_identity

        return resolve_identity(model_arg)
    except (ValueError, KeyError):
        return f"llm:{model_arg}"


def _annotations_to_narrow_rows(
    annotations: list[dict[str, Any]],
    *,
    annotator_type: str,
    annotator_identity: str,
    taxonomy_version: str,
) -> list[dict[str, Any]]:
    """Convert nested annotation dicts to narrow-tall rows for AnnotationStore.

    Each input annotation has a ``categories`` list.  For every category
    assignment, one output row is emitted with the fields required by
    :data:`annotation_store.ROW_FIELDS`.
    """
    from agent_diagnostics.signals import compute_trial_id

    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for ann in annotations:
        # Resolve trial_id: prefer pre-computed, otherwise derive from fields
        trial_id = ann.get("trial_id", "")
        if not trial_id:
            tid, _ = compute_trial_id(
                ann.get("task_id", ""),
                ann.get("config_name", ""),
                ann.get("started_at", ""),
                ann.get("model", ""),
            )
            trial_id = tid

        annotated_at = ann.get("annotated_at", now)
        for cat in ann.get("categories", []):
            rows.append(
                {
                    "trial_id": trial_id,
                    "category_name": cat.get("name", ""),
                    "confidence": cat.get("confidence", 0.0),
                    "evidence": cat.get("evidence", ""),
                    "annotator_type": annotator_type,
                    "annotator_identity": annotator_identity,
                    "taxonomy_version": taxonomy_version,
                    "annotated_at": annotated_at,
                }
            )
    return rows


def _write_to_annotation_store(
    rows: list[dict[str, Any]],
    annotations_out: str,
    taxonomy_version: str,
) -> None:
    """Write narrow-tall rows to an AnnotationStore file."""
    if not rows:
        return
    from agent_diagnostics.annotation_store import AnnotationStore

    store = AnnotationStore(Path(annotations_out))
    count = store.upsert_annotations(rows, taxonomy_version=taxonomy_version)
    logger.info("AnnotationStore: wrote %d rows to %s", count, annotations_out)
