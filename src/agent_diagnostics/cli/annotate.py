"""Annotation subcommands: heuristic annotate and LLM annotate."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_diagnostics.cli._helpers import (
    _annotations_to_narrow_rows,
    _resolve_llm_annotator_identity,
    _write_to_annotation_store,
)

logger = logging.getLogger(__name__)


def cmd_annotate(args):
    """Generate heuristic annotations from extracted signals."""
    from datetime import datetime, timezone

    from agent_diagnostics.annotator import annotate_trial
    from agent_diagnostics.signals import load_signals, write_output
    from agent_diagnostics.taxonomy import load_taxonomy

    signals_path = Path(args.signals)
    if not signals_path.is_file():
        logger.error("signals file not found: %s", signals_path)
        sys.exit(1)

    signals_list = load_signals(signals_path)
    taxonomy = load_taxonomy()
    now = datetime.now(timezone.utc).isoformat()

    annotation_list = []
    for sig in signals_list:
        assignments = annotate_trial(sig)
        record: dict[str, Any] = {
            "task_id": sig.get("task_id", ""),
            "trial_path": sig.get("trial_path", ""),
            "reward": sig.get("reward"),
            "passed": sig.get("passed", False),
            # Mirror the LLM-annotate v2 contract so downstream consumers
            # can branch on status uniformly regardless of annotator path.
            # Heuristic annotation is deterministic and never errors per-trial;
            # "ok" / "no_categories" are the only reachable states.
            "annotation_result_status": "ok" if assignments else "no_categories",
            "categories": [
                {
                    "name": a.name,
                    "confidence": a.confidence,
                    "evidence": a.evidence or "",
                }
                for a in assignments
            ],
        }
        # Carry analytic fields from signals through to annotation records so
        # downstream consumers (report.py) can slice by benchmark / agent /
        # trajectory availability without re-reading the signals file.
        for key in (
            "trial_id",
            "trial_id_full",
            "agent_name",
            "model",
            "config_name",
            "benchmark",
            "benchmark_source",
            "has_trajectory",
            "trajectory_length",
            "total_turns",
        ):
            value = sig.get(key)
            if value is not None and value != "":
                record[key] = value
        annotation_list.append(record)

    annotations = {
        "schema_version": "observatory-annotation-v1",
        "taxonomy_version": str(taxonomy.get("version", "")),
        "generated_at": now,
        "annotations": annotation_list,
    }

    output = Path(args.output)
    write_output(annotations, output)

    total_categories = sum(len(a["categories"]) for a in annotation_list)
    logger.info(
        "Annotated %d trials with %d total category assignments",
        len(annotation_list),
        total_categories,
    )

    # Write to AnnotationStore if --annotations-out is provided
    annotations_out = getattr(args, "annotations_out", None)
    if annotations_out:
        taxonomy_version = str(taxonomy.get("version", ""))
        # Enrich annotation_list with trial_id from signals for narrow-row conversion
        enriched = []
        for sig, ann in zip(signals_list, annotation_list):
            enriched.append({**ann, "trial_id": sig.get("trial_id", "")})
        rows = _annotations_to_narrow_rows(
            enriched,
            annotator_type="heuristic",
            annotator_identity="heuristic:rule-engine",
            taxonomy_version=taxonomy_version,
        )
        _write_to_annotation_store(rows, annotations_out, taxonomy_version)


def cmd_llm_annotate(args):
    """Generate LLM-assisted annotations for a sample of trials.

    Consumes :class:`~agent_diagnostics.types.AnnotationResult` variants
    returned by :func:`~agent_diagnostics.llm_annotator.annotate_batch` and
    serialises them into ``observatory-annotation-v2`` with a per-trial
    ``annotation_result_status`` (``"ok"`` / ``"no_categories"`` / ``"error"``)
    and a top-level ``annotation_summary`` counting each variant.  Error
    reasons are logged at WARNING and persisted in ``error_reason`` on the
    affected annotation so downstream consumers can exclude them deterministically.
    """
    import random

    from agent_diagnostics.llm_annotator import annotate_batch
    from agent_diagnostics.signals import load_signals
    from agent_diagnostics.taxonomy import load_taxonomy
    from agent_diagnostics.types import (
        AnnotationError,
        AnnotationNoCategoriesFound,
        AnnotationOk,
    )

    signals_path = Path(args.signals)
    if not signals_path.is_file():
        logger.error("signals file not found: %s", signals_path)
        sys.exit(1)

    signals_list = load_signals(signals_path)

    # Filter to trials that have trajectories on disk
    has_trajectory = [
        s
        for s in signals_list
        if s.get("trial_path") and (Path(s["trial_path"]) / "agent" / "trajectory.json").is_file()
    ]
    if not has_trajectory:
        logger.error("no trials with trajectory files found")
        sys.exit(1)

    backend = args.backend
    sample_size = min(args.sample_size, len(has_trajectory))
    sampled = random.sample(has_trajectory, sample_size)
    logger.info(
        "Sampled %d trials (from %d with trajectories)",
        sample_size,
        len(has_trajectory),
    )
    logger.info("Backend: %s, Model: %s", backend, args.model)

    taxonomy = load_taxonomy()
    now = datetime.now(timezone.utc).isoformat()

    # Use concurrent batch path for speed, falling back to sequential
    trial_dirs = [sig["trial_path"] for sig in sampled]
    batch_results = annotate_batch(
        trial_dirs,
        sampled,
        model=args.model,
        max_concurrent=1,
        backend=backend,
    )

    annotations = []
    summary_counts = {"ok": 0, "no_categories": 0, "error": 0}
    for sig, result in zip(sampled, batch_results):
        reward_val = sig.get("reward")
        match result:
            case AnnotationOk(categories=cats):
                status, categories, error_reason = "ok", list(cats), None
            case AnnotationNoCategoriesFound():
                status, categories, error_reason = "no_categories", [], None
            case AnnotationError(reason=reason):
                status, categories, error_reason = "error", [], reason
                # Truncate defensively at the log boundary even though
                # parsing._short_exc already caps reasons at construction.
                logger.warning(
                    "LLM annotation failed for trial %s: %s",
                    sig.get("trial_path", "<unknown>"),
                    reason[:500],
                )
            case _:  # pragma: no cover — defensive, union is exhaustive
                raise TypeError(f"Unexpected AnnotationResult variant: {type(result)!r}")
        summary_counts[status] += 1

        annotation: dict[str, Any] = {
            "task_id": sig.get("task_id") or "unknown",
            "trial_path": sig.get("trial_path") or "",
            "reward": float(reward_val) if reward_val is not None else 0.0,
            "passed": (bool(sig.get("passed")) if sig.get("passed") is not None else False),
            "categories": categories,
            "annotated_at": now,
            "annotation_result_status": status,
        }
        if error_reason is not None:
            annotation["error_reason"] = error_reason
        for key in ("config_name", "benchmark", "model"):
            if sig.get(key):
                annotation[key] = sig[key]
        annotations.append(annotation)

    annotation_summary = {**summary_counts, "total": len(annotations)}

    result_doc = {
        "schema_version": "observatory-annotation-v2",
        "taxonomy_version": str(taxonomy["version"]),
        "generated_at": now,
        "annotator": {
            "type": "llm",
            "identity": f"observatory.llm_annotator model={args.model} backend={backend}",
        },
        "annotation_summary": annotation_summary,
        "annotations": annotations,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result_doc, f, indent=2, default=str)

    total_categories = sum(len(a["categories"]) for a in annotations)
    logger.info(
        "Done: %d/%d trials annotated with %d category assignments "
        "(ok=%d, no_categories=%d, error=%d). Output: %s",
        len(annotations),
        sample_size,
        total_categories,
        summary_counts["ok"],
        summary_counts["no_categories"],
        summary_counts["error"],
        output,
    )

    # Write to AnnotationStore if --annotations-out is provided.
    # Errored trials produce zero narrow rows (status is not persisted in
    # AnnotationStore to keep the PK invariant; the document JSON carries
    # status + annotation_summary for downstream aggregation).
    annotations_out = getattr(args, "annotations_out", None)
    if annotations_out:
        taxonomy_version = str(taxonomy["version"])
        annotator_identity = _resolve_llm_annotator_identity(args.model)
        # Enrich annotations with trial_id from signals
        enriched = []
        for sig, ann in zip(sampled, annotations):
            enriched.append({**ann, "trial_id": sig.get("trial_id", "")})
        rows = _annotations_to_narrow_rows(
            enriched,
            annotator_type="llm",
            annotator_identity=annotator_identity,
            taxonomy_version=taxonomy_version,
        )
        _write_to_annotation_store(rows, annotations_out, taxonomy_version)
