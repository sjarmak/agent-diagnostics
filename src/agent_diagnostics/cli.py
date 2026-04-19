"""CLI entrypoint for the Agent Reliability Observatory.

Logging discipline
------------------
The root logger is configured only in :func:`main` (the process entry point);
library modules obtain loggers via ``logging.getLogger(__name__)`` and never
install handlers of their own.  Log records are emitted on stderr with the
format ``"%(levelname)s %(name)s: %(message)s"`` so structured pipelines can
parse them without interleaving user-facing stdout.

Genuine user-facing stdout output (tables, schema reports, pipeline summaries)
is still emitted with :func:`print`; each such call site carries an inline
``# STDOUT: <why>`` comment.  Every other caller in this module uses the module
logger below.
"""

import argparse
import json
import logging
import os
import sys
import warnings
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


def cmd_extract(args):
    """Extract signals from trial directories."""
    from agent_diagnostics.signals import extract_all, write_output

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        logger.error("runs directory not found: %s", runs_dir)
        sys.exit(1)

    cache = None
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir:
        from agent_diagnostics.extract_cache import SignalsCache

        cache = SignalsCache(Path(cache_dir))

    signals = extract_all(runs_dir, cache=cache)

    output = Path(args.output)
    write_output(signals, output)

    suffix = ""
    if cache is not None:
        stats = cache.stats
        suffix = (
            f" (cache: {stats['hits']} hits, {stats['misses']} misses, "
            f"{stats['entries']} entries)"
        )
    logger.info("Extracted signals from %d trials%s", len(signals), suffix)


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


def cmd_report(args):
    """Generate reliability report from annotations.

    Accepts either ``.json`` (annotation document with ``{"annotations":
    [...]}`` or bare list) or ``.jsonl`` (one record per line) so the
    output of ``observatory annotate`` is consumable directly regardless
    of the extension the user chose for that stage.

    Output destination is controlled by ``--output-dir`` (canonical) or
    ``--output`` (deprecated alias, retained for 0.8.x compatibility —
    slated for removal in 1.0).
    """
    from agent_diagnostics.report import generate_report

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        logger.error("annotations file not found: %s", annotations_path)
        sys.exit(1)

    if annotations_path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with open(annotations_path) as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.error(
                        "malformed JSON on line %d of %s: %s",
                        lineno,
                        annotations_path,
                        exc,
                    )
                    sys.exit(1)
        annotations: dict[str, Any] = {"annotations": records}
    else:
        with open(annotations_path) as f:
            annotations = json.load(f)
        # Tolerate a bare-list .json file (older tooling wrote this shape).
        if isinstance(annotations, list):
            annotations = {"annotations": annotations}

    # Resolve output directory from --output-dir (canonical) or --output
    # (deprecated). argparse's mutually_exclusive_group guarantees at most
    # one is set when invoked through main(); direct Namespace callers may
    # supply either attribute.
    output_dir_value = getattr(args, "output_dir", None)
    legacy_output = getattr(args, "output", None)
    if legacy_output is not None and output_dir_value is None:
        _deprecation_msg = (
            "`observatory report --output` is deprecated; use --output-dir "
            "instead. The --output alias will be removed in 1.0."
        )
        warnings.warn(_deprecation_msg, DeprecationWarning, stacklevel=2)
        # Python silences DeprecationWarning by default outside __main__, so
        # also log it via the CLI's configured logger — that path is visible
        # regardless of the user's warning filter.
        logger.warning(_deprecation_msg)
        output_dir_value = legacy_output
    if output_dir_value is None:
        logger.error("--output-dir is required")
        sys.exit(1)

    output_dir = Path(output_dir_value)
    md_path, json_path = generate_report(annotations, output_dir)

    logger.info("Report written to %s and %s", md_path, json_path)


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

    with open(signals_path) as f:
        signals_list = json.load(f)

    # Filter to trials that have trajectories on disk
    has_trajectory = [
        s
        for s in signals_list
        if s.get("trial_path")
        and (Path(s["trial_path"]) / "agent" / "trajectory.json").is_file()
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
            "passed": (
                bool(sig.get("passed")) if sig.get("passed") is not None else False
            ),
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


def cmd_train(args):
    """Train per-category classifiers from LLM-labeled data."""
    from agent_diagnostics.classifier import (
        evaluate,
        format_eval_markdown,
        save_model,
        train,
    )

    model = train(
        llm_file=args.labels,
        signals_file=args.signals,
        min_positive=args.min_positive,
        lr=args.lr,
        epochs=args.epochs,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output)

    n_clf = len(model["classifiers"])
    n_skip = len(model["skipped_categories"])
    logger.info(
        "Trained %d classifiers on %d samples (%d categories skipped, < %d positive)",
        n_clf,
        model["training_samples"],
        n_skip,
        model["min_positive"],
    )
    for cat, clf in sorted(model["classifiers"].items()):
        logger.info(
            "  %s: %d/%d positive, train_acc=%.2f",
            cat,
            clf["positive_count"],
            clf["total_count"],
            clf["train_accuracy"],
        )

    # If --eval is provided, evaluate on the same data (quick sanity check)
    if args.eval:
        eval_results = evaluate(model, args.labels, args.signals)
        logger.info("\n%s", format_eval_markdown(eval_results, model))

    logger.info("Model saved to %s", output)


def cmd_predict(args):
    """Predict categories for all trials using a trained classifier."""
    from agent_diagnostics.classifier import load_model, predict_all

    model = load_model(args.model)
    with open(args.signals) as f:
        signals_list = json.load(f)

    result = predict_all(signals_list, model, threshold=args.threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    n = len(result["annotations"])
    total_cats = sum(len(a["categories"]) for a in result["annotations"])
    logger.info(
        "Predicted %d category assignments across %d trials (threshold=%s). Output: %s",
        total_cats,
        n,
        args.threshold,
        output,
    )

    # Write to AnnotationStore if --annotations-out is provided
    annotations_out = getattr(args, "annotations_out", None)
    if annotations_out:
        taxonomy_version = str(result.get("taxonomy_version", ""))
        # Enrich with trial_id from the signals_list
        enriched = []
        for sig, ann in zip(signals_list, result["annotations"]):
            enriched.append({**ann, "trial_id": sig.get("trial_id", "")})
        rows = _annotations_to_narrow_rows(
            enriched,
            annotator_type="classifier",
            annotator_identity="classifier:trained-model",
            taxonomy_version=taxonomy_version,
        )
        _write_to_annotation_store(rows, annotations_out, taxonomy_version)


def cmd_ensemble(args):
    """Run two-tier ensemble annotation (heuristic + classifier) on full corpus."""
    from agent_diagnostics.classifier import load_model
    from agent_diagnostics.ensemble import ensemble_all

    model = load_model(args.model)
    with open(args.signals) as f:
        signals_list = json.load(f)

    annotations_out = getattr(args, "annotations_out", None)
    result = ensemble_all(
        signals_list,
        model,
        classifier_threshold=args.threshold,
        classifier_min_f1=args.min_f1,
        annotations_out=annotations_out,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    n = len(result["annotations"])
    total_cats = sum(len(a["categories"]) for a in result["annotations"])
    tiers = result.get("tier_counts", {})
    logger.info(
        "Ensemble: %d assignments across %d trials "
        "(heuristic=%d, classifier=%d). Output: %s",
        total_cats,
        n,
        tiers.get("heuristic", 0),
        tiers.get("classifier", 0),
        output,
    )


def cmd_ingest(args):
    """Run filter -> extract -> enrich -> write JSONL pipeline in one command."""
    from agent_diagnostics.signals import (
        _is_excluded_path,
        _is_valid_trial,
        _load_json,
        extract_signals,
        load_manifest,
        write_jsonl,
    )

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        logger.error("runs directory not found: %s", runs_dir)
        sys.exit(1)

    # Load manifest for benchmark resolution
    suite_mapping: dict[str, str] | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            logger.error("manifest file not found: %s", manifest_path)
            sys.exit(1)
        suite_mapping = load_manifest(manifest_path)

    # Load state file for incremental mode
    state: dict[str, dict[str, object]] = {}
    state_path: Path | None = None
    if args.state:
        state_path = Path(args.state)
        if state_path.is_file():
            loaded = _load_json(state_path)
            if isinstance(loaded, dict):
                state = loaded

    # Walk runs_dir finding result.json files
    signals_list: list[dict[str, object]] = []
    skipped = 0
    for result_file in sorted(runs_dir.rglob("result.json")):
        trial_dir = result_file.parent

        if _is_excluded_path(trial_dir):
            continue

        data = _load_json(result_file)
        if data is None or not _is_valid_trial(data):
            continue

        # Incremental mode: check mtime/size
        trial_key = str(trial_dir)
        if state_path is not None:
            stat = result_file.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size
            prev = state.get(trial_key)
            if (
                prev is not None
                and prev.get("mtime") == current_mtime
                and prev.get("size") == current_size
            ):
                skipped += 1
                continue

        signals = extract_signals(
            trial_dir,
            suite_mapping=suite_mapping,
        )
        # Add trial_path for downstream traceability
        signals["trial_path"] = str(trial_dir)
        signals_list.append(signals)

        # Update state entry
        if state_path is not None:
            stat = result_file.stat()
            state[trial_key] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }

    # Write output as JSONL
    output = Path(args.output)
    write_jsonl(signals_list, output)

    # Save state file
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    suffix = f" (skipped {skipped} unchanged)" if skipped else ""
    logger.info("Ingested %d trials%s -> %s", len(signals_list), suffix, output)


def cmd_db_schema(args):
    """Emit schema of signals/annotations/manifests tables."""
    from agent_diagnostics.query import get_schema

    try:
        output = get_schema(args.data_dir, fmt=args.format)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # STDOUT: schema report is the primary user-requested output of this command.
    print(output)


def cmd_query(args):
    """Run a SQL query against observatory data via DuckDB."""
    from agent_diagnostics.query import format_table, run_query

    try:
        rows = run_query(args.sql, args.data_dir)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # STDOUT: query result table is the primary user-requested output.
    print(format_table(rows))


def cmd_export(args):
    """Export JSONL data to Parquet format with a MANIFEST.json."""
    from agent_diagnostics.export import export_parquet

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("data directory not found: %s", data_dir)
        sys.exit(1)

    fmt = args.format
    if fmt != "parquet":
        logger.error("unsupported export format: %s", fmt)
        sys.exit(1)

    try:
        manifest = export_parquet(data_dir, args.out)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    row_counts = manifest.get("row_count", {})
    total = sum(row_counts.values())
    logger.info(
        "Exported %d rows to %s (signals=%d, annotations=%d, manifests=%d)",
        total,
        args.out,
        row_counts.get("signals", 0),
        row_counts.get("annotations", 0),
        row_counts.get("manifests", 0),
    )


def cmd_manifest_refresh(args):
    """Rewrite manifests.jsonl from signals data."""
    from agent_diagnostics.export import refresh_manifests

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("data directory not found: %s", data_dir)
        sys.exit(1)

    output_path = refresh_manifests(data_dir)
    logger.info("Manifests refreshed: %s", output_path)


def cmd_pipeline_run(args):
    """Run stale stages declared in pipeline.toml."""
    from agent_diagnostics.pipeline import (
        PipelineError,
        format_summary,
        run_pipeline,
    )

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("pipeline config not found: %s", config_path)
        sys.exit(1)

    project_root = Path(args.project_root) if args.project_root else config_path.parent
    try:
        results = run_pipeline(config_path, project_root)
    except PipelineError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # STDOUT: pipeline run summary is the user-requested output of this command.
    print(format_summary(results))

    if any(r.status == "failed" for r in results):
        sys.exit(1)


def cmd_calibrate(args):
    """Compare two annotation files and emit calibration metrics.

    Reference ("ground truth") comes from ``--reference``; predictor (with
    emitted confidences) comes from ``--predictor``.  If ``--golden-dir`` is
    supplied, the golden corpus is collected into an in-memory annotation
    document and used as the reference.

    Writes ``calibration.md`` and ``calibration.json`` under ``--output-dir``.

    Permission contract: ``--output-dir`` is user-provided and its contents
    are created with the caller's umask — the caller is responsible for
    choosing an appropriately-permissioned directory. Note that
    ``calibration.json`` and ``calibration.md`` may contain corpus-derived
    content (category names, confidence distributions, trial paths), so on
    multi-user shared hosts callers should point ``--output-dir`` at a
    private directory. The internal temp directory used when composing
    ``--golden-dir`` into a reference document is always owner-only
    (``0o700``) and the composed ``reference.json`` is ``0o600``, regardless
    of caller umask.
    """
    import tempfile

    from agent_diagnostics.calibrate import compare_annotations, format_markdown

    predictor_path = Path(args.predictor)
    if not predictor_path.is_file():
        logger.error("predictor annotations not found: %s", predictor_path)
        sys.exit(1)

    # Reference: either a plain annotations file or the golden corpus dir.
    # When the golden corpus is used we materialise the composed document in
    # an internal temp directory that is always owner-only regardless of
    # interpreter. CPython's ``mkdtemp`` already creates the directory with
    # mode 0o700 on POSIX, and the file is written inside it — the explicit
    # chmod calls below are a portability hedge (not a race fix) and pin the
    # file's own mode to 0o600 so that if the file ever escapes the temp dir
    # (backup sweep, future refactor) it is still owner-readable only.  The
    # window between open() and the post-write chmod is not exploitable here
    # because the enclosing directory is 0o700, so no other user can traverse
    # to the file during that window.
    tmp_dir: tempfile.TemporaryDirectory | None = None
    reference_path: Path
    if args.golden_dir:
        golden_dir_path = Path(args.golden_dir)
        if not golden_dir_path.is_dir():
            logger.error("golden corpus directory not found: %s", golden_dir_path)
            sys.exit(1)
        golden_doc = _collect_golden_corpus(golden_dir_path)
        tmp_dir = tempfile.TemporaryDirectory(prefix="observatory-calibrate-")
        os.chmod(tmp_dir.name, 0o700)
        reference_path = Path(tmp_dir.name) / "reference.json"
        with open(reference_path, "w", encoding="utf-8") as f:
            json.dump(golden_doc, f)
        os.chmod(reference_path, 0o600)
    elif args.reference:
        reference_path = Path(args.reference)
        if not reference_path.is_file():
            logger.error("reference annotations not found: %s", reference_path)
            sys.exit(1)
    else:
        logger.error("provide --reference or --golden-dir for the label source")
        sys.exit(1)

    try:
        summary = compare_annotations(predictor_path, reference_path)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "calibration.md"
    md_path.write_text(format_markdown(summary))

    json_path = output_dir / "calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    shared_trials = summary.get("shared_trials", 0)
    logger.info(
        "Calibration report: %s (markdown), %s (json). Shared trials: %d",
        md_path,
        json_path,
        shared_trials,
    )
    if shared_trials == 0:
        # Most common cause: the predictor was annotated against a runs/
        # directory (trial_path = filesystem path) while the golden corpus
        # uses trial_id_short dir names. The join silently produces an
        # empty report — surface the likely cause so the user doesn't have
        # to guess.
        logger.warning(
            "shared_trials=0: predictor and reference have no overlapping "
            "trial_path values. If --golden-dir was used, the predictor's "
            "trial_path fields must match the golden-corpus directory names "
            "(typically trial_id_short hashes), not filesystem paths from "
            "`ingest --runs-dir` / `annotate --signals` against a runs tree."
        )


def _collect_golden_corpus(dir_path: Path) -> dict[str, Any]:
    """Compose per-trial ``expected_annotations.json`` files into one document.

    Each subdirectory of *dir_path* is treated as a trial; its
    ``expected_annotations.json`` contributes one annotation record using the
    directory name as ``trial_path``.
    """
    if not dir_path.is_dir():
        raise FileNotFoundError(f"golden corpus directory not found: {dir_path}")

    annotations: list[dict[str, Any]] = []
    for trial_dir in sorted(dir_path.iterdir()):
        # Reject symlinks defensively: a symlinked trial dir could point
        # outside `dir_path`, letting a malicious --golden-dir read
        # arbitrary `expected_annotations.json` files from the filesystem.
        # Path.is_dir(follow_symlinks=False) is Python 3.12+; use an
        # explicit is_symlink() check for 3.10/3.11 compatibility.
        if trial_dir.is_symlink() or not trial_dir.is_dir():
            continue
        ann_file = trial_dir / "expected_annotations.json"
        if ann_file.is_symlink() or not ann_file.is_file():
            continue
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)
        categories = [
            {
                "name": c.get("name", ""),
                "confidence": c.get("confidence", 1.0),
                "evidence": c.get("evidence", ""),
            }
            for c in data.get("categories", [])
            if c.get("name")
        ]
        annotations.append(
            {
                "trial_path": trial_dir.name,
                "categories": categories,
            }
        )
    return {"annotations": annotations}


def cmd_validate(args):
    """Validate annotation files against schema and taxonomy."""
    import jsonschema

    from agent_diagnostics.taxonomy import valid_category_names

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        logger.error("annotations file not found: %s", annotations_path)
        sys.exit(1)

    # Load annotations
    try:
        with open(annotations_path) as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("invalid JSON: %s", e)
        sys.exit(1)

    errors = []

    # Validate against JSON Schema
    schema_path = Path(__file__).parent / "annotation_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=annotations, schema=schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")

    # Validate category names against taxonomy
    valid_names = valid_category_names()
    for i, ann in enumerate(annotations.get("annotations", [])):
        for cat in ann.get("categories", []):
            name = cat.get("name", "")
            if name not in valid_names:
                errors.append(
                    f"Annotation [{i}] ({ann.get('task_id', '?')}): "
                    f"unknown category '{name}'"
                )

    if errors:
        logger.error("Validation FAILED with %d error(s):", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        sys.exit(1)

    n_annotations = len(annotations.get("annotations", []))
    logger.info("Validation passed: %d annotations, all valid.", n_annotations)
    sys.exit(0)


_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _lookup_level_by_name(name: str) -> int | None:
    """Return the numeric logging level for *name*, or ``None`` if unregistered.

    Note that ``NOTSET`` is a registered name and returns ``0`` — callers that
    want to reject sub-DEBUG levels must gate on ``value >= logging.DEBUG``
    themselves; ``None`` here means "no such name", not "unusable value".

    Prefers :func:`logging.getLevelNamesMapping` (added in Python 3.11) over
    the legacy ``logging.getLevelName(str)`` path, which is documented as
    a historical mistake. Falls back to ``getLevelName`` on 3.10 so the
    module still imports there.
    """
    mapping_fn = getattr(logging, "getLevelNamesMapping", None)
    if mapping_fn is not None:
        return mapping_fn().get(name)
    # Python 3.10 fallback. getLevelName returns the input string unchanged
    # for unknown names, so guard with isinstance to only accept ints.
    legacy = logging.getLevelName(name)
    return legacy if isinstance(legacy, int) else None


def _resolve_log_level(verbose: int, quiet: bool) -> int:
    """Resolve the effective root log level from CLI flags and environment.

    Precedence: ``-q`` > ``-v`` > ``AGENT_DIAGNOSTICS_LOG_LEVEL`` > INFO.
    Unknown or sub-DEBUG env values fall back to INFO so a malformed or
    overly permissive environment never enables third-party library logging
    (which can include ``Authorization`` headers from ``httpx``).
    """
    if quiet:
        return logging.WARNING
    if verbose >= 1:
        return logging.DEBUG
    env = os.environ.get("AGENT_DIAGNOSTICS_LOG_LEVEL", "").strip().upper()
    if env:
        numeric = _lookup_level_by_name(env)
        if numeric is not None and numeric >= logging.DEBUG:
            return numeric
    return logging.INFO


def _configure_logging(level: int) -> None:
    """Install a single stderr handler on the root logger.

    ``force=True`` replaces any pre-existing handlers so repeated ``main``
    calls during tests don't accumulate duplicates; pytest's ``caplog``
    fixture attaches its own handler after this runs and therefore still
    captures records.
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        stream=sys.stderr,
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="observatory",
        description="Agent Reliability Observatory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v: DEBUG). Default level is INFO.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only emit WARNING and above (overrides -v).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # extract
    p_extract = subparsers.add_parser(
        "extract", help="Extract signals from trial directories"
    )
    p_extract.add_argument(
        "--runs-dir", required=True, help="Path to runs/_raw directory"
    )
    p_extract.add_argument("--output", required=True, help="Output signals JSON file")
    p_extract.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory for the content-hash extraction cache "
        "(e.g. 'data/.extract-cache'). Warm re-extractions skip unchanged trials.",
    )
    p_extract.set_defaults(func=cmd_extract)

    # annotate
    p_annotate = subparsers.add_parser(
        "annotate", help="Generate heuristic annotations from signals"
    )
    p_annotate.add_argument("--signals", required=True, help="Input signals JSON file")
    p_annotate.add_argument(
        "--output", required=True, help="Output annotations JSON file"
    )
    p_annotate.add_argument(
        "--annotations-out",
        default=None,
        help="Write narrow-tall rows to this JSONL via AnnotationStore (default: None)",
    )
    p_annotate.set_defaults(func=cmd_annotate)

    # report
    p_report = subparsers.add_parser(
        "report", help="Generate reliability report from annotations"
    )
    p_report.add_argument(
        "--annotations", required=True, help="Input annotations JSON file"
    )
    # Canonical flag is --output-dir (dir-writing commands use --output-dir;
    # file-writing commands use --output). --output is kept as a deprecated
    # alias for 0.8.x compatibility, slated for removal in 1.0.
    report_out_group = p_report.add_mutually_exclusive_group(required=True)
    report_out_group.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for report files",
    )
    report_out_group.add_argument(
        "--output",
        default=None,
        help="[deprecated: use --output-dir] Output directory for report files",
    )
    p_report.set_defaults(func=cmd_report)

    # llm-annotate
    p_llm = subparsers.add_parser(
        "llm-annotate", help="Generate LLM-assisted annotations for a sample of trials"
    )
    p_llm.add_argument("--signals", required=True, help="Input signals JSON file")
    p_llm.add_argument("--output", required=True, help="Output annotations JSON file")
    p_llm.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of trials to sample (default: 50)",
    )
    p_llm.add_argument(
        "--model",
        default="haiku",
        help="Model alias: haiku, sonnet, opus (default: haiku)",
    )
    p_llm.add_argument(
        "--backend",
        default="claude-code",
        choices=["claude-code", "api", "batch"],
        help="LLM backend: 'claude-code' uses the claude CLI (default), "
        "'api' uses the Anthropic SDK, 'batch' uses the Message Batches API "
        "(50%% cheaper, no rate limits). Both api/batch require ANTHROPIC_API_KEY.",
    )
    p_llm.add_argument(
        "--annotations-out",
        default=None,
        help="Write narrow-tall rows to this JSONL via AnnotationStore (default: None)",
    )
    p_llm.set_defaults(func=cmd_llm_annotate)

    # train
    p_train = subparsers.add_parser(
        "train", help="Train classifiers from LLM-labeled data"
    )
    p_train.add_argument(
        "--labels", required=True, help="LLM annotation JSON (training labels)"
    )
    p_train.add_argument(
        "--signals", required=True, help="Full signals JSON (features)"
    )
    p_train.add_argument("--output", required=True, help="Output model JSON file")
    p_train.add_argument(
        "--min-positive",
        type=int,
        default=3,
        help="Min positive examples per category (default: 3)",
    )
    p_train.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate (default: 0.1)"
    )
    p_train.add_argument(
        "--epochs", type=int, default=300, help="Training epochs (default: 300)"
    )
    p_train.add_argument(
        "--eval", action="store_true", help="Evaluate on training data after training"
    )
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = subparsers.add_parser(
        "predict", help="Predict categories using trained classifier"
    )
    p_predict.add_argument("--model", required=True, help="Trained model JSON file")
    p_predict.add_argument(
        "--signals", required=True, help="Signals JSON to predict on"
    )
    p_predict.add_argument(
        "--output", required=True, help="Output annotations JSON file"
    )
    p_predict.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold (default: 0.5)",
    )
    p_predict.add_argument(
        "--annotations-out",
        default=None,
        help="Write narrow-tall rows to this JSONL via AnnotationStore (default: None)",
    )
    p_predict.set_defaults(func=cmd_predict)

    # ensemble
    p_ensemble = subparsers.add_parser(
        "ensemble", help="Run two-tier ensemble annotation (heuristic + classifier)"
    )
    p_ensemble.add_argument("--signals", required=True, help="Signals JSON file")
    p_ensemble.add_argument(
        "--model", required=True, help="Trained classifier model JSON"
    )
    p_ensemble.add_argument(
        "--output", required=True, help="Output annotations JSON file"
    )
    p_ensemble.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classifier prediction threshold (default: 0.5)",
    )
    p_ensemble.add_argument(
        "--min-f1",
        type=float,
        default=0.7,
        help="Minimum train accuracy to use classifier (default: 0.7)",
    )
    p_ensemble.add_argument(
        "--annotations-out",
        default=None,
        help="Write narrow-tall rows to this JSONL via AnnotationStore (default: None)",
    )
    p_ensemble.set_defaults(func=cmd_ensemble)

    # ingest
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Run filter -> extract -> enrich -> write JSONL pipeline",
    )
    p_ingest.add_argument("--runs-dir", required=True, help="Path to runs directory")
    p_ingest.add_argument("--output", required=True, help="Output JSONL file")
    p_ingest.add_argument(
        "--manifest", default=None, help="MANIFEST.json for benchmark resolution"
    )
    p_ingest.add_argument(
        "--state", default=None, help="State JSON file for incremental mode"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # calibrate
    p_cal = subparsers.add_parser(
        "calibrate",
        help="Report ECE/Brier/reliability per category for a predictor vs reference",
    )
    p_cal.add_argument(
        "--predictor",
        required=True,
        help="Annotation JSON whose categories carry emitted confidences "
        "(typically the heuristic or LLM annotator output being scored)",
    )
    cal_ref_group = p_cal.add_mutually_exclusive_group(required=True)
    cal_ref_group.add_argument(
        "--reference",
        default=None,
        help="Ground-truth annotation JSON file (labels treated as observed)",
    )
    cal_ref_group.add_argument(
        "--golden-dir",
        default=None,
        help="Golden corpus directory containing per-trial "
        "expected_annotations.json files; used as the reference",
    )
    p_cal.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for calibration.md + calibration.json",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate annotation files")
    p_validate.add_argument(
        "--annotations", required=True, help="Annotation JSON file to validate"
    )
    p_validate.set_defaults(func=cmd_validate)

    # query
    p_query = subparsers.add_parser(
        "query", help="Run a SQL query against observatory data via DuckDB"
    )
    p_query.add_argument("sql", help="SQL query string")
    p_query.add_argument(
        "--data-dir",
        default="data/",
        help="Root data directory (default: data/)",
    )
    p_query.set_defaults(func=cmd_query)

    # export
    p_export = subparsers.add_parser(
        "export", help="Export JSONL data to Parquet with manifest"
    )
    p_export.add_argument(
        "--format",
        default="parquet",
        choices=["parquet"],
        help="Export format (default: parquet)",
    )
    p_export.add_argument(
        "--out",
        default="data/export/",
        help="Output directory (default: data/export/)",
    )
    p_export.add_argument(
        "--data-dir",
        default="data/",
        help="Root data directory (default: data/)",
    )
    p_export.set_defaults(func=cmd_export)

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Pipeline management commands")
    pipeline_sub = p_pipeline.add_subparsers(
        dest="pipeline_command", help="Pipeline subcommands"
    )
    p_pipeline_run = pipeline_sub.add_parser(
        "run", help="Run stale stages declared in pipeline.toml"
    )
    p_pipeline_run.add_argument(
        "--config",
        default="pipeline.toml",
        help="Path to pipeline config (default: pipeline.toml at project root)",
    )
    p_pipeline_run.add_argument(
        "--project-root",
        default=None,
        help="Project root for resolving relative input/output paths (default: dir of --config)",
    )
    p_pipeline_run.set_defaults(func=cmd_pipeline_run)

    # manifest
    p_manifest = subparsers.add_parser("manifest", help="Manifest management commands")
    manifest_sub = p_manifest.add_subparsers(
        dest="manifest_command", help="Manifest subcommands"
    )
    p_manifest_refresh = manifest_sub.add_parser(
        "refresh", help="Rewrite manifests.jsonl from signals data"
    )
    p_manifest_refresh.add_argument(
        "--data-dir",
        default="data/",
        help="Root data directory (default: data/)",
    )
    p_manifest_refresh.set_defaults(func=cmd_manifest_refresh)

    # db
    p_db = subparsers.add_parser("db", help="Database introspection commands")
    db_sub = p_db.add_subparsers(dest="db_command", help="Database subcommands")
    p_db_schema = db_sub.add_parser(
        "schema", help="Emit schema of signals/annotations/manifests tables"
    )
    p_db_schema.add_argument(
        "--format",
        default="markdown",
        choices=["json", "markdown"],
        help="Output format (default: markdown)",
    )
    p_db_schema.add_argument(
        "--data-dir",
        default="data/",
        help="Root data directory (default: data/)",
    )
    p_db_schema.set_defaults(func=cmd_db_schema)

    args = parser.parse_args()
    _configure_logging(_resolve_log_level(args.verbose, args.quiet))
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    if not hasattr(args, "func"):
        # Sub-subcommand not provided (e.g. 'observatory manifest' without 'refresh')
        parser.parse_args([args.command, "--help"])
        sys.exit(0)
    args.func(args)
