"""CLI entrypoint for the Agent Reliability Observatory."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    print(
        f"AnnotationStore: wrote {count} rows to {annotations_out}",
        file=sys.stderr,
    )


def cmd_extract(args):
    """Extract signals from trial directories."""
    from agent_diagnostics.signals import extract_all, write_output

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"Error: runs directory not found: {runs_dir}", file=sys.stderr)
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
    print(f"Extracted signals from {len(signals)} trials{suffix}", file=sys.stderr)


def cmd_annotate(args):
    """Generate heuristic annotations from extracted signals."""
    from datetime import datetime, timezone

    from agent_diagnostics.annotator import annotate_trial
    from agent_diagnostics.signals import load_signals, write_output
    from agent_diagnostics.taxonomy import load_taxonomy

    signals_path = Path(args.signals)
    if not signals_path.is_file():
        print(f"Error: signals file not found: {signals_path}", file=sys.stderr)
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
    print(
        f"Annotated {len(annotation_list)} trials "
        f"with {total_categories} total category assignments",
        file=sys.stderr,
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
    """Generate reliability report from annotations."""
    from agent_diagnostics.report import generate_report

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        print(f"Error: annotations file not found: {annotations_path}", file=sys.stderr)
        sys.exit(1)

    with open(annotations_path) as f:
        annotations = json.load(f)

    output_dir = Path(args.output)
    md_path, json_path = generate_report(annotations, output_dir)

    print(f"Report written to {md_path} and {json_path}", file=sys.stderr)


def cmd_llm_annotate(args):
    """Generate LLM-assisted annotations for a sample of trials."""
    import random
    from datetime import datetime, timezone

    from agent_diagnostics.llm_annotator import annotate_batch
    from agent_diagnostics.taxonomy import load_taxonomy

    signals_path = Path(args.signals)
    if not signals_path.is_file():
        print(f"Error: signals file not found: {signals_path}", file=sys.stderr)
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
        print("Error: no trials with trajectory files found", file=sys.stderr)
        sys.exit(1)

    backend = args.backend
    sample_size = min(args.sample_size, len(has_trajectory))
    sampled = random.sample(has_trajectory, sample_size)
    print(
        f"Sampled {sample_size} trials (from {len(has_trajectory)} with trajectories)",
        file=sys.stderr,
    )
    print(f"Backend: {backend}, Model: {args.model}", file=sys.stderr)

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
    for sig, cats in zip(sampled, batch_results):
        reward_val = sig.get("reward")
        annotation = {
            "task_id": sig.get("task_id") or "unknown",
            "trial_path": sig.get("trial_path") or "",
            "reward": float(reward_val) if reward_val is not None else 0.0,
            "passed": (
                bool(sig.get("passed")) if sig.get("passed") is not None else False
            ),
            "categories": cats,
            "annotated_at": now,
        }
        for key in ("config_name", "benchmark", "model"):
            if sig.get(key):
                annotation[key] = sig[key]
        annotations.append(annotation)

    result = {
        "schema_version": "observatory-annotation-v1",
        "taxonomy_version": str(taxonomy["version"]),
        "generated_at": now,
        "annotator": {
            "type": "llm",
            "identity": f"observatory.llm_annotator model={args.model} backend={backend}",
        },
        "annotations": annotations,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    total_categories = sum(len(a["categories"]) for a in annotations)
    summary = (
        f"Done: {len(annotations)}/{sample_size} trials annotated "
        f"with {total_categories} category assignments"
    )
    summary += f". Output: {output}"
    print(summary, file=sys.stderr)

    # Write to AnnotationStore if --annotations-out is provided
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
    print(
        f"Trained {n_clf} classifiers on {model['training_samples']} samples "
        f"({n_skip} categories skipped, < {model['min_positive']} positive)",
        file=sys.stderr,
    )
    for cat, clf in sorted(model["classifiers"].items()):
        print(
            f"  {cat}: {clf['positive_count']}/{clf['total_count']} positive, "
            f"train_acc={clf['train_accuracy']:.2f}",
            file=sys.stderr,
        )

    # If --eval is provided, evaluate on the same data (quick sanity check)
    if args.eval:
        eval_results = evaluate(model, args.labels, args.signals)
        print("\n" + format_eval_markdown(eval_results, model), file=sys.stderr)

    print(f"Model saved to {output}", file=sys.stderr)


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
    print(
        f"Predicted {total_cats} category assignments across {n} trials "
        f"(threshold={args.threshold}). Output: {output}",
        file=sys.stderr,
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
    print(
        f"Ensemble: {total_cats} assignments across {n} trials "
        f"(heuristic={tiers.get('heuristic', 0)}, classifier={tiers.get('classifier', 0)}). "
        f"Output: {output}",
        file=sys.stderr,
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
        print(f"Error: runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Load manifest for benchmark resolution
    suite_mapping: dict[str, str] | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"Error: manifest file not found: {manifest_path}", file=sys.stderr)
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

    print(
        f"Ingested {len(signals_list)} trials"
        + (f" (skipped {skipped} unchanged)" if skipped else "")
        + f" -> {output}",
        file=sys.stderr,
    )


def cmd_db_schema(args):
    """Emit schema of signals/annotations/manifests tables."""
    from agent_diagnostics.query import get_schema

    try:
        output = get_schema(args.data_dir, fmt=args.format)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(output)


def cmd_query(args):
    """Run a SQL query against observatory data via DuckDB."""
    from agent_diagnostics.query import format_table, run_query

    try:
        rows = run_query(args.sql, args.data_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_table(rows))


def cmd_export(args):
    """Export JSONL data to Parquet format with a MANIFEST.json."""
    from agent_diagnostics.export import export_parquet

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    if fmt != "parquet":
        print(f"Error: unsupported export format: {fmt}", file=sys.stderr)
        sys.exit(1)

    try:
        manifest = export_parquet(data_dir, args.out)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    row_counts = manifest.get("row_count", {})
    total = sum(row_counts.values())
    print(
        f"Exported {total} rows to {args.out} "
        f"(signals={row_counts.get('signals', 0)}, "
        f"annotations={row_counts.get('annotations', 0)}, "
        f"manifests={row_counts.get('manifests', 0)})",
        file=sys.stderr,
    )


def cmd_manifest_refresh(args):
    """Rewrite manifests.jsonl from signals data."""
    from agent_diagnostics.export import refresh_manifests

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = refresh_manifests(data_dir)
    print(f"Manifests refreshed: {output_path}", file=sys.stderr)


def cmd_pipeline_run(args):
    """Run stale stages declared in pipeline.toml."""
    from agent_diagnostics.pipeline import (
        PipelineError,
        format_summary,
        run_pipeline,
    )

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: pipeline config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(args.project_root) if args.project_root else config_path.parent
    try:
        results = run_pipeline(config_path, project_root)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_summary(results))

    if any(r.status == "failed" for r in results):
        sys.exit(1)


def cmd_validate(args):
    """Validate annotation files against schema and taxonomy."""
    import jsonschema

    from agent_diagnostics.taxonomy import valid_category_names

    annotations_path = Path(args.annotations)
    if not annotations_path.is_file():
        print(f"Error: annotations file not found: {annotations_path}", file=sys.stderr)
        sys.exit(1)

    # Load annotations
    try:
        with open(annotations_path) as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
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
        print(f"Validation FAILED with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    n_annotations = len(annotations.get("annotations", []))
    print(
        f"Validation passed: {n_annotations} annotations, all valid.",
        file=sys.stderr,
    )
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        prog="observatory",
        description="Agent Reliability Observatory",
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
    p_report.add_argument(
        "--output", required=True, help="Output directory for report files"
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
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    if not hasattr(args, "func"):
        # Sub-subcommand not provided (e.g. 'observatory manifest' without 'refresh')
        parser.parse_args([args.command, "--help"])
        sys.exit(0)
    args.func(args)
