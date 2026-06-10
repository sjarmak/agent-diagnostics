"""Argument parser construction and process entry point."""

import argparse
import logging
import os
import sys

from agent_diagnostics.cli.annotate import cmd_annotate, cmd_llm_annotate
from agent_diagnostics.cli.classify import cmd_blend, cmd_ensemble, cmd_predict, cmd_train
from agent_diagnostics.cli.data import (
    cmd_db_schema,
    cmd_export,
    cmd_manifest_refresh,
    cmd_pipeline_run,
    cmd_query,
)
from agent_diagnostics.cli.ingest import cmd_extract, cmd_ingest
from agent_diagnostics.cli.reporting import (
    cmd_agreement,
    cmd_calibrate,
    cmd_report,
    cmd_validate,
)

logger = logging.getLogger(__name__)

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
    p_extract = subparsers.add_parser("extract", help="Extract signals from trial directories")
    p_extract.add_argument("--runs-dir", required=True, help="Path to runs/_raw directory")
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
    p_annotate.add_argument("--output", required=True, help="Output annotations JSON file")
    p_annotate.add_argument(
        "--annotations-out",
        default=None,
        help="Write narrow-tall rows to this JSONL via AnnotationStore (default: None)",
    )
    p_annotate.set_defaults(func=cmd_annotate)

    # report
    p_report = subparsers.add_parser("report", help="Generate reliability report from annotations")
    p_report.add_argument("--annotations", required=True, help="Input annotations JSON file")
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

    # blend
    p_blend = subparsers.add_parser(
        "blend",
        help="Blend heuristic + LLM annotations into a training set for `train`",
    )
    p_blend.add_argument(
        "--heuristic", required=True, help="Heuristic annotation JSON (corpus-scale)"
    )
    p_blend.add_argument("--llm", required=True, help="LLM annotation JSON (sampled)")
    p_blend.add_argument("--output", required=True, help="Output blended annotation JSON")
    p_blend.add_argument(
        "--calibration",
        default=None,
        help="Calibration JSON from `calibrate`; per-category F1 decides which "
        "heuristic categories to trust (default: trust categories with "
        "signal_dependencies in the taxonomy)",
    )
    p_blend.add_argument(
        "--trust-threshold",
        type=float,
        default=0.7,
        help="Minimum calibration F1 to trust a heuristic category (default: 0.7)",
    )
    p_blend.add_argument(
        "--max-heuristic-samples",
        type=int,
        default=2000,
        help="Cap on heuristic-only trials so they don't swamp LLM labels "
        "(default: 2000); a warning is logged when the cap drops trials",
    )
    p_blend.set_defaults(func=cmd_blend)

    # train
    p_train = subparsers.add_parser("train", help="Train classifiers from LLM-labeled data")
    p_train.add_argument("--labels", required=True, help="LLM annotation JSON (training labels)")
    p_train.add_argument("--signals", required=True, help="Full signals JSON (features)")
    p_train.add_argument("--output", required=True, help="Output model JSON file")
    p_train.add_argument(
        "--min-positive",
        type=int,
        default=3,
        help="Min positive examples per category (default: 3)",
    )
    p_train.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    p_train.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    p_train.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for held-out eval_f1/cv_ece "
        "(default: 5; clamped per category to the minority class count)",
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
    p_predict.add_argument("--signals", required=True, help="Signals JSON to predict on")
    p_predict.add_argument("--output", required=True, help="Output annotations JSON file")
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
    p_ensemble.add_argument("--model", required=True, help="Trained classifier model JSON")
    p_ensemble.add_argument("--output", required=True, help="Output annotations JSON file")
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
        help="Minimum held-out cross-validated F1 to trust a classifier category (default: 0.7)",
    )
    p_ensemble.add_argument(
        "--max-ece",
        type=float,
        default=None,
        help="Optional maximum cross-validated ECE to trust a classifier "
        "category; categories above it are excluded even when F1 passes "
        "(default: no ECE gate)",
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
    p_ingest.add_argument("--state", default=None, help="State JSON file for incremental mode")
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

    # agreement
    p_agree = subparsers.add_parser(
        "agreement",
        help="Report pairwise inter-annotator agreement (Cohen's kappa) per category",
    )
    p_agree.add_argument(
        "--annotations",
        required=True,
        help="Narrow-tall annotations JSONL (the --annotations-out store)",
    )
    p_agree.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for agreement.md + agreement.json",
    )
    p_agree.set_defaults(func=cmd_agreement)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate annotation files")
    p_validate.add_argument(
        "--annotations", required=True, help="Annotation JSON file to validate"
    )
    p_validate.set_defaults(func=cmd_validate)

    # query
    from agent_diagnostics.query import OUTPUT_FORMATS as _QUERY_OUTPUT_FORMATS

    p_query = subparsers.add_parser(
        "query", help="Run a SQL query against observatory data via DuckDB"
    )
    p_query.add_argument("sql", help="SQL query string")
    p_query.add_argument(
        "--data-dir",
        default="data/",
        help="Root data directory (default: data/)",
    )
    p_query.add_argument(
        "--format",
        default="table",
        choices=_QUERY_OUTPUT_FORMATS,
        help="Output format: 'table' (human-readable, default), "
        "'json' (list of objects), 'jsonl' (one object per line, stream-friendly), "
        "or 'csv' (RFC 4180 with header row).",
    )
    p_query.set_defaults(func=cmd_query)

    # export
    p_export = subparsers.add_parser("export", help="Export JSONL data to Parquet with manifest")
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
    pipeline_sub = p_pipeline.add_subparsers(dest="pipeline_command", help="Pipeline subcommands")
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
    manifest_sub = p_manifest.add_subparsers(dest="manifest_command", help="Manifest subcommands")
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
