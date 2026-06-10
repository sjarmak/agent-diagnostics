"""Data access subcommands: db schema, query, export, manifest, pipeline."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    from agent_diagnostics.query import format_rows, run_query

    try:
        rows = run_query(args.sql, args.data_dir)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Absent attribute (programmatic Namespace callers) falls back to 'table'
    # so we preserve the pre-flag CLI contract for existing consumers.
    fmt = getattr(args, "format", None) or "table"
    # STDOUT: query result is the primary user-requested output.
    print(format_rows(rows, fmt))


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
