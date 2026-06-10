"""Ingestion subcommands: extract and ingest."""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
            f" (cache: {stats['hits']} hits, {stats['misses']} misses, {stats['entries']} entries)"
        )
    logger.info("Extracted signals from %d trials%s", len(signals), suffix)


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
