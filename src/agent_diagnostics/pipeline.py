"""Pipeline runner for the Agent Reliability Observatory (PRD NH-2).

Loads a ``pipeline.toml`` at the project root declaring stages (``extract``,
``annotate``, ``report``, ...), each with ``inputs``, ``outputs``, and a
``command`` template.  ``observatory pipeline run`` walks the stages in the
declared order and runs only stages whose outputs are stale.

Staleness is content-based, not mtime-based: after a stage runs
successfully, a SHA-256 fingerprint of its input files is recorded in
``.pipeline-state.json`` at the project root.  A stage is stale when any
declared output is missing, when no fingerprint has been recorded yet, or
when the current input content no longer matches the recorded fingerprint.
mtimes are deliberately ignored — they lie under ``git checkout``,
``cp -p``, and CI cache restores, which previously let a stage silently
skip on changed inputs.

Example ``pipeline.toml``::

    [[stage]]
    name = "extract"
    inputs = ["runs/"]
    outputs = ["data/signals.jsonl"]
    command = "python -m agent_diagnostics.cli extract --runs-dir runs --output data/signals.jsonl"

    [[stage]]
    name = "annotate"
    inputs = ["data/signals.jsonl"]
    outputs = ["data/annotations.jsonl"]
    command = "python -m agent_diagnostics.cli annotate --signals data/signals.jsonl --output data/annotations.jsonl"
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".pipeline-state.json"


class PipelineError(Exception):
    """Raised when a pipeline file is malformed or a stage command fails."""


@dataclass(frozen=True)
class Stage:
    """A single declarative pipeline stage."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    command: str


@dataclass(frozen=True)
class StageResult:
    """Outcome of evaluating (and possibly running) a stage."""

    name: str
    status: str  # "ran", "skipped", "failed"
    reason: str = ""


def _load_tomllib() -> Any:
    """Return a TOML parser module (``tomllib`` on 3.11+, fallback ``tomli``)."""
    try:
        import tomllib  # type: ignore[import-not-found]

        return tomllib
    except ImportError:  # pragma: no cover - exercised only on pre-3.11
        import tomli as tomllib  # type: ignore[import-not-found]

        return tomllib


def load_pipeline(path: Path) -> list[Stage]:
    """Parse *path* and return the list of declared stages.

    Raises
    ------
    PipelineError
        On malformed TOML or missing required stage fields.
    """
    tomllib = _load_tomllib()
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"pipeline config not found: {path}")

    with path.open("rb") as fh:
        try:
            data = tomllib.load(fh)
        except Exception as exc:
            raise PipelineError(f"failed to parse {path}: {exc}") from exc

    raw_stages = data.get("stage")
    if not isinstance(raw_stages, list):
        raise PipelineError(
            f"{path}: missing '[[stage]]' array; expected one or more [[stage]] tables"
        )

    stages: list[Stage] = []
    seen_names: set[str] = set()
    for idx, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise PipelineError(f"{path}: stage #{idx} is not a table")
        for required in ("name", "inputs", "outputs", "command"):
            if required not in raw:
                raise PipelineError(f"{path}: stage #{idx} missing required field {required!r}")
        name = str(raw["name"])
        if name in seen_names:
            raise PipelineError(f"{path}: duplicate stage name {name!r}")
        seen_names.add(name)

        inputs = tuple(str(v) for v in raw["inputs"])
        outputs = tuple(str(v) for v in raw["outputs"])
        command = str(raw["command"])
        stages.append(Stage(name=name, inputs=inputs, outputs=outputs, command=command))
    return stages


def _iter_files(path: Path) -> list[Path]:
    """Return *path* itself when it is a file, or its files recursively."""
    if path.is_dir():
        return sorted(child for child in path.rglob("*") if child.is_file())
    if path.is_file():
        return [path]
    return []


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stage_input_fingerprint(stage: Stage, project_root: Path) -> str | None:
    """SHA-256 fingerprint over the content of *stage*'s input files.

    The digest covers each file's project-relative path and content hash,
    so renames, additions, deletions, and edits all change the fingerprint
    while mtime churn does not.  Returns ``None`` when no input exists.
    """
    root = Path(project_root)
    entries: list[str] = []
    for rel in sorted(stage.inputs):
        for f in _iter_files(root / rel):
            try:
                rel_name = f.relative_to(root).as_posix()
            except ValueError:  # input declared outside the project root
                rel_name = str(f)
            entries.append(f"{rel_name}\x00{_hash_file(f)}")
    if not entries:
        return None
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def load_state(project_root: Path) -> dict[str, Any]:
    """Read ``.pipeline-state.json`` under *project_root* (empty on absence).

    A malformed state file is reported and treated as empty — every stage
    with inputs then re-runs once and the file is rewritten, which is safe
    because stages are re-runnable by contract.
    """
    path = Path(project_root) / _STATE_FILENAME
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("ignoring malformed pipeline state %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("ignoring malformed pipeline state %s: not an object", path)
        return {}
    return data


def save_state(project_root: Path, state: dict[str, Any]) -> None:
    path = Path(project_root) / _STATE_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def is_stale(
    stage: Stage,
    project_root: Path,
    state: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Decide whether *stage* needs to run.

    Returns ``(stale, reason)``.  A stage is stale when:

    - Any declared output is missing, or
    - No input fingerprint has been recorded for it in *state*, or
    - The recorded input fingerprint no longer matches the input content.

    *state* is the mapping loaded by :func:`load_state`; ``None`` behaves
    like an empty state (every stage with inputs is stale).
    """
    stale, reason, _ = _evaluate_stage(stage, project_root, state or {})
    return stale, reason


def _evaluate_stage(
    stage: Stage,
    project_root: Path,
    state: dict[str, Any],
) -> tuple[bool, str, str | None]:
    """``is_stale`` plus the computed input fingerprint (for state updates)."""
    root = Path(project_root)
    output_paths = [root / p for p in stage.outputs]

    missing_outputs = [str(p) for p in output_paths if not p.exists()]
    if missing_outputs:
        return True, f"missing output(s): {', '.join(missing_outputs)}", None

    fingerprint = stage_input_fingerprint(stage, root)
    if fingerprint is None:
        # No inputs exist — either declarative error or pre-flight stage.
        # Treat as not-stale so we don't endlessly re-run.
        return False, "no inputs present on disk", None

    entry = state.get(stage.name)
    recorded = entry.get("inputs_fingerprint") if isinstance(entry, dict) else None
    if recorded is None:
        return True, "no recorded input fingerprint", fingerprint
    if recorded != fingerprint:
        return True, "input content changed", fingerprint
    return False, "up-to-date", fingerprint


def run_pipeline(
    path: Path,
    project_root: Path,
    *,
    runner: Any = None,
) -> list[StageResult]:
    """Run every stale stage in *path* under *project_root*.

    Parameters
    ----------
    runner:
        Optional callable ``(command: str, cwd: Path) -> int`` used to execute
        commands.  Defaults to :func:`_default_runner` which invokes
        :func:`subprocess.run` with ``shell=False`` and ``shlex.split``.
        Tests inject a stub runner.
    """
    stages = load_pipeline(path)
    root = Path(project_root)
    runner = runner or _default_runner
    state = load_state(root)

    results: list[StageResult] = []
    for stage in stages:
        # Evaluated per stage at its turn (not up front) so a stage sees the
        # input content its upstream stages just produced.
        stale, reason, _ = _evaluate_stage(stage, root, state)
        if not stale:
            results.append(StageResult(name=stage.name, status="skipped", reason=reason))
            continue
        returncode = runner(stage.command, root)
        if returncode == 0:
            results.append(StageResult(name=stage.name, status="ran", reason=reason))
            # Re-fingerprint after the run: the command may itself have
            # (re)generated files under a declared input path.
            fingerprint = stage_input_fingerprint(stage, root)
            if fingerprint is not None:
                state[stage.name] = {"inputs_fingerprint": fingerprint}
                # Persist after every successful stage so a later failure
                # doesn't force completed stages to re-run next time.
                save_state(root, state)
        else:
            results.append(
                StageResult(
                    name=stage.name,
                    status="failed",
                    reason=f"exit code {returncode}",
                )
            )
            break  # do not run downstream stages after a failure
    return results


def _default_runner(command: str, cwd: Path) -> int:
    """Execute *command* in *cwd* via :func:`subprocess.run` without a shell."""
    argv = shlex.split(command)
    if not argv:
        raise PipelineError("empty command")
    proc = subprocess.run(argv, cwd=str(cwd), check=False)
    return proc.returncode


def format_summary(results: list[StageResult]) -> str:
    """Human-readable one-line-per-stage summary."""
    lines: list[str] = []
    ran = sum(1 for r in results if r.status == "ran")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    for r in results:
        marker = {"ran": "▶", "skipped": "✓", "failed": "✗"}.get(r.status, "?")
        lines.append(f"  {marker} {r.name}: {r.status} — {r.reason}")
    if ran == 0 and failed == 0:
        lines.append("all stages up to date")
    else:
        lines.append(
            f"summary: {ran} ran, {skipped} skipped" + (f", {failed} failed" if failed else "")
        )
    return "\n".join(lines)
