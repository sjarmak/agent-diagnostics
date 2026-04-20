"""Pipeline runner for the Agent Reliability Observatory (PRD NH-2).

Loads a ``pipeline.toml`` at the project root declaring stages (``extract``,
``annotate``, ``report``, ...), each with ``inputs``, ``outputs``, and a
``command`` template.  ``observatory pipeline run`` walks the stages in the
declared order, compares input vs output mtimes, and runs only stages whose
outputs are stale.

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

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


def _max_mtime(paths: list[Path]) -> float | None:
    """Return the latest mtime among *paths* (recursively for dirs), or None.

    ``None`` means no path exists at all.
    """
    mtimes: list[float] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    mtimes.append(child.stat().st_mtime)
        else:
            mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else None


def _min_mtime(paths: list[Path]) -> float | None:
    mtimes: list[float] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    mtimes.append(child.stat().st_mtime)
        else:
            mtimes.append(p.stat().st_mtime)
    return min(mtimes) if mtimes else None


def is_stale(stage: Stage, project_root: Path) -> tuple[bool, str]:
    """Decide whether *stage* needs to run.

    Returns ``(stale, reason)``.  A stage is stale when:

    - Any declared output is missing, or
    - The oldest output is older than the newest input.
    """
    root = Path(project_root)
    input_paths = [root / p for p in stage.inputs]
    output_paths = [root / p for p in stage.outputs]

    missing_outputs = [str(p) for p in output_paths if not p.exists()]
    if missing_outputs:
        return True, f"missing output(s): {', '.join(missing_outputs)}"

    newest_input = _max_mtime(input_paths)
    if newest_input is None:
        # No inputs exist — either declarative error or pre-flight stage.
        # Treat as not-stale so we don't endlessly re-run.
        return False, "no inputs present on disk"

    oldest_output = _min_mtime(output_paths)
    if oldest_output is None:
        return True, "no outputs produced yet"

    if newest_input > oldest_output:
        return True, "input newer than output"
    return False, "up-to-date"


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

    results: list[StageResult] = []
    for stage in stages:
        stale, reason = is_stale(stage, root)
        if not stale:
            results.append(StageResult(name=stage.name, status="skipped", reason=reason))
            continue
        returncode = runner(stage.command, root)
        if returncode == 0:
            results.append(StageResult(name=stage.name, status="ran", reason=reason))
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
