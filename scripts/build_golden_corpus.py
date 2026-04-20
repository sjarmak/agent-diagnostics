"""Build the golden regression corpus from data/export/signals.parquet.

This script is the reproducible source of truth for
``tests/fixtures/golden_corpus/``.  Running it regenerates the corpus from
the signals parquet.  It is NOT executed in CI; the committed fixtures are
the ground truth.  Re-run manually when source data changes.

Usage
-----
    python scripts/build_golden_corpus.py \
        --parquet data/export/signals.parquet \
        --output tests/fixtures/golden_corpus \
        --target 40

The script performs three passes:

1. **Selection**: stratifies the signals parquet across
   (agent, benchmark, passed) and picks a set of trials that
   covers >=3 agents, >=5 benchmarks, and as many of the 40 v3
   taxonomy categories as possible.

2. **Resolution**: for each candidate, resolves the trajectory
   from ``trial_path``.  Candidates with missing/truncated
   trajectories are dropped.

3. **Write**: writes ``signals.json``, ``trajectory.json``,
   ``expected_annotations.json`` (heuristic seed — the LLM
   curation pass fills in ``curator_notes``), and ``metadata.json``
   into each trial directory.

The LLM curation and reviewer passes are performed separately.
See ``tests/fixtures/golden_corpus/README.md`` for the full
methodology.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Make the src package importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_diagnostics.annotator import annotate_trial  # noqa: E402
from agent_diagnostics.taxonomy import valid_category_names  # noqa: E402

TAXONOMY_PATH = SRC_ROOT / "agent_diagnostics" / "taxonomy_v3.yaml"

# Trajectory trimming — keep head + tail + "interesting" middle steps when
# a trajectory is larger than this byte threshold.
TRAJECTORY_BYTE_THRESHOLD = 64 * 1024
TRAJECTORY_HEAD_STEPS = 15
TRAJECTORY_TAIL_STEPS = 15
TRAJECTORY_INTERESTING_MAX = 10
# Maximum byte size for any single serialized step; longer steps are
# replaced with a placeholder to prevent one huge observation blowing up
# the fixture size.
MAX_STEP_BYTES = 8 * 1024

# Category priority — MH-10 "new" categories listed first so the selector
# biases toward them when choosing among otherwise-equivalent candidates.
PRIORITY_CATEGORIES: tuple[str, ...] = (
    "reward_hacking",
    "fabricated_success",
    "hallucinated_api",
    "tool_argument_error",
    "premature_termination",
    "verification_skipped",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A selected trial with resolution metadata."""

    trial_id: str
    row: dict[str, Any]
    trial_dir: Path
    trajectory_path: Path
    heuristic_categories: tuple[str, ...]


@dataclass
class SelectionResult:
    """Result of the selection pass."""

    candidates: list[Candidate] = field(default_factory=list)
    agents: set[str] = field(default_factory=set)
    benchmarks: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    passed_count: int = 0
    failed_count: int = 0
    skipped_missing_trajectory: int = 0

    def add(self, cand: Candidate) -> None:
        self.candidates.append(cand)
        self.agents.add(str(cand.row["agent_name"]))
        self.benchmarks.add(str(cand.row["benchmark"]))
        self.categories.update(cand.heuristic_categories)
        if bool(cand.row["passed"]):
            self.passed_count += 1
        else:
            self.failed_count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_trajectory_path(trial_path_str: str) -> Path | None:
    """Return the first existing trajectory artifact path for *trial_path_str*.

    Falls back to agent-specific alternate filenames (e.g.
    ``agent/cursor-agent.jsonl`` for cursor-cli trials) so the corpus can
    include agents whose raw trace schema differs from ATIF-v1.2.
    """
    base = Path(trial_path_str)
    if not base.exists():
        return None
    for rel in (
        "agent/trajectory.json",
        "trajectory.json",
        "agent/cursor-agent.jsonl",
    ):
        candidate = base / rel
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _row_to_signals(row: pd.Series) -> dict[str, Any]:
    """Convert a parquet row into a plain-Python signals dict."""
    raw = row.to_dict()
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, bool):
            out[k] = v
        elif pd.isna(v):
            out[k] = None
        else:
            out[k] = v
    # Normalize sequence fields that may be numpy arrays -> python lists
    for seq_key in ("tool_call_sequence", "files_read_list", "files_edited_list"):
        val = out.get(seq_key)
        if val is None:
            out[seq_key] = []
        elif not isinstance(val, list):
            out[seq_key] = list(val)
    return out


def _safe_heuristic_categories(signals: dict[str, Any]) -> tuple[str, ...]:
    """Run the heuristic annotator, returning just the category names."""
    try:
        assignments = annotate_trial(signals)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — best-effort in a build script
        return ()
    return tuple(a.name for a in assignments)


_API_HALLUCINATION_MARKERS = (
    "modulenotfounderror",
    "no module named",
    "has no attribute",
    "importerror",
    "attributeerror",
    "name is not defined",
)
_FABRICATION_MARKERS = (
    "task is complete",
    "i have completed",
    "successfully implemented",
    "all tests pass",
    "the fix is complete",
)
_DESTRUCTIVE_MARKERS = (
    "rm -rf",
    "git push --force",
    "drop table",
    "force-push",
)
_CREDENTIAL_MARKERS = (
    "api_key=",
    "password=",
    "secret_key=",
    "aws_access_key",
    "private_key",
)


def _scan_trajectory_markers(traj: dict[str, Any]) -> dict[str, bool]:
    """Return boolean flags for interesting trajectory markers.

    Scans the FULL trajectory BEFORE trimming so diagnostic markers in
    the middle of long runs are not lost.  Stored in metadata for use
    by the curator.
    """
    try:
        text = json.dumps(traj, default=str).lower()
    except TypeError:
        return {}
    return {
        "has_api_hallucination_marker": any(m in text for m in _API_HALLUCINATION_MARKERS),
        "has_fabrication_marker": any(m in text for m in _FABRICATION_MARKERS),
        "has_destructive_marker": any(m in text for m in _DESTRUCTIVE_MARKERS),
        "has_credential_marker": any(m in text for m in _CREDENTIAL_MARKERS),
    }


def _load_trajectory(path: Path) -> dict[str, Any]:
    """Load a trajectory, normalizing JSONL -> dict(steps=[...])."""
    if path.suffix.lower() == ".jsonl":
        steps: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines but preserve the line number so
                    # the curator can still see where the break occurred.
                    steps.append({"step_id": line_no, "source": "malformed"})
        return {
            "schema_version": "cursor-jsonl-v1",
            "source_format": "jsonl",
            "steps": steps,
        }
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _shrink_step(step: dict[str, Any]) -> dict[str, Any]:
    """Return a version of *step* with large payloads truncated.

    Preserves top-level structural keys but truncates long strings and
    large nested objects so a single monster observation can't blow up
    the fixture.  Fully serializable.
    """
    try:
        encoded = json.dumps(step, default=str)
    except TypeError:
        return {
            "step_id": step.get("step_id"),
            "source": step.get("source"),
            "_corpus_truncated": "unserializable",
        }
    if len(encoded) <= MAX_STEP_BYTES:
        return step

    shrunk: dict[str, Any] = {}
    for k, v in step.items():
        if isinstance(v, str) and len(v) > 2048:
            shrunk[k] = v[:1024] + "\n...[truncated]..." + v[-512:]
        elif isinstance(v, (dict, list)):
            nested = json.dumps(v, default=str)
            if len(nested) > 2048:
                shrunk[k] = {
                    "_corpus_truncated": True,
                    "preview": nested[:1024] + "...[truncated]...",
                    "original_bytes": len(nested),
                }
            else:
                shrunk[k] = v
        else:
            shrunk[k] = v
    shrunk["_corpus_step_shrunk"] = True
    return shrunk


def _trim_trajectory(traj: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (trimmed_traj, trim_applied).

    Two-pass trim:
    1. Shrink any individual step > ``MAX_STEP_BYTES``.
    2. If still > ``TRAJECTORY_BYTE_THRESHOLD``, keep head + tail of
       ``steps`` plus up to N "interesting" middle steps (those that
       mention errors/exceptions).
    """
    trim_applied = False

    steps = traj.get("steps")
    if isinstance(steps, list):
        shrunk_steps = [_shrink_step(s) if isinstance(s, dict) else s for s in steps]
        if any(isinstance(s, dict) and s.get("_corpus_step_shrunk") for s in shrunk_steps):
            trim_applied = True
        traj = dict(traj)
        traj["steps"] = shrunk_steps
        steps = shrunk_steps

    encoded = json.dumps(traj, default=str)
    if len(encoded) <= TRAJECTORY_BYTE_THRESHOLD:
        return traj, trim_applied

    if not isinstance(steps, list) or len(steps) <= (
        TRAJECTORY_HEAD_STEPS + TRAJECTORY_TAIL_STEPS
    ):
        # Can't sensibly trim step list any further.
        return traj, trim_applied

    head = steps[:TRAJECTORY_HEAD_STEPS]
    tail = steps[-TRAJECTORY_TAIL_STEPS:]
    middle = steps[TRAJECTORY_HEAD_STEPS:-TRAJECTORY_TAIL_STEPS]

    interesting: list[dict[str, Any]] = []
    for step in middle:
        if not isinstance(step, dict):
            continue
        text = json.dumps(step, default=str).lower()
        if "error" in text or "traceback" in text or "exception" in text:
            interesting.append(step)
            if len(interesting) >= TRAJECTORY_INTERESTING_MAX:
                break

    omitted = len(steps) - len(head) - len(tail) - len(interesting)
    placeholder = {
        "step_id": None,
        "source": "corpus_builder",
        "message": (
            f"[trimmed] {omitted} intervening steps omitted for fixture "
            "size; interesting error/exception steps retained separately."
        ),
        "extra": {"trimmed": True},
    }

    trimmed = dict(traj)
    trimmed["steps"] = head + [placeholder] + interesting + tail
    trimmed["_corpus_trim_applied"] = True
    trimmed["_corpus_original_step_count"] = len(steps)
    return trimmed, True


def _short_trial_id(trial_id: str) -> str:
    """Return a file-safe 12-hex-char short id."""
    return trial_id[:12]


def _rank_candidate_row(row: pd.Series) -> tuple:
    """Ranking key that biases toward informative trials.

    Preferred (lower sort key = earlier):
    1. Failed trials (more diagnostic categories to curate).
    2. Non-trivial tool call count (>5) for better trajectories.
    3. has_trajectory True.
    4. deterministic hash of trial_id as final tiebreaker.
    """
    passed = bool(row["passed"])
    tool_calls = int(row.get("tool_calls_total", 0) or 0)
    has_traj = bool(row["has_trajectory"])
    return (
        0 if not passed else 1,  # prefer failures
        0 if has_traj else 1,
        0 if 5 <= tool_calls <= 200 else 1,
        str(row["trial_id"]),
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_trials(
    df: pd.DataFrame,
    *,
    target: int = 40,
    max_per_stratum: int = 3,
) -> SelectionResult:
    """Greedy stratified selection across (agent, benchmark, passed).

    Strategy
    --------
    1. Enumerate strata (agent, benchmark, passed).
    2. Within each stratum, rank rows by ``_rank_candidate_row``.
    3. Three rotating sub-quotas to guarantee balance:
         - agent rotation
         - passed/failed rotation
       combined into the stratum visit order each round.
    4. Resolve trajectory; skip missing.
    """
    result = SelectionResult()

    # Build stratum -> sorted list of rows
    strata: dict[tuple[str, str, bool], list[pd.Series]] = {}
    for _, row in df.iterrows():
        key = (str(row["agent_name"]), str(row["benchmark"]), bool(row["passed"]))
        strata.setdefault(key, []).append(row)

    for rows in strata.values():
        rows.sort(key=_rank_candidate_row)

    cursors: dict[tuple[str, str, bool], int] = {k: 0 for k in strata}
    per_stratum_taken: dict[tuple[str, str, bool], int] = {k: 0 for k in strata}
    seen_trial_ids: set[str] = set()

    agents_all = sorted({k[0] for k in strata})
    # Round-robin ordering: every (agent, passed_flag) pair gets exactly
    # one slot per round.  This guarantees balance across both axes.
    round_order: list[tuple[str, bool]] = [
        (agent, passed_flag) for agent in agents_all for passed_flag in (False, True)
    ]

    def _pick_from_subset(subset_keys: list[tuple[str, str, bool]]) -> bool:
        """Try to pick one trial from the given subset of strata keys.

        Returns True if a trial was added.
        """
        # Sort subset by fewest picks so far to spread across benchmarks
        subset_keys = sorted(subset_keys, key=lambda k: per_stratum_taken[k])
        for key in subset_keys:
            if per_stratum_taken[key] >= max_per_stratum:
                continue
            idx = cursors[key]
            rows = strata[key]
            while idx < len(rows):
                row = rows[idx]
                idx += 1
                trial_id = str(row["trial_id"])
                if trial_id in seen_trial_ids:
                    continue
                traj_path = _resolve_trajectory_path(str(row["trial_path"]))
                if traj_path is None:
                    result.skipped_missing_trajectory += 1
                    continue
                signals = _row_to_signals(row)
                cats = _safe_heuristic_categories(signals)
                cand = Candidate(
                    trial_id=trial_id,
                    row=signals,
                    trial_dir=Path(str(row["trial_path"])),
                    trajectory_path=traj_path,
                    heuristic_categories=cats,
                )
                result.add(cand)
                seen_trial_ids.add(trial_id)
                per_stratum_taken[key] += 1
                cursors[key] = idx
                return True
            cursors[key] = idx
        return False

    while len(result.candidates) < target:
        progress = False
        for agent, passed_flag in round_order:
            if len(result.candidates) >= target:
                break
            subset = [k for k in strata if k[0] == agent and k[2] == passed_flag]
            if not subset:
                continue
            if _pick_from_subset(subset):
                progress = True
        if not progress:
            break

    return result


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_candidate(
    cand: Candidate,
    output_dir: Path,
    *,
    valid_category_names_set: frozenset[str],
) -> Path:
    """Write the four JSON files for *cand* under *output_dir*."""
    short_id = _short_trial_id(cand.trial_id)
    trial_out = output_dir / short_id
    trial_out.mkdir(parents=True, exist_ok=True)

    # signals.json — full row, with host-absolute trial_path redacted
    row_redacted = dict(cand.row)
    if "trial_path" in row_redacted:
        row_redacted["trial_path"] = None
    (trial_out / "signals.json").write_text(
        json.dumps(row_redacted, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # trajectory.json — trimmed if large.  Supports ATIF JSON and
    # cursor-style JSONL.  JSONL entries are read into a ``steps`` list.
    traj = _load_trajectory(cand.trajectory_path)
    traj_markers = _scan_trajectory_markers(traj)
    trimmed_traj, trim_applied = _trim_trajectory(traj)
    (trial_out / "trajectory.json").write_text(
        json.dumps(trimmed_traj, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # metadata.json — small manifest
    metadata = {
        "trial_id_short": short_id,
        "trial_id_full": cand.row.get("trial_id_full") or cand.trial_id,
        "model": cand.row.get("model"),
        "agent": cand.row.get("agent_name"),
        "benchmark": cand.row.get("benchmark"),
        "reward": cand.row.get("reward"),
        "passed": cand.row.get("passed"),
        "has_trajectory": cand.row.get("has_trajectory"),
        "total_turns": cand.row.get("total_turns"),
        "tool_calls_total": cand.row.get("tool_calls_total"),
        "duration_seconds": cand.row.get("duration_seconds"),
        "trajectory_trim_applied": trim_applied,
        "selection_reason": (
            f"stratified sample (agent={cand.row.get('agent_name')}, "
            f"benchmark={cand.row.get('benchmark')}, "
            f"passed={cand.row.get('passed')})"
        ),
        "heuristic_categories": list(cand.heuristic_categories),
        "trajectory_markers": traj_markers,
    }
    (trial_out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # expected_annotations.json — seeded with heuristic output; curation
    # pass overwrites curator_notes, categories, and adds reviewer block.
    expected: dict[str, Any] = {
        "trial_id_short": short_id,
        "categories": [
            {
                "name": name,
                "confidence": 0.5,
                "evidence": "heuristic seed; curator to refine.",
            }
            for name in cand.heuristic_categories
            if name in valid_category_names_set
        ],
        "curator_notes": {
            "methodology": "LLM-assisted curation seeded by heuristic annotator.",
            "curator_model": "pending",
            "curator_confidence": "pending",
            "ambiguities_considered": [],
            "rejected_categories": [],
        },
        "reviewer": {
            "reviewer_model": "pending",
            "agreed_categories": [],
            "disputed_categories": [],
        },
    }
    (trial_out / "expected_annotations.json").write_text(
        json.dumps(expected, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    return trial_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=REPO_ROOT / "data" / "export" / "signals.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "golden_corpus",
    )
    parser.add_argument("--target", type=int, default=40)
    parser.add_argument("--max-per-stratum", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.parquet.exists():
        print(f"parquet file not found: {args.parquet}", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df)} rows from {args.parquet}", file=sys.stderr)

    valid_names = frozenset(valid_category_names(TAXONOMY_PATH))

    selection = select_trials(
        df,
        target=args.target,
        max_per_stratum=args.max_per_stratum,
    )
    print(
        f"selected {len(selection.candidates)} trials "
        f"(skipped {selection.skipped_missing_trajectory} missing trajectories)",
        file=sys.stderr,
    )
    print(f"  agents: {sorted(selection.agents)}", file=sys.stderr)
    print(
        f"  benchmarks: {len(selection.benchmarks)} ({sorted(selection.benchmarks)[:5]}...)",
        file=sys.stderr,
    )
    print(f"  heuristic categories: {sorted(selection.categories)}", file=sys.stderr)
    print(
        f"  pass/fail: {selection.passed_count}/{selection.failed_count}",
        file=sys.stderr,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for cand in selection.candidates:
        trial_dir = write_candidate(cand, args.output, valid_category_names_set=valid_names)
        print(f"  wrote {trial_dir}", file=sys.stderr)

    # Write a manifest summarizing the corpus composition.
    manifest = {
        "corpus_version": 1,
        "trial_count": len(selection.candidates),
        "agents": sorted(selection.agents),
        "benchmarks": sorted(selection.benchmarks),
        "heuristic_categories_covered": sorted(selection.categories),
        "passed_count": selection.passed_count,
        "failed_count": selection.failed_count,
        "skipped_missing_trajectory": selection.skipped_missing_trajectory,
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output / 'MANIFEST.json'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
