"""Aggregation helpers for the reliability report.

Contains pure, report-internal aggregation primitives shared by both the
Markdown and JSON renderers. All functions here are side-effect free and
depend only on stdlib + ``annotator`` constants; they deliberately have no
dependency on other ``report`` submodules to stay a leaf of the package DAG.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, cast

from agent_diagnostics.annotator import CHECKER_REQUIRES_TRAJECTORY

LOW_CONFIDENCE_N = 3
"""Matrix cells with fewer than this many trials are flagged as low-confidence."""


def _build_trials_frame(annotations: list[dict]) -> list[dict]:
    """Normalize annotation dicts into a flat frame for comparative analysis.

    Applies the `"unknown"` bucket for missing `agent_name` / `benchmark`
    exactly once so downstream helpers stay consistent. Does not mutate input.
    """
    return [
        {
            "agent": a.get("agent_name") or "unknown",
            "model": a.get("model") or None,
            "benchmark": a.get("benchmark") or "unknown",
            "config_name": a.get("config_name"),
            "passed": bool(a.get("passed")),
            "reward": a.get("reward"),
            "categories": a.get("categories", []),
            # Trajectory-volume metrics. ``None`` means "not measured"
            # (no trajectory file); ``0`` is a legitimate value
            # (crashed before first turn). Aggregators must exclude
            # ``None`` from the mean but include ``0``.
            "trajectory_length": a.get("trajectory_length"),
            "total_turns": a.get("total_turns"),
        }
        for a in annotations
    ]


def _top_failure_categories(trials: list[dict], limit: int = 3) -> list[dict]:
    """Top failure categories from FAILED trials in the given slice.

    Categories missing a ``name`` field are skipped rather than raising
    so a single malformed annotation cannot abort report generation.
    """
    counts: Counter[str] = Counter()
    for t in trials:
        if t["passed"]:
            continue
        for cat in t.get("categories", []):
            name = cat.get("name")
            if name:
                counts[name] += 1
    return [{"name": name, "count": count} for name, count in counts.most_common(limit)]


def _aggregate_slice(trials: list[dict]) -> dict:
    """Shared aggregation used by per-benchmark and per-agent summaries.

    ``mean_reward`` / ``mean_trajectory_length`` / ``mean_total_turns`` are
    ``None`` when no trial in the slice carries the corresponding value, so
    consumers can distinguish "no data" from "all zeros". For trajectory
    volume, ``None`` means "not measured" (trial lacked a trajectory file)
    and is excluded from the denominator; ``0`` is a legitimate value
    (crashed before first turn) and is included.

    The JSON output carries both ``mean_trajectory_length`` and
    ``mean_total_turns``; the markdown summary tables render only the
    former as a "Mean Trajectory Length" column. This asymmetry is
    intentional progressive disclosure — the machine-readable JSON is
    richer than the human-readable markdown so dashboards can surface
    either metric without bloating the primary report.
    """
    n = len(trials)
    passed = sum(1 for t in trials if t["passed"])
    rewards = [t["reward"] for t in trials if t["reward"] is not None]
    mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else None
    # Trajectory and turn means round to 2 decimals — these are count-like
    # metrics (tens to hundreds of turns), so sub-centi precision would be
    # noise. Markdown also renders with ``:.2f`` for the same reason; JSON
    # carries the same precision for consistency between surfaces.
    lengths = [t["trajectory_length"] for t in trials if t["trajectory_length"] is not None]
    mean_trajectory_length = round(sum(lengths) / len(lengths), 2) if lengths else None
    turns = [t["total_turns"] for t in trials if t["total_turns"] is not None]
    mean_total_turns = round(sum(turns) / len(turns), 2) if turns else None
    return {
        "n_trials": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "mean_reward": mean_reward,
        "mean_trajectory_length": mean_trajectory_length,
        "mean_total_turns": mean_total_turns,
        "top_failures": _top_failure_categories(trials),
    }


def _load_taxonomy_polarity() -> dict[str, str]:
    """Return a mapping of category name -> polarity from the taxonomy."""
    try:
        from agent_diagnostics.taxonomy import load_taxonomy

        tax = load_taxonomy()
        return {c["name"]: c["polarity"] for c in tax["categories"]}
    except Exception:
        return {}


def _corpus_stats(annotations: list[dict]) -> dict[str, Any]:
    """Compute top-level corpus statistics."""
    total = len(annotations)
    passed = sum(1 for a in annotations if a.get("passed"))
    failed = total - passed
    rewards = [a["reward"] for a in annotations if a.get("reward") is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    configs = set()
    benchmarks = set()
    for a in annotations:
        if a.get("config_name"):
            configs.add(a["config_name"])
        if a.get("benchmark"):
            benchmarks.add(a["benchmark"])

    return {
        "total_trials": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "avg_reward": round(avg_reward, 4),
        "configs": sorted(configs),
        "benchmarks": sorted(benchmarks),
    }


def _count_trajectory_available(annotations: list[dict]) -> int:
    """Count annotations that have trajectory data available."""
    count = 0
    for a in annotations:
        # Check signals sub-dict first, then top-level
        signals = a.get("signals") or {}
        if signals.get("has_trajectory", False) or a.get("has_trajectory", False):
            count += 1
    return count


def _category_counts(annotations: list[dict]) -> dict[str, int]:
    """Count how many trials are assigned each category."""
    counts: Counter[str] = Counter()
    for a in annotations:
        for cat in a.get("categories", []):
            name = cat.get("name")
            if name:
                counts[name] += 1
    return dict(counts.most_common())


def _category_counts_with_denominators(
    annotations: list[dict],
) -> dict[str, dict[str, Any]]:
    """Count categories with appropriate denominators.

    Returns a dict mapping category name to
    ``{"count": N, "denominator": M, "rate": N/M}``.

    Trajectory-dependent categories use trajectory-available trials as
    denominator; reward-dependent categories use full corpus size.
    """
    total = len(annotations)
    traj_available = _count_trajectory_available(annotations)

    counts: Counter[str] = Counter()
    for a in annotations:
        for cat in a.get("categories", []):
            name = cat.get("name")
            if name:
                counts[name] += 1

    result: dict[str, dict[str, Any]] = {}
    for name, count in counts.most_common():
        requires_traj = CHECKER_REQUIRES_TRAJECTORY.get(name, True)
        denominator = traj_available if requires_traj else total
        rate = round(count / denominator, 4) if denominator > 0 else 0.0
        result[name] = {"count": count, "denominator": denominator, "rate": rate}
    return result


def _category_by_config(annotations: list[dict]) -> dict[str, dict[str, int]]:
    """Nested dict: config_name -> category_name -> count."""
    result: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for a in annotations:
        cfg = a.get("config_name") or "unknown"
        for cat in a.get("categories", []):
            name = cat.get("name")
            if name:
                result[cfg][name] += 1
    # Convert counters to plain dicts, sorted
    return {k: dict(sorted(v.items(), key=lambda x: -x[1])) for k, v in sorted(result.items())}


def _category_by_suite(annotations: list[dict]) -> dict[str, dict[str, int]]:
    """Nested dict: benchmark_suite -> category_name -> count."""
    result: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for a in annotations:
        suite = a.get("benchmark") or "unknown"
        for cat in a.get("categories", []):
            name = cat.get("name")
            if name:
                result[suite][name] += 1
    # Convert counters to plain dicts, sorted by total count desc
    return {
        k: dict(sorted(v.items(), key=lambda x: -x[1]))
        for k, v in sorted(result.items(), key=lambda x: -sum(x[1].values()))
    }


def _core_task_name(trial_path: str) -> str:
    """Extract canonical task name from trial_path for cross-config pairing.

    Trial dirs follow the pattern ``{prefix}_{task}__{hash}`` or ``{task}__{hash}``.
    Strips config-specific prefixes (bl_, mcp_, sgonly_, baseline_) and the
    trailing ``__hash`` to produce a case-insensitive key.
    """
    trial_dir = trial_path.rstrip("/").rsplit("/", 1)[-1]
    # Strip __hash suffix
    if "__" in trial_dir:
        trial_dir = trial_dir.rsplit("__", 1)[0]
    # Strip known config-specific prefixes
    for prefix in ("baseline_", "bl_", "mcp_", "sgonly_"):
        if trial_dir.startswith(prefix):
            trial_dir = trial_dir[len(prefix) :]
            break
    return trial_dir.lower()


def _paired_comparison(annotations: list[dict]) -> list[dict]:
    """Compare failure-mode deltas between config pairs on shared tasks.

    For each pair of configs that share >= 20 tasks, compute per-category
    delta = count_in_A - count_in_B.  Returns a list of pair dicts with
    the top 5 categories in each direction (fixed-by-B and introduced-by-B).
    """
    # Group annotations by (core_task, config) -> set of category names
    task_config_cats: dict[tuple[str, str], set[str]] = defaultdict(set)
    for a in annotations:
        cfg = a.get("config_name")
        path = a.get("trial_path", "")
        if not cfg or not path:
            continue
        core = _core_task_name(path)
        for cat in a.get("categories", []):
            name = cat.get("name")
            if name:
                task_config_cats[(core, cfg)].add(name)

    # Build per-task -> configs that ran it
    task_configs: dict[str, set[str]] = defaultdict(set)
    for core, cfg in task_config_cats:
        task_configs[core].add(cfg)

    # Only consider tasks with 2+ configs
    multi_config_tasks = {t: cfgs for t, cfgs in task_configs.items() if len(cfgs) >= 2}

    # For each config pair, compute deltas
    all_configs = sorted({cfg for cfgs in multi_config_tasks.values() for cfg in cfgs})
    pair_results = []

    for i, cfg_a in enumerate(all_configs):
        for cfg_b in all_configs[i + 1 :]:
            # Find shared tasks
            shared = [
                t
                for t in multi_config_tasks
                if cfg_a in multi_config_tasks[t] and cfg_b in multi_config_tasks[t]
            ]
            if len(shared) < 20:
                continue

            # Delta: positive = more in A, negative = more in B
            delta: Counter = Counter()
            for task in shared:
                cats_a = task_config_cats.get((task, cfg_a), set())
                cats_b = task_config_cats.get((task, cfg_b), set())
                for cat in cats_a - cats_b:
                    delta[cat] += 1
                for cat in cats_b - cats_a:
                    delta[cat] -= 1

            # Top 5 where A has more (introduced by A / fixed by B)
            introduced_by_a = sorted(
                [(cat, cnt) for cat, cnt in delta.items() if cnt > 0],
                key=lambda x: -x[1],
            )[:5]
            # Top 5 where B has more (introduced by B / fixed by A)
            introduced_by_b = sorted(
                [(cat, cnt) for cat, cnt in delta.items() if cnt < 0],
                key=lambda x: x[1],
            )[:5]

            pair_results.append(
                {
                    "config_a": cfg_a,
                    "config_b": cfg_b,
                    "shared_tasks": len(shared),
                    "introduced_by_a": [{"category": c, "delta": d} for c, d in introduced_by_a],
                    "introduced_by_b": [
                        {"category": c, "delta": abs(d)} for c, d in introduced_by_b
                    ],
                }
            )

    # Sort by number of shared tasks descending
    pair_results.sort(key=lambda x: -cast(int, x["shared_tasks"]))
    return pair_results


def _top_categories_with_examples(
    annotations: list[dict],
    polarity_map: dict[str, str],
    polarity: str,
    top_n: int = 3,
    examples_per: int = 2,
) -> list[dict]:
    """Get top N categories of given polarity with example trials."""
    # Count by polarity
    counts: Counter[str] = Counter()
    cat_trials: dict[str, list[dict]] = defaultdict(list)

    for a in annotations:
        for cat in a.get("categories", []):
            name = cat.get("name")
            if name and polarity_map.get(name) == polarity:
                counts[name] += 1
                cat_trials[name].append(
                    {
                        "task_id": a.get("task_id", "unknown"),
                        "config_name": a.get("config_name", "unknown"),
                        "reward": a.get("reward", 0),
                        "evidence": cat.get("evidence", ""),
                    }
                )

    results = []
    for name, count in counts.most_common(top_n):
        examples = cat_trials[name][:examples_per]
        results.append({"name": name, "count": count, "examples": examples})
    return results
