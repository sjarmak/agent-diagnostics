"""Report generator for the Agent Reliability Observatory.

Aggregates annotations into a Markdown reliability report with a JSON
companion file containing machine-readable statistics.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_diagnostics.annotator import CHECKER_REQUIRES_TRAJECTORY

LOW_CONFIDENCE_N = 3
"""Matrix cells with fewer than this many trials are flagged as low-confidence."""


def _md_cell(value: object) -> str:
    """Escape a value for safe embedding in a GFM table cell.

    Replaces pipe and newline characters that would otherwise break
    column alignment or smuggle new rows/sections into the output.
    """
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")


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

    ``mean_reward`` is ``None`` when no trial in the slice carries a reward,
    so consumers can distinguish "no data" from "all zeros".
    """
    n = len(trials)
    passed = sum(1 for t in trials if t["passed"])
    rewards = [t["reward"] for t in trials if t["reward"] is not None]
    mean_reward = round(sum(rewards) / len(rewards), 4) if rewards else None
    return {
        "n_trials": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "mean_reward": mean_reward,
        "top_failures": _top_failure_categories(trials),
    }


def _per_benchmark_summary(frame: list[dict]) -> list[dict]:
    """Per-benchmark aggregates, sorted desc by n_trials."""
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for row in frame:
        by_bench[row["benchmark"]].append(row)

    out = [
        {"benchmark": bench, **_aggregate_slice(trials)}
        for bench, trials in by_bench.items()
    ]
    out.sort(key=lambda r: (-r["n_trials"], r["benchmark"]))
    return out


def _per_agent_summary(frame: list[dict]) -> list[dict]:
    """Per-(agent, model) aggregates, sorted desc by n_trials.

    Labels use `"agent / model"` only when the agent has more than one model
    in the dataset; otherwise the bare agent name. Scoping is per-agent so
    the label rule is local, not global.
    """
    by_key: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    for row in frame:
        by_key[(row["agent"], row["model"])].append(row)

    models_per_agent: dict[str, set[str | None]] = defaultdict(set)
    for agent, model in by_key:
        models_per_agent[agent].add(model)

    out: list[dict] = []
    for (agent, model), trials in by_key.items():
        multi_model = len(models_per_agent[agent]) > 1
        model_display = model if model is not None else "(no model)"
        label = f"{agent} / {model_display}" if multi_model else agent
        out.append(
            {
                "agent": agent,
                "model": model,
                "label": label,
                **_aggregate_slice(trials),
            }
        )
    out.sort(key=lambda r: (-r["n_trials"], r["label"]))
    return out


def _agent_benchmark_matrix(frame: list[dict]) -> dict[str, dict[str, dict]]:
    """Agent x benchmark cross-tab marginalized across configs.

    Returns `{agent: {benchmark: {pass_rate, n_trials, low_confidence}}}`.
    Cells below `LOW_CONFIDENCE_N` trials are flagged but NOT hidden.
    """
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in frame:
        cells[(row["agent"], row["benchmark"])].append(row["passed"])

    matrix: dict[str, dict[str, dict]] = defaultdict(dict)
    for (agent, bench), passes in cells.items():
        n = len(passes)
        pass_rate = sum(passes) / n if n else 0.0
        matrix[agent][bench] = {
            "pass_rate": round(pass_rate, 4),
            "n_trials": n,
            "low_confidence": n < LOW_CONFIDENCE_N,
        }
    return {k: dict(v) for k, v in matrix.items()}


def _top_divergences(matrix: dict[str, dict[str, dict]], top_k: int = 10) -> list[dict]:
    """Per-benchmark largest pairwise pass-rate delta across agents.

    Benchmarks with fewer than 2 agents are skipped. Ties broken by
    benchmark name ascending. Each row carries the full per-agent list
    so consumers can re-sort beyond the primary metric.
    """
    benchmarks: set[str] = set()
    for agent_cells in matrix.values():
        benchmarks.update(agent_cells.keys())

    rows: list[dict] = []
    for bench in benchmarks:
        agents_here = [
            (agent, cells[bench]) for agent, cells in matrix.items() if bench in cells
        ]
        if len(agents_here) < 2:
            continue
        # Sort desc by pass_rate, tiebreaker by agent name ascending so
        # pair_a / pair_b are stable across runs with different input orderings.
        agents_here.sort(key=lambda ac: (-ac[1]["pass_rate"], ac[0]))
        high_agent, high_cell = agents_here[0]
        low_agent, low_cell = agents_here[-1]
        rows.append(
            {
                "benchmark": bench,
                "max_delta": round(high_cell["pass_rate"] - low_cell["pass_rate"], 4),
                "pair_a": high_agent,
                "pair_b": low_agent,
                "pair_a_rate": high_cell["pass_rate"],
                "pair_b_rate": low_cell["pass_rate"],
                "all_agents": [
                    {
                        "agent": agent,
                        "pass_rate": cell["pass_rate"],
                        "n_trials": cell["n_trials"],
                        "low_confidence": cell["low_confidence"],
                    }
                    for agent, cell in agents_here
                ],
            }
        )
    rows.sort(key=lambda r: (-r["max_delta"], r["benchmark"]))
    return rows[:top_k]


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
    return {
        k: dict(sorted(v.items(), key=lambda x: -x[1]))
        for k, v in sorted(result.items())
    }


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
                    "introduced_by_a": [
                        {"category": c, "delta": d} for c, d in introduced_by_a
                    ],
                    "introduced_by_b": [
                        {"category": c, "delta": abs(d)} for c, d in introduced_by_b
                    ],
                }
            )

    # Sort by number of shared tasks descending
    pair_results.sort(key=lambda x: -x["shared_tasks"])
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


def co_occurrence_matrix(annotations: list[dict]) -> dict[str, dict[str, float]]:
    """Compute category co-occurrence matrix with phi coefficients.

    Returns a symmetric dict-of-dicts where:
    - Diagonal entries contain the prevalence count for that category.
    - Off-diagonal entries contain the phi coefficient in [-1, 1].

    Parameters
    ----------
    annotations : list[dict]
        List of annotation dicts, each with a ``categories`` key.

    Returns
    -------
    dict[str, dict[str, float]]
        Symmetric matrix as nested dicts.
    """
    n = len(annotations)
    if n == 0:
        return {}

    # Build per-annotation category sets
    trial_cats: list[set[str]] = []
    all_cats: set[str] = set()
    for a in annotations:
        cats = {name for cat in a.get("categories", []) if (name := cat.get("name"))}
        trial_cats.append(cats)
        all_cats.update(cats)

    if not all_cats:
        return {}

    sorted_cats = sorted(all_cats)

    # Count occurrences per category
    cat_count: dict[str, int] = Counter()
    for cats in trial_cats:
        for c in cats:
            cat_count[c] += 1

    # Count pairwise co-occurrences
    pair_count: dict[tuple[str, str], int] = Counter()
    for cats in trial_cats:
        cat_list = sorted(cats)
        for i, a in enumerate(cat_list):
            for b in cat_list[i + 1 :]:
                pair_count[(a, b)] += 1

    # Build matrix
    matrix: dict[str, dict[str, float]] = {}
    for cat in sorted_cats:
        matrix[cat] = {}

    for cat in sorted_cats:
        # Diagonal: prevalence count
        matrix[cat][cat] = float(cat_count[cat])

    for i, cat_a in enumerate(sorted_cats):
        for cat_b in sorted_cats[i + 1 :]:
            n1 = cat_count[cat_a]
            n2 = cat_count[cat_b]
            n11 = pair_count.get((cat_a, cat_b), 0)

            # Phi coefficient
            # phi = (n*n11 - n1*n2) / sqrt(n1 * n2 * (n - n1) * (n - n2))
            numerator = n * n11 - n1 * n2
            denominator_val = n1 * n2 * (n - n1) * (n - n2)
            if denominator_val <= 0:
                phi = 0.0
            else:
                phi = numerator / math.sqrt(denominator_val)

            matrix[cat_a][cat_b] = round(phi, 6)
            matrix[cat_b][cat_a] = round(phi, 6)

    return matrix


def dimension_aggregation(
    annotations: list[dict], taxonomy: dict
) -> dict[str, dict[str, Any]]:
    """Roll up categories to parent dimensions and compute per-dimension failure rates.

    Parameters
    ----------
    annotations : list[dict]
        List of annotation dicts.
    taxonomy : dict
        Taxonomy dict with ``dimensions`` list (v2/v3 format).

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{dimension_name: {"failure_rate": float, "trial_count": int}}``.
    """
    # Build category -> dimension mapping
    cat_to_dim: dict[str, str] = {}
    for dim in taxonomy.get("dimensions", []):
        dim_name = dim["name"]
        for cat in dim.get("categories", []):
            cat_name = cat["name"] if isinstance(cat, dict) else cat
            cat_to_dim[cat_name] = dim_name

    # Per-dimension: collect trial indices that have at least one category in that dimension
    dim_trials: dict[str, list[dict]] = defaultdict(list)
    for a in annotations:
        dims_seen: set[str] = set()
        for cat in a.get("categories", []):
            name = cat.get("name")
            if not name:
                continue
            dim = cat_to_dim.get(name)
            if dim and dim not in dims_seen:
                dims_seen.add(dim)
                dim_trials[dim].append(a)

    result: dict[str, dict[str, Any]] = {}
    for dim_name, trials in sorted(dim_trials.items()):
        total = len(trials)
        failed = sum(1 for t in trials if not t.get("passed"))
        failure_rate = round(failed / total, 4) if total > 0 else 0.0
        result[dim_name] = {"failure_rate": failure_rate, "trial_count": total}

    return result


def _render_summary_table(
    lines: list[str],
    title: str,
    key_header: str,
    rows: list[dict],
    key_field: str,
    empty_msg: str,
) -> None:
    """Render a per-<key> summary table (benchmark or agent)."""
    lines.append(f"## {title}")
    lines.append("")
    if rows:
        lines.append(
            f"| {key_header} | Trials | Passed | Pass Rate | Mean Reward | Top Failures |"
        )
        lines.append(
            "|"
            + "-" * (len(key_header) + 2)
            + "|--------|--------|-----------|-------------|--------------|"
        )
        for row in rows:
            top = (
                ", ".join(
                    f"{_md_cell(f['name'])} ({f['count']})" for f in row["top_failures"]
                )
                or "—"
            )
            mean_reward = row.get("mean_reward")
            mean_reward_str = f"{mean_reward:.4f}" if mean_reward is not None else "—"
            lines.append(
                f"| {_md_cell(row[key_field])} | {row['n_trials']:,} | "
                f"{row['passed']:,} | {row['pass_rate']:.1%} | {mean_reward_str} | "
                f"{top} |"
            )
    else:
        lines.append(empty_msg)
    lines.append("")


def _render_comparative_sections(
    lines: list[str],
    per_benchmark: list[dict],
    per_agent: list[dict],
    matrix: dict[str, dict[str, dict]],
    divergences: list[dict],
) -> None:
    """Render the four comparative-analysis sections into `lines`."""
    _render_summary_table(
        lines,
        "Per-Benchmark Summary",
        "Benchmark",
        per_benchmark,
        "benchmark",
        "No annotations to summarize by benchmark.",
    )
    _render_summary_table(
        lines,
        "Per-Agent Summary",
        "Agent",
        per_agent,
        "label",
        "No annotations to summarize by agent.",
    )

    # Agent x Benchmark Matrix
    lines.append("## Agent x Benchmark Matrix")
    lines.append("")
    has_low_confidence = False
    if matrix:
        agents = sorted(matrix.keys())
        benchmarks_set: set[str] = set()
        for cells in matrix.values():
            benchmarks_set.update(cells.keys())
        benchmarks = sorted(benchmarks_set)
        lines.append(
            f"Values are pass rates; `†` marks cells with n < {LOW_CONFIDENCE_N} trials."
        )
        lines.append("")
        lines.append("| Agent | " + " | ".join(_md_cell(b) for b in benchmarks) + " |")
        lines.append("|-------|" + "|".join(["-------"] * len(benchmarks)) + "|")
        for agent in agents:
            cells = matrix[agent]
            row_cells: list[str] = []
            for bench in benchmarks:
                cell = cells.get(bench)
                if cell is None:
                    row_cells.append("—")
                else:
                    low = cell.get("low_confidence")
                    has_low_confidence = has_low_confidence or bool(low)
                    marker = "†" if low else ""
                    row_cells.append(f"{cell['pass_rate']:.1%}{marker}")
            lines.append(f"| {_md_cell(agent)} | " + " | ".join(row_cells) + " |")
    else:
        lines.append("No agent-by-benchmark data available.")
    lines.append("")

    if has_low_confidence:
        lines.append(f"`†` n < {LOW_CONFIDENCE_N} trials — treat as low-confidence.")
        lines.append("")

    # Top Divergences
    lines.append("## Top Divergences")
    lines.append("")
    if divergences:
        lines.append("Largest pairwise pass-rate delta per benchmark across agents.")
        lines.append("")
        lines.append(
            "| Benchmark | Delta | Higher | Lower | Higher Rate | Lower Rate |"
        )
        lines.append(
            "|-----------|-------|--------|-------|-------------|------------|"
        )
        for row in divergences:
            lines.append(
                f"| {_md_cell(row['benchmark'])} | {row['max_delta']:.1%} | "
                f"{_md_cell(row['pair_a'])} | {_md_cell(row['pair_b'])} | "
                f"{row['pair_a_rate']:.1%} | {row['pair_b_rate']:.1%} |"
            )
    else:
        lines.append("No benchmarks with multiple agents to compare.")
    lines.append("")


def _render_markdown(
    stats: dict,
    cat_counts: dict[str, int],
    cat_by_config: dict[str, dict[str, int]],
    cat_by_suite: dict[str, dict[str, int]],
    top_failures: list[dict],
    top_successes: list[dict],
    polarity_map: dict[str, str],
    generated_at: str,
    paired_comparisons: list[dict] | None = None,
    cat_counts_with_denominators: dict[str, dict[str, Any]] | None = None,
    dimension_summary: dict[str, dict[str, Any]] | None = None,
    per_benchmark_summary: list[dict] | None = None,
    per_agent_summary: list[dict] | None = None,
    agent_benchmark_matrix: dict[str, dict[str, dict]] | None = None,
    top_divergences: list[dict] | None = None,
) -> str:
    """Render the Markdown reliability report."""
    lines: list[str] = []

    # Header
    lines.append("# Agent Reliability Observatory — Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    # Corpus Stats
    lines.append("## Corpus Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total annotated trials | {stats['total_trials']:,} |")
    lines.append(f"| Passed | {stats['passed']:,} |")
    lines.append(f"| Failed | {stats['failed']:,} |")
    lines.append(f"| Pass rate | {stats['pass_rate']:.1%} |")
    lines.append(f"| Average reward | {stats['avg_reward']:.4f} |")
    lines.append(f"| Configs | {len(stats['configs'])} |")
    lines.append(f"| Benchmarks | {len(stats['benchmarks'])} |")
    lines.append("")

    # Comparative analysis sections (headline slices — near the top)
    _render_comparative_sections(
        lines,
        per_benchmark_summary or [],
        per_agent_summary or [],
        agent_benchmark_matrix or {},
        top_divergences or [],
    )

    # Category Frequency — split by denominator type
    denom_data = cat_counts_with_denominators or {}

    # Partition categories into trajectory-dependent and reward-dependent
    traj_cats = {
        name: count
        for name, count in cat_counts.items()
        if CHECKER_REQUIRES_TRAJECTORY.get(name, True)
    }
    reward_cats = {
        name: count
        for name, count in cat_counts.items()
        if not CHECKER_REQUIRES_TRAJECTORY.get(name, True)
    }

    lines.append("## Trajectory-Dependent Categories")
    lines.append("")
    if traj_cats:
        # Determine denominator from any trajectory category entry
        traj_denom = next(
            (denom_data[n]["denominator"] for n in traj_cats if n in denom_data),
            stats["total_trials"],
        )
        lines.append(f"Denominator: {traj_denom:,} trials with trajectory data")
        lines.append("")
        lines.append("| # | Category | Polarity | Count | Rate |")
        lines.append("|---|----------|----------|-------|------|")
        for i, (name, count) in enumerate(traj_cats.items(), 1):
            pol = polarity_map.get(name, "unknown")
            info = denom_data.get(name, {})
            rate = info.get("rate", 0.0)
            lines.append(f"| {i} | {name} | {pol} | {count:,} | {rate:.1%} |")
    else:
        lines.append("No trajectory-dependent categories found.")
    lines.append("")

    lines.append("## Reward-Dependent Categories")
    lines.append("")
    if reward_cats:
        reward_denom = stats["total_trials"]
        lines.append(f"Denominator: {reward_denom:,} total trials")
        lines.append("")
        lines.append("| # | Category | Polarity | Count | Rate |")
        lines.append("|---|----------|----------|-------|------|")
        for i, (name, count) in enumerate(reward_cats.items(), 1):
            pol = polarity_map.get(name, "unknown")
            info = denom_data.get(name, {})
            rate = info.get("rate", 0.0)
            lines.append(f"| {i} | {name} | {pol} | {count:,} | {rate:.1%} |")
    else:
        lines.append("No reward-dependent categories found.")
    lines.append("")

    # Config Breakdown
    lines.append("## Category Breakdown by Config")
    lines.append("")
    for cfg, cats in cat_by_config.items():
        lines.append(f"### {cfg}")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for name, count in cats.items():
            lines.append(f"| {name} | {count:,} |")
        lines.append("")

    # Configuration Comparison (Paired)
    if paired_comparisons:
        lines.append("## Configuration Comparison (Paired)")
        lines.append("")
        lines.append(
            "Compares failure-mode categories between config pairs on shared tasks."
        )
        lines.append(
            "Deltas show how many more shared tasks had a category in one config vs the other."
        )
        lines.append("")
        for pair in paired_comparisons:
            lines.append(
                f"### {pair['config_a']} vs {pair['config_b']} "
                f"({pair['shared_tasks']} shared tasks)"
            )
            lines.append("")
            if pair["introduced_by_a"]:
                lines.append(f"**More frequent in {pair['config_a']}:**")
                lines.append("")
                lines.append("| Category | Delta |")
                lines.append("|----------|-------|")
                for item in pair["introduced_by_a"]:
                    lines.append(f"| {item['category']} | +{item['delta']} |")
                lines.append("")
            if pair["introduced_by_b"]:
                lines.append(f"**More frequent in {pair['config_b']}:**")
                lines.append("")
                lines.append("| Category | Delta |")
                lines.append("|----------|-------|")
                for item in pair["introduced_by_b"]:
                    lines.append(f"| {item['category']} | +{item['delta']} |")
                lines.append("")

    # Suite Breakdown (top 10 suites by annotation volume)
    lines.append("## Category Breakdown by Benchmark Suite (Top 10)")
    lines.append("")
    for i, (suite, cats) in enumerate(cat_by_suite.items()):
        if i >= 10:
            break
        total = sum(cats.values())
        lines.append(f"### {suite} ({total:,} annotations)")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for name, count in list(cats.items())[:8]:  # Top 8 per suite
            lines.append(f"| {name} | {count:,} |")
        lines.append("")

    # Dimension Summary
    if dimension_summary:
        lines.append("## Dimension Summary")
        lines.append("")
        lines.append("| Dimension | Trial Count | Failure Rate |")
        lines.append("|-----------|-------------|--------------|")
        for dim_name, info in sorted(
            dimension_summary.items(), key=lambda x: -x[1]["trial_count"]
        ):
            lines.append(
                f"| {dim_name} | {info['trial_count']:,} "
                f"| {info['failure_rate']:.1%} |"
            )
        lines.append("")

    # Top Failure Categories
    lines.append("## Top Failure Categories")
    lines.append("")
    for entry in top_failures:
        lines.append(f"### {entry['name']} ({entry['count']:,} trials)")
        lines.append("")
        for ex in entry["examples"]:
            lines.append(
                f"- **{ex['task_id']}** (config: {ex['config_name']}, "
                f"reward: {ex['reward']:.2f}): {ex['evidence']}"
            )
        lines.append("")

    # Success Mode Summary
    lines.append("## Success Mode Summary")
    lines.append("")
    if top_successes:
        for entry in top_successes:
            lines.append(f"### {entry['name']} ({entry['count']:,} trials)")
            lines.append("")
            for ex in entry["examples"]:
                lines.append(
                    f"- **{ex['task_id']}** (config: {ex['config_name']}, "
                    f"reward: {ex['reward']:.2f}): {ex['evidence']}"
                )
            lines.append("")
    else:
        lines.append("No success-mode annotations found in this corpus.")
        lines.append("")

    return "\n".join(lines)


def generate_report(annotations: dict, output_dir: Path) -> tuple[Path, Path]:
    """Generate a Markdown reliability report and JSON companion.

    Parameters
    ----------
    annotations : dict
        Full annotation document matching annotation_schema.json.
    output_dir : Path
        Directory where reliability_report.md and .json will be written.

    Returns
    -------
    tuple[Path, Path]
        Paths to the generated .md and .json files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_list = annotations.get("annotations", [])
    polarity_map = _load_taxonomy_polarity()
    generated_at = datetime.now(timezone.utc).isoformat()

    stats = _corpus_stats(ann_list)
    cat_counts = _category_counts(ann_list)
    cat_counts_denom = _category_counts_with_denominators(ann_list)
    cat_by_config = _category_by_config(ann_list)
    cat_by_suite = _category_by_suite(ann_list)
    top_failures = _top_categories_with_examples(ann_list, polarity_map, "failure")
    top_successes = _top_categories_with_examples(ann_list, polarity_map, "success")
    paired_comparisons = _paired_comparison(ann_list)

    frame = _build_trials_frame(ann_list)
    per_benchmark = _per_benchmark_summary(frame)
    per_agent = _per_agent_summary(frame)
    matrix = _agent_benchmark_matrix(frame)
    divergences = _top_divergences(matrix)

    # Co-occurrence and dimension aggregation
    cooccurrence = co_occurrence_matrix(ann_list)
    try:
        from agent_diagnostics.taxonomy import load_taxonomy

        taxonomy = load_taxonomy()
    except Exception:
        taxonomy = {"dimensions": []}
    dim_summary = dimension_aggregation(ann_list, taxonomy)

    # Add trajectory-available count to corpus stats
    stats["trajectory_available"] = _count_trajectory_available(ann_list)

    # Markdown report
    md_content = _render_markdown(
        stats,
        cat_counts,
        cat_by_config,
        cat_by_suite,
        top_failures,
        top_successes,
        polarity_map,
        generated_at,
        paired_comparisons=paired_comparisons,
        cat_counts_with_denominators=cat_counts_denom,
        dimension_summary=dim_summary,
        per_benchmark_summary=per_benchmark,
        per_agent_summary=per_agent,
        agent_benchmark_matrix=matrix,
        top_divergences=divergences,
    )
    md_path = output_dir / "reliability_report.md"
    md_path.write_text(md_content)

    # JSON companion
    json_data = {
        "generated_at": generated_at,
        "corpus_stats": stats,
        "category_counts": cat_counts_denom,
        "category_by_config": cat_by_config,
        "category_by_suite": cat_by_suite,
        "paired_comparisons": paired_comparisons,
        "co_occurrence": cooccurrence,
        "dimension_summary": dim_summary,
        "per_benchmark_summary": per_benchmark,
        "per_agent_summary": per_agent,
        "agent_benchmark_matrix": matrix,
        "top_divergences": divergences,
    }
    json_path = output_dir / "reliability_report.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return md_path, json_path
