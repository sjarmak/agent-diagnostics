"""Comparative analysis helpers for the reliability report.

Produces per-benchmark, per-agent, and agent x benchmark cross-tab views,
plus the top-divergences summary. Depends only on
:mod:`agent_diagnostics.report.aggregation` for the shared slice aggregator
— one-way so there is no cycle.
"""

from __future__ import annotations

from collections import defaultdict

from agent_diagnostics.report.aggregation import (
    LOW_CONFIDENCE_N,
    _aggregate_slice,
    _top_failure_categories,
)

__all__ = [
    "_agent_benchmark_matrix",
    "_per_agent_summary",
    "_per_benchmark_summary",
    "_top_divergences",
    "_top_failure_categories",
]


def _per_benchmark_summary(frame: list[dict]) -> list[dict]:
    """Per-benchmark aggregates, sorted desc by n_trials."""
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for row in frame:
        by_bench[row["benchmark"]].append(row)

    out = [{"benchmark": bench, **_aggregate_slice(trials)} for bench, trials in by_bench.items()]
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
        agents_here = [(agent, cells[bench]) for agent, cells in matrix.items() if bench in cells]
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
