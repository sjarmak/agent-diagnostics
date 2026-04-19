"""Markdown renderer for the reliability report.

Builds the human-readable report body from pre-aggregated inputs. Imports
only from :mod:`agent_diagnostics.report.aggregation` (for
``LOW_CONFIDENCE_N``) and the annotator constants — never pulls in
``comparative`` / ``cooccurrence`` / ``orchestration`` so the package DAG
remains a forest with ``orchestration`` as its only multi-dep consumer.
"""

from __future__ import annotations

from typing import Any

from agent_diagnostics.annotator import CHECKER_REQUIRES_TRAJECTORY
from agent_diagnostics.report.aggregation import LOW_CONFIDENCE_N

__all__ = [
    "_md_cell",
    "_render_comparative_sections",
    "_render_markdown",
    "_render_summary_table",
]


def _md_cell(value: object) -> str:
    """Escape a value for safe embedding in a GFM table cell.

    Replaces pipe and newline characters that would otherwise break
    column alignment or smuggle new rows/sections into the output.
    """
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")


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
        header = (
            f"| {key_header} | Trials | Passed | Pass Rate | Mean Reward"
            f" | Mean Trajectory Length | Top Failures |"
        )
        separator = (
            "|"
            + "-" * (len(key_header) + 2)
            + "|--------|--------|-----------|-------------"
            + "|------------------------|--------------|"
        )
        lines.append(header)
        lines.append(separator)
        for row in rows:
            top = (
                ", ".join(
                    f"{_md_cell(f['name'])} ({f['count']})" for f in row["top_failures"]
                )
                or "—"
            )
            mean_reward = row.get("mean_reward")
            mean_reward_str = f"{mean_reward:.4f}" if mean_reward is not None else "—"
            mean_traj = row.get("mean_trajectory_length")
            mean_traj_str = f"{mean_traj:.2f}" if mean_traj is not None else "—"
            lines.append(
                f"| {_md_cell(row[key_field])} | {row['n_trials']:,} | "
                f"{row['passed']:,} | {row['pass_rate']:.1%} | {mean_reward_str} | "
                f"{mean_traj_str} | {top} |"
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
    annotation_summary: dict[str, int] | None = None,
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

    # Annotation quality — upstream annotator attempt outcomes, shown before
    # per-category / per-agent slices so readers can gauge how much of the
    # dataset is trustworthy before absorbing numbers with shrunk denominators.
    if annotation_summary:
        total_raw = annotation_summary.get("total", 0)
        total = total_raw or 1  # avoid zero-div in the percentage formatter
        lines.append("## Annotation Quality")
        lines.append("")
        lines.append(
            "Outcomes from the upstream LLM annotation pass. "
            "Error rows are excluded from calibration and blending."
        )
        lines.append("")
        lines.append("| Status | Count | Rate |")
        lines.append("|--------|-------|------|")
        for status in ("ok", "no_categories", "error"):
            count = annotation_summary.get(status, 0)
            lines.append(f"| {status} | {count:,} | {count / total:.1%} |")
        lines.append(f"| **total** | **{total_raw:,}** | |")
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
