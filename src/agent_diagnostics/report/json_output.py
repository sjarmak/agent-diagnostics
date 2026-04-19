"""JSON companion builder for the reliability report.

Assembles the machine-readable JSON payload from pre-computed aggregates.
Keeping the shape in one place ensures the JSON schema evolves coherently
with any additions to the markdown report without accidental divergence.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_json_payload"]


def build_json_payload(
    *,
    generated_at: str,
    stats: dict,
    cat_counts_with_denominators: dict[str, dict[str, Any]],
    cat_by_config: dict[str, dict[str, int]],
    cat_by_suite: dict[str, dict[str, int]],
    paired_comparisons: list[dict],
    cooccurrence: dict[str, dict[str, float]],
    dimension_summary: dict[str, dict[str, Any]],
    per_benchmark_summary: list[dict],
    per_agent_summary: list[dict],
    agent_benchmark_matrix: dict[str, dict[str, dict]],
    top_divergences: list[dict],
) -> dict[str, Any]:
    """Assemble the JSON companion dict.

    Field order is preserved for byte-identical output against the
    pre-split report.
    """
    return {
        "generated_at": generated_at,
        "corpus_stats": stats,
        "category_counts": cat_counts_with_denominators,
        "category_by_config": cat_by_config,
        "category_by_suite": cat_by_suite,
        "paired_comparisons": paired_comparisons,
        "co_occurrence": cooccurrence,
        "dimension_summary": dimension_summary,
        "per_benchmark_summary": per_benchmark_summary,
        "per_agent_summary": per_agent_summary,
        "agent_benchmark_matrix": agent_benchmark_matrix,
        "top_divergences": top_divergences,
    }
