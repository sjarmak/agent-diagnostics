"""Reliability report package.

This package splits the former monolithic ``report.py`` into focused modules
along responsibility lines:

- ``aggregation``: corpus/category aggregation helpers and trial framing.
- ``comparative``: per-benchmark / per-agent / cross-tab comparative analysis.
- ``markdown``: Markdown rendering.
- ``json_output``: JSON companion assembly.
- ``cooccurrence``: category co-occurrence + dimension rollup.
- ``orchestration``: ``generate_report`` entry point.

The public API previously exposed by ``report.py`` is re-exported here so that
existing callers (``from agent_diagnostics.report import generate_report``)
and test imports of internal helpers continue to resolve unchanged.
"""

from __future__ import annotations

from agent_diagnostics.report.aggregation import (
    LOW_CONFIDENCE_N,
    _aggregate_slice,
    _build_trials_frame,
    _category_by_config,
    _category_by_suite,
    _category_counts,
    _category_counts_with_denominators,
    _core_task_name,
    _corpus_stats,
    _count_trajectory_available,
    _load_taxonomy_polarity,
    _paired_comparison,
    _top_categories_with_examples,
)
from agent_diagnostics.report.comparative import (
    _agent_benchmark_matrix,
    _per_agent_summary,
    _per_benchmark_summary,
    _top_divergences,
    _top_failure_categories,
)
from agent_diagnostics.report.cooccurrence import (
    co_occurrence_matrix,
    dimension_aggregation,
)
from agent_diagnostics.report.markdown import (
    _md_cell,
    _render_comparative_sections,
    _render_markdown,
    _render_summary_table,
)
from agent_diagnostics.report.orchestration import generate_report

__all__ = [
    "LOW_CONFIDENCE_N",
    "_agent_benchmark_matrix",
    "_aggregate_slice",
    "_build_trials_frame",
    "_category_by_config",
    "_category_by_suite",
    "_category_counts",
    "_category_counts_with_denominators",
    "_core_task_name",
    "_corpus_stats",
    "_count_trajectory_available",
    "_load_taxonomy_polarity",
    "_md_cell",
    "_paired_comparison",
    "_per_agent_summary",
    "_per_benchmark_summary",
    "_render_comparative_sections",
    "_render_markdown",
    "_render_summary_table",
    "_top_categories_with_examples",
    "_top_divergences",
    "_top_failure_categories",
    "co_occurrence_matrix",
    "dimension_aggregation",
    "generate_report",
]
