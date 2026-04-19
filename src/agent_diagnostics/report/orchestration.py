"""Public entry point for the reliability report.

Composes the aggregation, comparative, markdown, json_output, and
cooccurrence submodules into the single ``generate_report`` call exposed
as the package's public API. All I/O lives here; the helper modules
stay pure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent_diagnostics.report.aggregation import (
    _build_trials_frame,
    _category_by_config,
    _category_by_suite,
    _category_counts,
    _category_counts_with_denominators,
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
)
from agent_diagnostics.report.cooccurrence import (
    co_occurrence_matrix,
    dimension_aggregation,
)
from agent_diagnostics.report.json_output import build_json_payload
from agent_diagnostics.report.markdown import _render_markdown

__all__ = ["generate_report"]


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
    annotation_summary = annotations.get("annotation_summary")
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
        annotation_summary=annotation_summary,
    )
    md_path = output_dir / "reliability_report.md"
    md_path.write_text(md_content)

    # JSON companion
    json_data = build_json_payload(
        generated_at=generated_at,
        stats=stats,
        cat_counts_with_denominators=cat_counts_denom,
        cat_by_config=cat_by_config,
        cat_by_suite=cat_by_suite,
        paired_comparisons=paired_comparisons,
        cooccurrence=cooccurrence,
        dimension_summary=dim_summary,
        per_benchmark_summary=per_benchmark,
        per_agent_summary=per_agent,
        agent_benchmark_matrix=matrix,
        top_divergences=divergences,
        annotation_summary=annotation_summary,
    )
    json_path = output_dir / "reliability_report.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return md_path, json_path
