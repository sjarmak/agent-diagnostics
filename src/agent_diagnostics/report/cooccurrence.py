"""Category co-occurrence and dimension rollup helpers.

These helpers form a self-contained slice of the report: they operate on
raw annotations and produce matrix / dimension summaries consumed by both
the Markdown renderer and the JSON companion. No dependency on other
``report`` submodules so they can be imported independently.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

__all__ = ["co_occurrence_matrix", "dimension_aggregation"]


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
