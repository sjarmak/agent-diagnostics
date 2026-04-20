"""Agreement and calibration analysis for the Agent Reliability Observatory.

Supports two comparison modes:

1. **Heuristic-vs-LLM** (original): Aligns annotations from two sources
   (typically heuristic and LLM) by trial_path and computes per-category
   agreement metrics: true positives, false positives, false negatives,
   precision, recall, and F1.

2. **Cross-model** (new): Compares annotations from two different model
   families using Cohen's kappa for inter-rater reliability measurement.
   Identifies categories where annotators disagree beyond chance.

Used to identify categories where annotators are unreliable and to
calibrate confidence thresholds.

Error-row policy
----------------
When an input file is ``observatory-annotation-v2`` and a trial carries
``annotation_result_status == "error"``, the trial is excluded from the
shared set at the trial granularity.  This prevents silently-failed
annotations from being counted as "category absent" on one side, which
would inflate FP/FN against the other side's labels.  Excluded counts
are surfaced in the returned summary.

Legacy ``observatory-annotation-v1`` rows (no ``annotation_result_status``
field) default to ``"ok"`` on read — we cannot retroactively distinguish
silent-failure empties from genuine no-categories rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypedDict


class ReliabilityDiagram(TypedDict):
    """Per-bin confidence vs accuracy breakdown returned by
    :func:`reliability_diagram`.

    All list fields are parallel and of length ``n_bins`` except ``bin_edges``,
    which has length ``n_bins + 1``.
    """

    bin_edges: list[float]
    bin_centers: list[float]
    mean_confidence: list[float]
    accuracy: list[float]
    count: list[int]
    n_bins: int
    total: int


def _load_annotations(
    path: str | Path,
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, dict[str, float]]]:
    """Load an annotation file.

    Returns ``(categories_by_trial, status_by_trial, confidences_by_trial)``.

    Accepts three input shapes:

    - ``.jsonl``: one annotation record per line (as emitted by
      ``observatory annotate --output X.jsonl``).
    - ``.json`` document: ``{"annotations": [...], "schema_version": ...}``.
    - ``.json`` bare list: ``[{...}, {...}]`` (legacy tooling shape).

    ``status_by_trial`` defaults to ``"ok"`` for legacy
    ``observatory-annotation-v1`` rows that lack the field.
    ``confidences_by_trial[trial][category_name]`` is the emitted confidence
    scalar (falls back to ``1.0`` when a category is listed without one — the
    legacy heuristic annotator did not always emit confidences).
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        annotations: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                annotations.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
        annotations = data.get("annotations", data) if isinstance(data, dict) else data
    if not isinstance(annotations, list):
        raise ValueError(f"Expected annotations list, got {type(annotations).__name__}")

    cats_by_trial: dict[str, set[str]] = {}
    status_by_trial: dict[str, str] = {}
    confidences_by_trial: dict[str, dict[str, float]] = {}
    for ann in annotations:
        trial = ann.get("trial_path", "")
        if not trial:
            continue
        cats: set[str] = set()
        confs: dict[str, float] = {}
        for c in ann.get("categories", []):
            name = c.get("name")
            if not name:
                continue
            cats.add(name)
            raw_conf = c.get("confidence")
            # Missing/None confidence defaults to 1.0 (legacy assignments
            # predating confidence emission are treated as full-confidence).
            try:
                confs[name] = float(raw_conf) if raw_conf is not None else 1.0
            except (TypeError, ValueError):
                confs[name] = 1.0
        cats_by_trial[trial] = cats
        confidences_by_trial[trial] = confs
        status_by_trial[trial] = ann.get("annotation_result_status", "ok")
    return cats_by_trial, status_by_trial, confidences_by_trial


# ---------------------------------------------------------------------------
# Calibration scoring rules (ECE, Brier, reliability diagrams)
# ---------------------------------------------------------------------------


def _validate_pairs(
    pairs: Iterable[tuple[float, int]],
) -> list[tuple[float, int]]:
    """Eagerly consume *pairs* and validate each ``(confidence, observed)``.

    Raises
    ------
    ValueError
        If *pairs* is empty, any confidence is outside ``[0, 1]``, or any
        observed label is not ``0`` or ``1``.
    """
    items = list(pairs)
    if not items:
        raise ValueError("empty input: no predictions to score")
    for conf, obs in items:
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {conf!r}")
        if obs not in (0, 1):
            raise ValueError(f"observed label must be 0 or 1, got {obs!r}")
    return items


def _bin_pairs(
    items: list[tuple[float, int]],
    n_bins: int,
) -> tuple[list[float], list[int], list[int]]:
    """Bucket validated pairs into ``n_bins`` equal-width bins on ``[0, 1]``.

    Returns ``(bin_conf_sum, bin_correct_sum, bin_count)`` — three parallel
    lists of length ``n_bins``.  The upper edge ``1.0`` is included in the
    last bin; floating-point drift near edges is clamped defensively.
    """
    bin_conf_sum = [0.0] * n_bins
    bin_correct_sum = [0] * n_bins
    bin_count = [0] * n_bins
    for conf, obs in items:
        # Multiply rather than floor-divide: `0.3 // 0.1` returns 2.0 under
        # IEEE-754 because 0.3 is stored as 0.2999…, which would silently
        # misassign common LLM confidences (0.3, 0.7, 0.9) to the wrong bin.
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_conf_sum[idx] += conf
        bin_correct_sum[idx] += obs
        bin_count[idx] += 1
    return bin_conf_sum, bin_correct_sum, bin_count


def compute_ece(
    pairs: Iterable[tuple[float, int]],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over equal-width bins on ``[0, 1]``.

    ECE = sum_b (|bin_b| / N) * |mean_confidence_b - accuracy_b|

    Bins are right-exclusive with the upper edge at 1.0 included in the last
    bin.  Empty bins contribute 0.

    Parameters
    ----------
    pairs
        Iterable of ``(confidence, observed)`` where ``observed ∈ {0, 1}``.
    n_bins
        Number of equal-width bins on ``[0, 1]``.  Must be positive.

    Returns
    -------
    float
        ECE in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the iterable is empty, any confidence is outside ``[0, 1]``, any
        label is non-binary, or ``n_bins <= 0``.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    items = _validate_pairs(pairs)
    total = len(items)
    bin_conf_sum, bin_correct_sum, bin_count = _bin_pairs(items, n_bins)

    ece = 0.0
    for i in range(n_bins):
        n = bin_count[i]
        if n == 0:
            continue
        mean_conf = bin_conf_sum[i] / n
        accuracy = bin_correct_sum[i] / n
        ece += (n / total) * abs(mean_conf - accuracy)
    return ece


def compute_brier(pairs: Iterable[tuple[float, int]]) -> float:
    """Brier score: mean squared error between confidence and binary outcome.

    Brier = mean((confidence - observed)^2)

    Parameters
    ----------
    pairs
        Iterable of ``(confidence, observed)`` where ``observed ∈ {0, 1}``.

    Returns
    -------
    float
        Brier score in ``[0, 1]`` (lower is better).

    Raises
    ------
    ValueError
        If the iterable is empty, any confidence is outside ``[0, 1]``, or any
        label is non-binary.
    """
    items = _validate_pairs(pairs)
    total = len(items)
    return sum((conf - obs) ** 2 for conf, obs in items) / total


def reliability_diagram(
    pairs: Iterable[tuple[float, int]],
    *,
    n_bins: int = 10,
) -> ReliabilityDiagram:
    """Per-bin confidence vs observed-accuracy breakdown for plotting.

    Bins are equal-width on ``[0, 1]``; the upper edge (1.0) is included in
    the last bin.  Empty bins report ``count = 0`` and zero for both
    ``mean_confidence`` and ``accuracy``.

    Parameters
    ----------
    pairs
        Iterable of ``(confidence, observed)`` where ``observed ∈ {0, 1}``.
    n_bins
        Number of equal-width bins on ``[0, 1]``.  Must be positive.

    Returns
    -------
    ReliabilityDiagram
        See the :class:`ReliabilityDiagram` TypedDict for the exact key set
        and field types.

    Raises
    ------
    ValueError
        Same as :func:`compute_ece`.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    items = _validate_pairs(pairs)
    bin_width = 1.0 / n_bins

    bin_edges = [i * bin_width for i in range(n_bins + 1)]
    bin_edges[-1] = 1.0  # lock the top edge to avoid float drift
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]

    bin_conf_sum, bin_correct_sum, bin_count = _bin_pairs(items, n_bins)

    mean_confidence = [
        (bin_conf_sum[i] / bin_count[i]) if bin_count[i] else 0.0 for i in range(n_bins)
    ]
    accuracy = [
        (bin_correct_sum[i] / bin_count[i]) if bin_count[i] else 0.0 for i in range(n_bins)
    ]

    return ReliabilityDiagram(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        mean_confidence=mean_confidence,
        accuracy=accuracy,
        count=bin_count,
        n_bins=n_bins,
        total=len(items),
    )


def _partition_by_error(
    shared_all: set[str],
    a_status: dict[str, str],
    b_status: dict[str, str],
) -> tuple[set[str], int, int]:
    """Drop trials whose status on either side is ``"error"``.

    Returns ``(shared, a_errors, b_errors)``.
    """
    a_errors = sum(1 for t in shared_all if a_status.get(t, "ok") == "error")
    b_errors = sum(1 for t in shared_all if b_status.get(t, "ok") == "error")
    shared = {
        t
        for t in shared_all
        if a_status.get(t, "ok") != "error" and b_status.get(t, "ok") != "error"
    }
    return shared, a_errors, b_errors


def compare_annotations(
    heuristic_file: str | Path,
    llm_file: str | Path,
) -> dict[str, Any]:
    """Compare heuristic and LLM annotations and compute agreement metrics.

    Aligns annotations by ``trial_path``. For each shared trial, compares
    the set of category names from each source:

    - **TP**: both sources assign the category
    - **FP**: heuristic assigns it, LLM does not
    - **FN**: LLM assigns it, heuristic does not

    Parameters
    ----------
    heuristic_file : str or Path
        Path to the heuristic annotation JSON file.
    llm_file : str or Path
        Path to the LLM annotation JSON file.

    Returns
    -------
    dict
        Summary with keys:

        - ``shared_trials``: number of trials present in both files
        - ``categories``: dict mapping category name to metrics dict with
          agreement fields (``true_positive``, ``false_positive``,
          ``false_negative``, ``precision``, ``recall``, ``f1``) and
          calibration fields (``support``, ``ece``, ``brier``,
          ``reliability_bins``).  Calibration treats the heuristic's
          emitted confidence as the prediction (0.0 when the category is
          absent) and the LLM's presence label as the binary observation.
        - ``macro_avg``: macro-averaged precision, recall, f1 across
          categories with at least one TP+FP+FN
    """
    heuristic, heuristic_status, heuristic_confs = _load_annotations(heuristic_file)
    llm, llm_status, _llm_confs = _load_annotations(llm_file)

    shared_all = set(heuristic.keys()) & set(llm.keys())

    # Exclude trials where either side reports an error: their empty category
    # sets would inflate FP/FN counts against the other side's labels.
    shared, heuristic_error_count, llm_error_count = _partition_by_error(
        shared_all, heuristic_status, llm_status
    )
    excluded_errored_trials = len(shared_all) - len(shared)

    # Collect all category names seen in shared trials
    all_cats: set[str] = set()
    for trial in shared:
        all_cats |= heuristic[trial]
        all_cats |= llm[trial]

    # Compute per-category TP / FP / FN and calibration metrics.  Pairs are
    # built over the non-error shared subset: prediction = heuristic confidence
    # (or 0.0 when the category is absent); reference label = 1 iff LLM
    # assigns the category.
    category_metrics: dict[str, dict[str, Any]] = {}
    for cat in sorted(all_cats):
        tp = fp = fn = 0
        pairs: list[tuple[float, int]] = []
        for trial in shared:
            in_h = cat in heuristic[trial]
            in_l = cat in llm[trial]
            if in_h and in_l:
                tp += 1
            elif in_h and not in_l:
                fp += 1
            elif not in_h and in_l:
                fn += 1
            conf = heuristic_confs.get(trial, {}).get(cat, 0.0)
            # Guard against pathological inputs (out-of-range confidences
            # in upstream files) — clamp to [0, 1] so scoring never raises.
            conf = max(0.0, min(1.0, conf))
            pairs.append((conf, 1 if in_l else 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ``all_cats`` is built from the shared trials, so every category
        # reaches this point with at least one paired observation.
        ece = compute_ece(pairs, n_bins=10)
        brier = compute_brier(pairs)
        reliability = reliability_diagram(pairs, n_bins=10)
        support = len(pairs)

        category_metrics[cat] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "ece": round(ece, 4),
            "brier": round(brier, 4),
            "reliability_bins": reliability,
        }

    # Macro average over categories that had any signal
    active = [
        m
        for m in category_metrics.values()
        if m["true_positive"] + m["false_positive"] + m["false_negative"] > 0
    ]
    if active:
        macro_p = sum(m["precision"] for m in active) / len(active)
        macro_r = sum(m["recall"] for m in active) / len(active)
        macro_f1 = sum(m["f1"] for m in active) / len(active)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    shared_all_count = len(shared_all) or 1
    return {
        "shared_trials": len(shared),
        "excluded_errored_trials": excluded_errored_trials,
        "heuristic_error_count": heuristic_error_count,
        "llm_error_count": llm_error_count,
        "heuristic_error_rate": round(heuristic_error_count / shared_all_count, 4),
        "llm_error_rate": round(llm_error_count / shared_all_count, 4),
        "categories": category_metrics,
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
    }


def format_markdown(summary: dict[str, Any]) -> str:
    """Render the agreement summary as a Markdown table.

    Parameters
    ----------
    summary : dict
        Output of ``compare_annotations()``.

    Returns
    -------
    str
        Markdown string with a metrics table and macro averages.
    """
    lines: list[str] = []
    lines.append("## Heuristic vs LLM Agreement\n")
    lines.append(f"Shared trials: {summary['shared_trials']}\n")
    if summary.get("excluded_errored_trials", 0) > 0:
        lines.append(
            f"Excluded errored trials: {summary['excluded_errored_trials']} "
            f"(heuristic errors={summary.get('heuristic_error_count', 0)}, "
            f"llm errors={summary.get('llm_error_count', 0)})\n"
        )

    lines.append("| Category | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|----------|---:|---:|---:|----------:|-------:|---:|")

    cats = summary["categories"]
    # Sort by F1 descending, then name ascending
    for name in sorted(cats, key=lambda n: (-cats[n]["f1"], n)):
        m = cats[name]
        lines.append(
            f"| {name} | {m['true_positive']} | {m['false_positive']} "
            f"| {m['false_negative']} | {m['precision']:.2f} "
            f"| {m['recall']:.2f} | {m['f1']:.2f} |"
        )

    macro = summary["macro_avg"]
    lines.append(
        f"| **Macro avg** | | | | **{macro['precision']:.2f}** "
        f"| **{macro['recall']:.2f}** | **{macro['f1']:.2f}** |"
    )

    # Calibration section (ECE / Brier / direction of miscalibration).  Only
    # emitted when at least one category carries the new fields — this keeps
    # the report lean when callers pass a legacy summary dict that predates
    # the calibration extension.
    has_calibration = any("ece" in m and "reliability_bins" in m for m in cats.values())
    if has_calibration:
        lines.append("")
        lines.append("### Calibration")
        lines.append("")
        lines.append(
            "Direction: arrow points from mean confidence toward observed "
            "accuracy; >> means overconfident, << means underconfident, "
            "= means well calibrated."
        )
        lines.append("")
        lines.append("| Category | Support | ECE | Brier | Direction |")
        lines.append("|----------|--------:|----:|------:|:---------:|")
        for name in sorted(cats, key=lambda n: (-cats[n].get("ece", 0.0), n)):
            m = cats[name]
            if "ece" not in m:
                continue
            lines.append(
                f"| {name} | {m.get('support', 0)} | {m['ece']:.3f} "
                f"| {m['brier']:.3f} | {_direction_arrow(m)} |"
            )

    return "\n".join(lines) + "\n"


def _direction_arrow(metrics: dict[str, Any]) -> str:
    """Render a compact text indicator of miscalibration direction.

    Compares the overall mean confidence in the reliability bins to the
    overall observed accuracy (both weighted by bin count).  Returns
    ``">>"`` for overconfident, ``"<<"`` for underconfident, ``"="`` when
    within a small epsilon, and ``"-"`` when there is no support.

    Callers in this module always pass a metrics dict whose
    ``reliability_bins`` value was produced by :func:`reliability_diagram`,
    so the structural keys are guaranteed present.  The ``total == 0``
    branch handles the degenerate case where a caller hand-builds a
    ``ReliabilityDiagram`` with no samples.
    """
    rb: ReliabilityDiagram = metrics["reliability_bins"]
    total = rb["total"]
    if not total:
        return "-"
    counts = rb["count"]
    mean_conf = rb["mean_confidence"]
    acc = rb["accuracy"]
    weighted_conf = sum(c * n for c, n in zip(mean_conf, counts)) / total
    weighted_acc = sum(a * n for a, n in zip(acc, counts)) / total
    diff = weighted_conf - weighted_acc
    if abs(diff) < 0.01:
        return "="
    return ">>" if diff > 0 else "<<"


# ---------------------------------------------------------------------------
# Cross-model calibration (Cohen's kappa)
# ---------------------------------------------------------------------------

UNCALIBRATED_THRESHOLD = 0.4
"""Categories with kappa below this value are flagged as 'uncalibrated'."""


def cohen_kappa(a_labels: list[int], b_labels: list[int]) -> float:
    """Compute Cohen's kappa for two binary label vectors.

    Parameters
    ----------
    a_labels : list of int
        Binary labels (0 or 1) from rater A, one per item.
    b_labels : list of int
        Binary labels (0 or 1) from rater B, same length as *a_labels*.

    Returns
    -------
    float
        Cohen's kappa coefficient. Returns 0.0 when both raters assign the
        same label to every item (perfect agreement by chance, undefined kappa).

    Raises
    ------
    ValueError
        If the input vectors differ in length or are empty.
    """
    if len(a_labels) != len(b_labels):
        raise ValueError(
            f"Label vectors must be the same length (got {len(a_labels)} and {len(b_labels)})"
        )
    if not a_labels:
        raise ValueError("Label vectors must not be empty")

    n = len(a_labels)

    # Confusion matrix cells
    both_pos = sum(1 for a, b in zip(a_labels, b_labels) if a == 1 and b == 1)
    both_neg = sum(1 for a, b in zip(a_labels, b_labels) if a == 0 and b == 0)
    a_pos_b_neg = sum(1 for a, b in zip(a_labels, b_labels) if a == 1 and b == 0)
    a_neg_b_pos = sum(1 for a, b in zip(a_labels, b_labels) if a == 0 and b == 1)

    observed_agreement = (both_pos + both_neg) / n

    # Marginal probabilities
    a_pos_rate = (both_pos + a_pos_b_neg) / n
    b_pos_rate = (both_pos + a_neg_b_pos) / n

    expected_agreement = (a_pos_rate * b_pos_rate) + ((1 - a_pos_rate) * (1 - b_pos_rate))

    if expected_agreement >= 1.0:
        return 0.0

    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def compare_cross_model(
    model_a_file: str | Path,
    model_b_file: str | Path,
    *,
    uncalibrated_threshold: float = UNCALIBRATED_THRESHOLD,
) -> dict[str, Any]:
    """Compare annotations from two model families using Cohen's kappa.

    For each category observed in the shared trials, builds a binary vector
    (1 = category present, 0 = absent) per model and computes Cohen's kappa.

    Parameters
    ----------
    model_a_file : str or Path
        Path to annotation JSON from model A.
    model_b_file : str or Path
        Path to annotation JSON from model B.
    uncalibrated_threshold : float
        Kappa threshold below which a category is flagged as 'uncalibrated'.

    Returns
    -------
    dict
        Summary with keys:

        - ``shared_trials``: count of trials present in both files
        - ``categories``: dict mapping category name to
          ``kappa``, ``agreement``, ``a_count``, ``b_count``, ``calibrated``
        - ``uncalibrated_categories``: list of category names with kappa
          below the threshold
        - ``macro_kappa``: average kappa across all categories with signal
    """
    model_a, model_a_status, _ = _load_annotations(model_a_file)
    model_b, model_b_status, _ = _load_annotations(model_b_file)

    shared_all = set(model_a.keys()) & set(model_b.keys())
    shared_set, a_error_count, b_error_count = _partition_by_error(
        shared_all, model_a_status, model_b_status
    )
    shared_trials = sorted(shared_set)
    excluded_errored_trials = len(shared_all) - len(shared_trials)

    if not shared_trials:
        return {
            "shared_trials": 0,
            "excluded_errored_trials": excluded_errored_trials,
            "a_error_count": a_error_count,
            "b_error_count": b_error_count,
            "categories": {},
            "uncalibrated_categories": [],
            "macro_kappa": 0.0,
        }

    # Collect all category names from shared trials
    all_cats: set[str] = set()
    for trial in shared_trials:
        all_cats |= model_a[trial]
        all_cats |= model_b[trial]

    category_results: dict[str, dict[str, Any]] = {}
    uncalibrated: list[str] = []

    for cat in sorted(all_cats):
        a_vec = [1 if cat in model_a[t] else 0 for t in shared_trials]
        b_vec = [1 if cat in model_b[t] else 0 for t in shared_trials]

        kappa_val = cohen_kappa(a_vec, b_vec)
        agreement = sum(1 for a, b in zip(a_vec, b_vec) if a == b) / len(shared_trials)
        is_calibrated = kappa_val >= uncalibrated_threshold

        if not is_calibrated:
            uncalibrated.append(cat)

        category_results[cat] = {
            "kappa": round(kappa_val, 4),
            "agreement": round(agreement, 4),
            "a_count": sum(a_vec),
            "b_count": sum(b_vec),
            "calibrated": is_calibrated,
        }

    # Macro-average kappa across categories with any positive label
    active_kappas = [
        m["kappa"] for m in category_results.values() if m["a_count"] + m["b_count"] > 0
    ]
    macro_kappa = sum(active_kappas) / len(active_kappas) if active_kappas else 0.0

    return {
        "shared_trials": len(shared_trials),
        "excluded_errored_trials": excluded_errored_trials,
        "a_error_count": a_error_count,
        "b_error_count": b_error_count,
        "categories": category_results,
        "uncalibrated_categories": uncalibrated,
        "macro_kappa": round(macro_kappa, 4),
    }


def format_cross_model_markdown(
    summary: dict[str, Any],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> str:
    """Render cross-model calibration summary as Markdown.

    Parameters
    ----------
    summary : dict
        Output of ``compare_cross_model()``.
    model_a_name : str
        Display name for model A.
    model_b_name : str
        Display name for model B.

    Returns
    -------
    str
        Markdown report string.
    """
    lines: list[str] = []
    lines.append(f"## Cross-Model Calibration: {model_a_name} vs {model_b_name}\n")
    lines.append(f"Shared trials: {summary['shared_trials']}\n")
    lines.append(f"Macro kappa: **{summary['macro_kappa']:.4f}**\n")

    lines.append(
        f"| Category | Kappa | Agreement | {model_a_name} Count "
        f"| {model_b_name} Count | Calibrated |"
    )
    lines.append("|----------|------:|----------:|-----------:|-----------:|:----------:|")

    cats = summary["categories"]
    for name in sorted(cats, key=lambda n: (-cats[n]["kappa"], n)):
        m = cats[name]
        cal_icon = "Yes" if m["calibrated"] else "**No**"
        lines.append(
            f"| {name} | {m['kappa']:.4f} | {m['agreement']:.2f} "
            f"| {m['a_count']} | {m['b_count']} | {cal_icon} |"
        )

    if summary["uncalibrated_categories"]:
        lines.append("")
        lines.append(
            f"**Uncalibrated categories** (kappa < {UNCALIBRATED_THRESHOLD}): "
            + ", ".join(sorted(summary["uncalibrated_categories"]))
        )

    return "\n".join(lines) + "\n"
