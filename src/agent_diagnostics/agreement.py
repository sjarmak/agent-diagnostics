"""Inter-annotator agreement over the narrow-tall annotation store.

Computes pairwise Cohen's kappa per category between annotator identities
(e.g. ``heuristic:rule-engine`` vs ``llm:haiku-4``).  Calibration asks
"are the confidences honest?"; agreement asks the complementary question
"do independent annotators assign the same categories at all?".

Method
------
For each pair of annotator identities, the comparison universe is the
intersection of the trial sets each identity has annotated.  Within that
universe, each category becomes a binary present/absent judgment per trial,
and Cohen's kappa corrects the observed agreement for the agreement
expected by chance from each annotator's marginal rates.

Known limitation: the narrow-tall store only records *positive*
assignments, so an annotator's "annotated trial set" is approximated as
the trials where it assigned at least one category.  A trial an annotator
processed but left fully unlabeled is indistinguishable from one it never
saw and is excluded from that annotator's universe.  The report states the
universe size per pair so this is auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "observatory-agreement-v1"


def cohens_kappa(n11: int, n10: int, n01: int, n00: int) -> float | None:
    """Cohen's kappa for a 2x2 contingency table.

    ``n11``: both assigned, ``n10``: only A, ``n01``: only B, ``n00``: neither.
    Returns ``None`` when kappa is undefined (chance agreement is 1.0, i.e.
    both annotators are constant).
    """
    n = n11 + n10 + n01 + n00
    if n == 0:
        return None
    po = (n11 + n00) / n
    a_yes = (n11 + n10) / n
    b_yes = (n11 + n01) / n
    pe = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def _index_by_identity(rows: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """Map annotator_identity -> trial_id -> set of assigned category names."""
    by_identity: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        identity = row.get("annotator_identity", "")
        trial_id = row.get("trial_id", "")
        category = row.get("category_name", "")
        if not identity or not trial_id or not category:
            continue
        by_identity.setdefault(identity, {}).setdefault(trial_id, set()).add(category)
    return by_identity


def compute_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pairwise per-category Cohen's kappa across annotator identities.

    Parameters
    ----------
    rows:
        Narrow-tall annotation rows (``AnnotationStore.read_annotations()``).

    Returns
    -------
    dict
        ``{schema_version, generated_at, annotators, pairs}`` where each
        pair entry carries ``shared_trials``, per-category contingency
        counts with ``kappa``, and ``mean_kappa`` over defined categories.
    """
    by_identity = _index_by_identity(rows)
    identities = sorted(by_identity)

    pairs: list[dict[str, Any]] = []
    for i, ident_a in enumerate(identities):
        for ident_b in identities[i + 1 :]:
            trials_a = by_identity[ident_a]
            trials_b = by_identity[ident_b]
            shared = sorted(set(trials_a) & set(trials_b))

            categories: dict[str, dict[str, Any]] = {}
            if shared:
                cat_names = set()
                for t in shared:
                    cat_names |= trials_a[t] | trials_b[t]
                for cat in sorted(cat_names):
                    n11 = n10 = n01 = n00 = 0
                    for t in shared:
                        in_a = cat in trials_a[t]
                        in_b = cat in trials_b[t]
                        if in_a and in_b:
                            n11 += 1
                        elif in_a:
                            n10 += 1
                        elif in_b:
                            n01 += 1
                        else:
                            n00 += 1
                    kappa = cohens_kappa(n11, n10, n01, n00)
                    categories[cat] = {
                        "kappa": round(kappa, 4) if kappa is not None else None,
                        "both_present": n11,
                        "only_a": n10,
                        "only_b": n01,
                        "both_absent": n00,
                    }

            defined = [c["kappa"] for c in categories.values() if c["kappa"] is not None]
            pairs.append(
                {
                    "annotator_a": ident_a,
                    "annotator_b": ident_b,
                    "shared_trials": len(shared),
                    "categories": categories,
                    "mean_kappa": (round(sum(defined) / len(defined), 4) if defined else None),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "annotators": {ident: len(by_identity[ident]) for ident in identities},
        "pairs": pairs,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    """Render an agreement summary as a Markdown report."""
    lines = ["# Inter-Annotator Agreement", ""]
    lines.append(
        "Cohen's kappa per category between annotator identities, computed "
        "over the trials both annotators labeled. kappa = 1 is perfect "
        "agreement, 0 is chance level, < 0 is systematic disagreement; "
        "`n/a` means kappa is undefined (both annotators constant)."
    )
    lines.append("")

    annotators = summary.get("annotators", {})
    if annotators:
        lines.append("## Annotators")
        lines.append("")
        lines.append("| Identity | Trials labeled |")
        lines.append("|----------|---------------:|")
        for ident, count in annotators.items():
            lines.append(f"| {ident} | {count} |")
        lines.append("")

    for pair in summary.get("pairs", []):
        lines.append(f"## {pair['annotator_a']} vs {pair['annotator_b']}")
        lines.append("")
        lines.append(f"Shared trials: {pair['shared_trials']}")
        mean_kappa = pair.get("mean_kappa")
        lines.append(f"Mean kappa (defined categories): {_fmt_kappa(mean_kappa)}")
        lines.append("")
        if not pair["categories"]:
            lines.append("No overlapping trials — no categories to compare.")
            lines.append("")
            continue
        lines.append("| Category | kappa | Both | Only A | Only B | Neither |")
        lines.append("|----------|------:|-----:|-------:|-------:|--------:|")
        ranked = sorted(
            pair["categories"].items(),
            key=lambda kv: (kv[1]["kappa"] is None, -(kv[1]["kappa"] or 0.0)),
        )
        for cat, c in ranked:
            lines.append(
                f"| {cat} | {_fmt_kappa(c['kappa'])} | {c['both_present']} "
                f"| {c['only_a']} | {c['only_b']} | {c['both_absent']} |"
            )
        lines.append("")

    return "\n".join(lines)


def _fmt_kappa(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"
