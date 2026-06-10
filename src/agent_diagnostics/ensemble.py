"""Two-tier ensemble annotator: heuristic for structural categories,
classifier for learned categories.

Tier 1 (heuristic): Structural categories that are deterministic
    (exception_crash, rate_limited_run, edit_verify_loop_failure).
    These have simple, reliable signal rules.

Tier 2 (classifier): Categories whose held-out cross-validated F1
    (``eval_f1`` in the model artifact) clears the trust threshold, and —
    when an ECE gate is set — whose cross-validated calibration error
    (``cv_ece``) is acceptable. Fast, runs on full corpus.

Models trained before held-out validation existed carry only
``train_accuracy``; their categories are never trusted (training-set
metrics overstate reliability) — retrain to populate ``eval_f1``.

Usage::

    from agent_diagnostics.ensemble import ensemble_annotate, ensemble_all
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent_diagnostics.annotator import annotate_trial as heuristic_annotate
from agent_diagnostics.classifier import predict_trial
from agent_diagnostics.taxonomy import load_taxonomy

logger = logging.getLogger(__name__)

# Categories where heuristic rules are deterministic and reliable.
# These don't need the classifier — the rule IS the ground truth.
HEURISTIC_ONLY: frozenset[str] = frozenset(
    {
        "exception_crash",
        "rate_limited_run",
        "edit_verify_loop_failure",
    }
)


def trusted_classifier_categories(
    model: dict,
    classifier_min_f1: float = 0.7,
    classifier_max_ece: float | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Partition the model's categories into trusted and legacy-unvalidated.

    Returns ``(trusted, unvalidated)``:

    - ``trusted``: held-out ``eval_f1`` >= *classifier_min_f1* and, when
      *classifier_max_ece* is set, ``cv_ece`` <= *classifier_max_ece*.
    - ``unvalidated``: categories with no ``eval_f1`` at all (model predates
      held-out CV, or the category had too few examples to cross-validate).
      These are never trusted — a training-set metric is not a substitute.
    """
    trusted: set[str] = set()
    unvalidated: set[str] = set()
    for cat_name, clf_meta in model.get("classifiers", {}).items():
        eval_f1 = clf_meta.get("eval_f1")
        if eval_f1 is None:
            unvalidated.add(cat_name)
            continue
        if eval_f1 < classifier_min_f1:
            continue
        if classifier_max_ece is not None:
            cv_ece = clf_meta.get("cv_ece")
            if cv_ece is None or cv_ece > classifier_max_ece:
                continue
        trusted.add(cat_name)
    return frozenset(trusted), frozenset(unvalidated)


def ensemble_annotate(
    signals: dict,
    model: dict,
    classifier_threshold: float = 0.5,
    classifier_min_f1: float = 0.7,
    classifier_max_ece: float | None = None,
) -> list[dict]:
    """Annotate a single trial using the two-tier ensemble.

    Returns a list of {name, confidence, evidence, source} dicts.
    """
    results: dict[str, dict] = {}

    # Tier 1: Heuristic for structural/deterministic categories
    heur_cats = heuristic_annotate(signals)
    for c in heur_cats:
        if c.name in HEURISTIC_ONLY:
            results[c.name] = {
                "name": c.name,
                "confidence": c.confidence,
                "evidence": c.evidence,
                "source": "heuristic",
            }

    # Tier 2: Classifier, only for categories whose held-out CV metrics
    # clear the trust gate.
    trusted, _ = trusted_classifier_categories(
        model,
        classifier_min_f1=classifier_min_f1,
        classifier_max_ece=classifier_max_ece,
    )
    clf_cats = predict_trial(signals, model, threshold=classifier_threshold)
    for c in clf_cats:
        cat_name = c["name"]
        if cat_name in HEURISTIC_ONLY:
            continue  # Heuristic already handled these
        if cat_name not in trusted or cat_name in results:
            continue
        clf_meta = model["classifiers"][cat_name]
        eval_f1 = clf_meta["eval_f1"]
        cv_ece = clf_meta.get("cv_ece")
        evidence = c["evidence"] + f" [eval_f1={eval_f1:.2f}]"
        if cv_ece is not None:
            evidence += f" [cv_ece={cv_ece:.2f}]"
        results[cat_name] = {
            "name": cat_name,
            "confidence": c["confidence"],
            "evidence": evidence,
            "source": "classifier",
        }

    return list(results.values())


def ensemble_all(
    signals_list: list[dict],
    model: dict,
    classifier_threshold: float = 0.5,
    classifier_min_f1: float = 0.7,
    classifier_max_ece: float | None = None,
    annotations_out: str | None = None,
) -> dict:
    """Run ensemble annotation on the full corpus.

    Parameters
    ----------
    signals_list:
        List of signal dicts, one per trial.
    model:
        Trained classifier model dict.
    classifier_threshold:
        Prediction threshold for classifier tier.
    classifier_min_f1:
        Minimum held-out cross-validated F1 to trust a classifier category.
    classifier_max_ece:
        Optional maximum cross-validated ECE; categories above it are
        excluded even when their F1 clears ``classifier_min_f1``.
    annotations_out:
        Optional path to a JSONL file.  When provided, narrow-tall rows
        are written via :class:`~agent_diagnostics.annotation_store.AnnotationStore`
        in addition to the returned dict.
    """
    taxonomy = load_taxonomy()
    now = datetime.now(timezone.utc).isoformat()

    _, unvalidated = trusted_classifier_categories(
        model,
        classifier_min_f1=classifier_min_f1,
        classifier_max_ece=classifier_max_ece,
    )
    if unvalidated:
        logger.warning(
            "model has no held-out eval_f1 for %d categories (%s); they will "
            "never be trusted — retrain with `observatory train` to populate "
            "cross-validated metrics",
            len(unvalidated),
            ", ".join(sorted(unvalidated)),
        )

    annotations = []
    tier_counts: dict[str, int] = {"heuristic": 0, "classifier": 0}
    for sig in signals_list:
        cats = ensemble_annotate(
            sig,
            model,
            classifier_threshold=classifier_threshold,
            classifier_min_f1=classifier_min_f1,
            classifier_max_ece=classifier_max_ece,
        )
        if not cats:
            continue

        reward = sig.get("reward")
        # Strip internal 'source' key before output (not in annotation schema)
        clean_cats = [{k: v for k, v in c.items() if k != "source"} for c in cats]
        annotations.append(
            {
                "task_id": sig.get("task_id") or "unknown",
                "trial_path": sig.get("trial_path") or "",
                "config_name": sig.get("config_name"),
                "benchmark": sig.get("benchmark"),
                "model": sig.get("model"),
                "reward": float(reward) if reward is not None else 0.0,
                "passed": bool(sig.get("passed")),
                "categories": clean_cats,
                "annotated_at": now,
                "trial_id": sig.get("trial_id", ""),
            }
        )
        # Track tier counts from unstripped cats
        for c in cats:
            source = c.get("source", "unknown")
            tier_counts[source] = tier_counts.get(source, 0) + 1

    taxonomy_version = str(taxonomy["version"])

    # Write to AnnotationStore if annotations_out is provided
    if annotations_out:
        from pathlib import Path

        from agent_diagnostics.annotation_store import AnnotationStore
        from agent_diagnostics.signals import compute_trial_id

        rows: list[dict] = []
        for ann in annotations:
            trial_id = ann.get("trial_id", "")
            if not trial_id:
                tid, _ = compute_trial_id(
                    ann.get("task_id", ""),
                    ann.get("config_name", ""),
                    ann.get("started_at", ""),
                    ann.get("model", ""),
                )
                trial_id = tid
            annotated_at = ann.get("annotated_at", now)
            for cat in ann.get("categories", []):
                rows.append(
                    {
                        "trial_id": trial_id,
                        "category_name": cat.get("name", ""),
                        "confidence": cat.get("confidence", 0.0),
                        "evidence": cat.get("evidence", ""),
                        "annotator_type": "ensemble",
                        "annotator_identity": "ensemble:heuristic+classifier",
                        "taxonomy_version": taxonomy_version,
                        "annotated_at": annotated_at,
                    }
                )
        if rows:
            store = AnnotationStore(Path(annotations_out))
            store.upsert_annotations(rows, taxonomy_version=taxonomy_version)

    result = {
        "schema_version": "observatory-annotation-v1",
        "taxonomy_version": taxonomy_version,
        "generated_at": now,
        "annotator": {
            "type": "ensemble",
            "identity": (
                f"heuristic({len(HEURISTIC_ONLY)} structural, "
                f"{tier_counts.get('heuristic', 0)} assignments) + "
                f"classifier({len(model['classifiers'])} trained, "
                f"{tier_counts.get('classifier', 0)} assignments, "
                f"threshold={classifier_threshold}, min_f1={classifier_min_f1}"
                + (f", max_ece={classifier_max_ece}" if classifier_max_ece is not None else "")
                + ")"
            ),
        },
        "annotations": annotations,
    }

    # Strip trial_id from annotations in the returned dict for backwards compat
    # (the legacy format does not include trial_id at the annotation level)
    for ann in result["annotations"]:
        ann.pop("trial_id", None)

    return result
