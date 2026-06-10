"""Classifier subcommands: blend, train, predict, ensemble."""

import json
import logging
import sys
from pathlib import Path

from agent_diagnostics.cli._helpers import (
    _annotations_to_narrow_rows,
    _write_to_annotation_store,
)

logger = logging.getLogger(__name__)


def cmd_blend(args):
    """Blend heuristic + LLM annotations into a unified training set."""
    from agent_diagnostics.blend_labels import blend

    for label, value in (("heuristic", args.heuristic), ("llm", args.llm)):
        if not Path(value).is_file():
            logger.error("%s annotations not found: %s", label, value)
            sys.exit(1)
    if args.calibration and not Path(args.calibration).is_file():
        logger.error("calibration file not found: %s", args.calibration)
        sys.exit(1)

    result = blend(
        heuristic_file=args.heuristic,
        llm_file=args.llm,
        calibration_file=args.calibration,
        heuristic_trust_threshold=args.trust_threshold,
        max_heuristic_samples=args.max_heuristic_samples,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    meta = result["blend_metadata"]
    if meta["heuristic_only_dropped"]:
        logger.warning(
            "heuristic-only sample cap (%d) dropped %d trials — raise "
            "--max-heuristic-samples to include them",
            args.max_heuristic_samples,
            meta["heuristic_only_dropped"],
        )
    logger.info(
        "Blended %d annotations (llm=%d, heuristic-only=%d, skipped errored llm=%d). Output: %s",
        meta["total_blended"],
        meta["llm_trials"],
        meta["heuristic_only_trials"],
        meta["skipped_errored_llm_trials"],
        output,
    )


def cmd_train(args):
    """Train per-category classifiers from LLM-labeled data."""
    from agent_diagnostics.classifier import (
        evaluate,
        format_eval_markdown,
        save_model,
        train,
    )

    model = train(
        llm_file=args.labels,
        signals_file=args.signals,
        min_positive=args.min_positive,
        lr=args.lr,
        epochs=args.epochs,
        cv_folds=getattr(args, "cv_folds", 5),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output)

    n_clf = len(model["classifiers"])
    n_skip = len(model["skipped_categories"])
    logger.info(
        "Trained %d classifiers on %d samples (%d categories skipped, < %d positive)",
        n_clf,
        model["training_samples"],
        n_skip,
        model["min_positive"],
    )
    for cat, clf in sorted(model["classifiers"].items()):
        eval_f1 = clf.get("eval_f1")
        cv_ece = clf.get("cv_ece")
        cv_suffix = (
            f", cv_f1={eval_f1:.2f}, cv_ece={cv_ece:.2f}"
            if eval_f1 is not None and cv_ece is not None
            else ", cv=insufficient_data"
        )
        logger.info(
            "  %s: %d/%d positive, train_acc=%.2f%s",
            cat,
            clf["positive_count"],
            clf["total_count"],
            clf["train_accuracy"],
            cv_suffix,
        )

    # If --eval is provided, evaluate on the same data (quick sanity check)
    if args.eval:
        eval_results = evaluate(model, args.labels, args.signals)
        logger.info("\n%s", format_eval_markdown(eval_results, model))

    logger.info("Model saved to %s", output)


def cmd_predict(args):
    """Predict categories for all trials using a trained classifier."""
    from agent_diagnostics.classifier import load_model, predict_all
    from agent_diagnostics.signals import load_signals

    model = load_model(args.model)
    signals_list = load_signals(args.signals)

    result = predict_all(signals_list, model, threshold=args.threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    n = len(result["annotations"])
    total_cats = sum(len(a["categories"]) for a in result["annotations"])
    logger.info(
        "Predicted %d category assignments across %d trials (threshold=%s). Output: %s",
        total_cats,
        n,
        args.threshold,
        output,
    )

    # Write to AnnotationStore if --annotations-out is provided
    annotations_out = getattr(args, "annotations_out", None)
    if annotations_out:
        taxonomy_version = str(result.get("taxonomy_version", ""))
        # Enrich with trial_id from the signals_list
        enriched = []
        for sig, ann in zip(signals_list, result["annotations"]):
            enriched.append({**ann, "trial_id": sig.get("trial_id", "")})
        rows = _annotations_to_narrow_rows(
            enriched,
            annotator_type="classifier",
            annotator_identity="classifier:trained-model",
            taxonomy_version=taxonomy_version,
        )
        _write_to_annotation_store(rows, annotations_out, taxonomy_version)


def cmd_ensemble(args):
    """Run two-tier ensemble annotation (heuristic + classifier) on full corpus."""
    from agent_diagnostics.classifier import load_model
    from agent_diagnostics.ensemble import ensemble_all
    from agent_diagnostics.signals import load_signals

    model = load_model(args.model)
    signals_list = load_signals(args.signals)

    annotations_out = getattr(args, "annotations_out", None)
    result = ensemble_all(
        signals_list,
        model,
        classifier_threshold=args.threshold,
        classifier_min_f1=args.min_f1,
        classifier_max_ece=getattr(args, "max_ece", None),
        annotations_out=annotations_out,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    n = len(result["annotations"])
    total_cats = sum(len(a["categories"]) for a in result["annotations"])
    tiers = result.get("tier_counts", {})
    logger.info(
        "Ensemble: %d assignments across %d trials (heuristic=%d, classifier=%d). Output: %s",
        total_cats,
        n,
        tiers.get("heuristic", 0),
        tiers.get("classifier", 0),
        output,
    )
