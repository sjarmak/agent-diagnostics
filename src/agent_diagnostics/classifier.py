"""Lightweight multi-label classifier for taxonomy category prediction.

Trains one binary classifier per taxonomy category on the numeric signal
features extracted by ``signals.py``. Uses LLM-generated annotations as
training labels.

Design:
- Features: 21 numeric/boolean signals per trial (no text features needed).
- Model: per-category logistic regression with class-weight balancing.
- Validation: stratified k-fold cross-validation per category; the pooled
  out-of-fold predictions yield ``eval_f1`` / ``eval_precision`` /
  ``eval_recall`` / ``cv_ece`` stored in the model artifact. These are the
  numbers the ensemble trust gate consumes — never training-set metrics.
- Output: predicted categories with calibrated probabilities.
- Storage: single JSON file containing per-category model weights.
- Inference: pure Python, no external dependencies.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

# --- Feature extraction ---

# Ordered feature list — must be stable across train and predict.
# Keys mapped from TrialSignals where possible:
#   CSB has_exception      → exception_crashed
#   CSB wall_clock_seconds → duration_seconds
#   CSB trajectory_steps   → trajectory_length
FEATURE_NAMES: list[str] = [
    "reward",
    "passed",
    "exception_crashed",
    "tool_calls_total",
    "trajectory_length",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "duration_seconds",
    "search_calls_keyword",
    "search_calls_nls",
    "search_calls_deepsearch",
    "mcp_ratio",
    "ttfr",
    "query_churn_count",
    "edit_verify_cycles",
    "repeated_tool_failures",
    "has_code_nav_tools",
    "has_semantic_search",
    "has_git_tools",
    "has_trajectory",
]


def _to_float(val: Any) -> float:
    """Convert a signal value to a float for the feature vector."""
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def signals_to_features(signals: dict) -> list[float]:
    """Convert a signal dict to a fixed-length numeric feature vector."""
    return [_to_float(signals.get(name)) for name in FEATURE_NAMES]


# --- Logistic regression (pure Python, no sklearn dependency) ---


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _scale(features_matrix: list[list[float]]) -> tuple[list[float], list[float]]:
    """Compute per-feature mean and std for standardization."""
    n = len(features_matrix)
    if n == 0:
        d = len(FEATURE_NAMES)
        return [0.0] * d, [1.0] * d
    d = len(features_matrix[0])
    means = [0.0] * d
    for row in features_matrix:
        for j in range(d):
            means[j] += row[j]
    means = [m / n for m in means]

    stds = [0.0] * d
    for row in features_matrix:
        for j in range(d):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [max(math.sqrt(s / n), 1e-8) for s in stds]
    return means, stds


def _standardize(row: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(row[j] - means[j]) / stds[j] for j in range(len(row))]


def _train_binary_lr(
    X: list[list[float]],
    y: list[int],
    means: list[float],
    stds: list[float],
    lr: float = 0.1,
    epochs: int = 200,
    l2: float = 0.01,
) -> tuple[list[float], float]:
    """Train a single binary logistic regression via gradient descent.

    Returns (weights, bias).
    """
    d = len(X[0]) if X else len(FEATURE_NAMES)
    n = len(X)
    w = [0.0] * d
    b = 0.0

    # Class weight: upweight the minority class
    pos = sum(y)
    neg = n - pos
    if pos == 0 or neg == 0:
        w_pos = 1.0
        w_neg = 1.0
    else:
        w_pos = n / (2.0 * pos)
        w_neg = n / (2.0 * neg)

    for _ in range(epochs):
        grad_w = [l2 * wi for wi in w]
        grad_b = 0.0

        for i in range(n):
            xi = _standardize(X[i], means, stds)
            z = _dot(w, xi) + b
            p = _sigmoid(z)
            weight = w_pos if y[i] == 1 else w_neg
            err = weight * (p - y[i])
            for j in range(d):
                grad_w[j] += err * xi[j] / n
            grad_b += err / n

        for j in range(d):
            w[j] -= lr * grad_w[j]
        b -= lr * grad_b

    return w, b


def _predict_proba(
    row: list[float],
    weights: list[float],
    bias: float,
    means: list[float],
    stds: list[float],
) -> float:
    xs = _standardize(row, means, stds)
    return _sigmoid(_dot(weights, xs) + bias)


# --- Cross-validation ---

# Fixed seed so fold assignment (and therefore eval_f1 / cv_ece in the model
# artifact) is reproducible across train runs on identical inputs.
_CV_SEED = 42

_ECE_BINS = 10


def _stratified_folds(labels: list[int], k: int) -> list[list[int]]:
    """Assign sample indices to *k* folds, stratified by class.

    Positives and negatives are shuffled (seeded) and dealt round-robin so
    every fold carries approximately the same class balance.
    """
    rng = random.Random(_CV_SEED)
    pos_idx = [i for i, label in enumerate(labels) if label == 1]
    neg_idx = [i for i, label in enumerate(labels) if label == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds: list[list[int]] = [[] for _ in range(k)]
    for j, i in enumerate(pos_idx):
        folds[j % k].append(i)
    for j, i in enumerate(neg_idx):
        folds[j % k].append(i)
    return folds


def _ece(probs: list[float], labels: list[int], bins: int = _ECE_BINS) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    n = len(probs)
    if n == 0:
        return 0.0
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins
    bin_count = [0] * bins
    for p, label in zip(probs, labels):
        b = min(int(p * bins), bins - 1)
        bin_conf[b] += p
        bin_acc[b] += label
        bin_count[b] += 1
    ece = 0.0
    for b in range(bins):
        if bin_count[b] == 0:
            continue
        gap = abs(bin_conf[b] / bin_count[b] - bin_acc[b] / bin_count[b])
        ece += gap * bin_count[b] / n
    return ece


def _cross_validate(
    X: list[list[float]],
    labels: list[int],
    cv_folds: int,
    lr: float,
    epochs: int,
    l2: float,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Stratified k-fold CV for one category; metrics on pooled out-of-fold predictions.

    Scaling parameters are recomputed per fold from the training portion only,
    so no information from the held-out fold leaks into its predictions.

    Returns a dict with ``eval_f1`` / ``eval_precision`` / ``eval_recall`` /
    ``cv_ece`` / ``cv_folds``, or ``{"cv_status": "insufficient_data"}`` when
    the category lacks two examples of either class.
    """
    pos = sum(labels)
    neg = len(labels) - pos
    # Each training portion needs at least one example of each class, which
    # stratified folding guarantees once both classes have >= 2 members.
    if pos < 2 or neg < 2:
        return {"cv_status": "insufficient_data"}

    k = max(2, min(cv_folds, pos, neg))
    folds = _stratified_folds(labels, k)

    oof_probs: list[float] = []
    oof_labels: list[int] = []
    for fold in folds:
        held_out = set(fold)
        train_X = [X[i] for i in range(len(X)) if i not in held_out]
        train_y = [labels[i] for i in range(len(X)) if i not in held_out]
        means, stds = _scale(train_X)
        w, b = _train_binary_lr(train_X, train_y, means, stds, lr=lr, epochs=epochs, l2=l2)
        for i in fold:
            oof_probs.append(_predict_proba(X[i], w, b, means, stds))
            oof_labels.append(labels[i])

    tp = fp = fn = 0
    for p, label in zip(oof_probs, oof_labels):
        pred = 1 if p >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "eval_f1": round(f1, 4),
        "eval_precision": round(precision, 4),
        "eval_recall": round(recall, 4),
        "cv_ece": round(_ece(oof_probs, oof_labels), 4),
        "cv_folds": k,
    }


# --- Training pipeline ---


def _align_labels(
    llm_annotations: list[dict],
    all_signals: list[dict],
) -> tuple[list[list[float]], dict[str, list[int]]]:
    """Align LLM labels with signal features by trial_path.

    Returns:
        X: feature matrix (one row per matched trial)
        y_per_cat: dict mapping category name to binary label vector
    """
    # Build signal lookup by trial_path
    sig_by_path: dict[str, dict] = {}
    for s in all_signals:
        p = s.get("trial_path")
        if p:
            sig_by_path[p] = s

    X: list[list[float]] = []
    matched_cats: list[set[str]] = []
    all_cat_names: set[str] = set()

    for ann in llm_annotations:
        path = ann.get("trial_path", "")
        if path not in sig_by_path:
            continue
        sig = sig_by_path[path]
        X.append(signals_to_features(sig))
        cats = {c["name"] for c in ann.get("categories", [])}
        matched_cats.append(cats)
        all_cat_names |= cats

    # Build per-category label vectors
    y_per_cat: dict[str, list[int]] = {}
    for cat in sorted(all_cat_names):
        y_per_cat[cat] = [1 if cat in cats else 0 for cats in matched_cats]

    return X, y_per_cat


def train(
    llm_file: str | Path,
    signals_file: str | Path,
    min_positive: int = 3,
    lr: float = 0.1,
    epochs: int = 300,
    l2: float = 0.01,
    cv_folds: int = 5,
) -> dict:
    """Train per-category logistic regression classifiers.

    Parameters
    ----------
    llm_file : path
        LLM annotation JSON file (training labels).
    signals_file : path
        Full signal corpus JSON (features for all trials).
    min_positive : int
        Minimum positive examples to train a category classifier.
        Categories with fewer are skipped (too little signal).
    lr, epochs, l2 : float
        Learning rate, training epochs, L2 regularization.
    cv_folds : int
        Folds for the held-out cross-validation that produces ``eval_f1``
        and ``cv_ece`` per category (clamped per category to the minority
        class count, floor 2).

    Returns
    -------
    dict
        Model artifact with weights, biases, scaling params, held-out CV
        metrics per category, and metadata.
    """
    from agent_diagnostics.signals import load_annotations, load_signals

    llm_data = load_annotations(llm_file)
    all_signals = load_signals(signals_file)

    llm_anns = llm_data.get("annotations", llm_data)
    if not isinstance(llm_anns, list):
        raise ValueError("Expected annotations list in LLM file")

    X, y_per_cat = _align_labels(llm_anns, all_signals)
    if not X:
        raise ValueError("No trials matched between LLM labels and signals")

    # Skip categories marked derived_from_signal in the v3 taxonomy.
    # These are reward-band categories whose labels are deterministic
    # functions of the reward signal and should not be learned.
    from agent_diagnostics.taxonomy import (
        _extract_categories,
        _package_data_path,
        load_taxonomy,
    )

    _v3_tax = load_taxonomy(_package_data_path("taxonomy_v3.yaml"))
    _derived_cats: frozenset[str] = frozenset(
        cat["name"]
        for cat in _extract_categories(_v3_tax)
        if cat.get("derived_from_signal", False)
    )

    means, stds = _scale(X)

    # Train per-category classifier
    classifiers: dict[str, dict] = {}
    skipped: list[str] = []

    for cat, labels in sorted(y_per_cat.items()):
        if cat in _derived_cats:
            skipped.append(cat)
            continue
        pos = sum(labels)
        if pos < min_positive:
            skipped.append(cat)
            continue

        w, b = _train_binary_lr(X, labels, means, stds, lr=lr, epochs=epochs, l2=l2)

        # Training accuracy
        correct = 0
        for i in range(len(X)):
            p = _predict_proba(X[i], w, b, means, stds)
            pred = 1 if p >= 0.5 else 0
            if pred == labels[i]:
                correct += 1

        # Held-out validation: the ensemble trust gate reads eval_f1/cv_ece,
        # so they must come from out-of-fold predictions, not the train set.
        cv_metrics = _cross_validate(X, labels, cv_folds, lr=lr, epochs=epochs, l2=l2)

        classifiers[cat] = {
            "weights": w,
            "bias": b,
            "positive_count": pos,
            "total_count": len(labels),
            "train_accuracy": round(correct / len(labels), 4),
            **cv_metrics,
        }

    model = {
        "schema_version": "observatory-classifier-v1",
        "feature_names": FEATURE_NAMES,
        "means": means,
        "stds": stds,
        "classifiers": classifiers,
        "skipped_categories": skipped,
        "training_samples": len(X),
        "min_positive": min_positive,
        "hyperparams": {"lr": lr, "epochs": epochs, "l2": l2, "cv_folds": cv_folds},
    }

    return model


def save_model(model: dict, path: str | Path) -> None:
    """Save a trained model to JSON."""
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def load_model(path: str | Path) -> dict:
    """Load a trained model from JSON."""
    with open(path) as f:
        return json.load(f)


# --- Inference ---


def predict_trial(
    signals: dict,
    model: dict,
    threshold: float = 0.5,
) -> list[dict]:
    """Predict categories for a single trial.

    Returns list of {name, confidence, evidence} dicts.
    """
    features = signals_to_features(signals)
    means = model["means"]
    stds = model["stds"]
    results: list[dict] = []

    for cat, clf in model["classifiers"].items():
        prob = _predict_proba(features, clf["weights"], clf["bias"], means, stds)
        if prob >= threshold:
            results.append(
                {
                    "name": cat,
                    "confidence": round(prob, 4),
                    "evidence": f"classifier prob={prob:.3f} (threshold={threshold})",
                }
            )

    return results


def predict_all(
    signals_list: list[dict],
    model: dict,
    threshold: float = 0.5,
) -> dict:
    """Predict categories for all trials, return annotation document."""
    from datetime import datetime, timezone

    from agent_diagnostics.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    now = datetime.now(timezone.utc).isoformat()

    annotations: list[dict] = []
    for sig in signals_list:
        cats = predict_trial(sig, model, threshold)
        if not cats:
            continue
        reward = sig.get("reward")
        annotations.append(
            {
                "task_id": sig.get("task_id") or "unknown",
                "trial_path": sig.get("trial_path") or "",
                "config_name": sig.get("config_name"),
                "benchmark": sig.get("benchmark"),
                "model": sig.get("model"),
                "reward": float(reward) if reward is not None else 0.0,
                "passed": bool(sig.get("passed")),
                "categories": cats,
                "annotated_at": now,
            }
        )

    return {
        "schema_version": "observatory-annotation-v1",
        "taxonomy_version": str(taxonomy["version"]),
        "generated_at": now,
        "annotator": {
            "type": "classifier",
            "identity": (
                f"agent_diagnostics.classifier v1 ({len(model['classifiers'])} "
                f"categories, trained on {model['training_samples']} samples)"
            ),
        },
        "annotations": annotations,
    }


# --- Evaluation ---


def evaluate(
    model: dict,
    llm_file: str | Path,
    signals_file: str | Path,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model against held-out LLM labels.

    Returns per-category precision, recall, F1 and confusion counts.
    """
    from agent_diagnostics.signals import load_annotations, load_signals

    llm_data = load_annotations(llm_file)
    all_signals = load_signals(signals_file)

    llm_anns = llm_data.get("annotations", llm_data)
    if not isinstance(llm_anns, list):
        raise ValueError("Expected annotations list")

    X, y_per_cat = _align_labels(llm_anns, all_signals)
    means = model["means"]
    stds = model["stds"]

    results: dict[str, dict] = {}
    for cat, labels in y_per_cat.items():
        clf = model["classifiers"].get(cat)
        if clf is None:
            results[cat] = {
                "status": "no_classifier",
                "positive_in_eval": sum(labels),
            }
            continue

        tp = fp = fn = tn = 0
        for i in range(len(X)):
            prob = _predict_proba(X[i], clf["weights"], clf["bias"], means, stds)
            pred = 1 if prob >= threshold else 0
            actual = labels[i]
            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 0 and actual == 1:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[cat] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return results


# --- CLI helpers ---


def format_eval_markdown(eval_results: dict, model: dict) -> str:
    """Format evaluation results as Markdown."""
    lines = ["## Classifier Evaluation", ""]
    lines.append(f"Training samples: {model['training_samples']}")
    lines.append(f"Categories trained: {len(model['classifiers'])}")
    lines.append(
        f"Categories skipped (< {model['min_positive']} positive): "
        f"{', '.join(model['skipped_categories']) or 'none'}"
    )
    lines.append("")
    lines.append(
        "| Category | TP | FP | FN | Precision | Recall | F1 | Train Acc | CV F1 | CV ECE |"
    )
    lines.append(
        "|----------|---:|---:|---:|----------:|-------:|---:|----------:|------:|-------:|"
    )

    for cat in sorted(eval_results, key=lambda c: -eval_results[c].get("f1", -1)):
        r = eval_results[cat]
        if r.get("status") == "no_classifier":
            lines.append(
                f"| {cat} | — | — | {r['positive_in_eval']} | — | — | — | no clf | — | — |"
            )
            continue
        clf = model["classifiers"].get(cat, {})
        eval_f1 = clf.get("eval_f1")
        cv_ece = clf.get("cv_ece")
        cv_f1_cell = f"{eval_f1:.2f}" if eval_f1 is not None else "—"
        cv_ece_cell = f"{cv_ece:.2f}" if cv_ece is not None else "—"
        lines.append(
            f"| {cat} | {r['tp']} | {r['fp']} | {r['fn']} "
            f"| {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} "
            f"| {clf.get('train_accuracy', 0):.2f} | {cv_f1_cell} | {cv_ece_cell} |"
        )

    return "\n".join(lines)
