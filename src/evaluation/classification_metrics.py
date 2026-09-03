"""Comprehensive classification and calibration metrics for Tropical Cyclone Evolution & Rapid Intensification."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) and reliability diagram bins.
    Returns:
        ece: scalar ECE value
        bin_accuracies: fraction of positive samples per bin
        bin_confidences: mean predicted probability per bin
        bin_counts: number of samples per bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        mask = (y_prob >= b_low) & (y_prob < b_high if i < n_bins - 1 else y_prob <= b_high)
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            ece += (count / n) * abs(acc - conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((b_low + b_high) / 2.0)

    return float(ece), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float, float, float]:
    """Find decision threshold maximizing F1 on precision-recall curve.
    Returns:
        best_threshold, best_f1, precision_at_best, recall_at_best
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0.0,
    )
    # Exclude the last precision/recall point which has no threshold
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return best_thresh, float(f1s[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def compute_ri_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, any]:
    """Compute comprehensive Rapid Intensification evaluation metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Threshold predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Basic discrimination
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = float(n_pos / max(len(y_true), 1))

    # ROC-AUC & PR-AUC
    if n_pos > 0 and n_neg > 0:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        pr_auc = prevalence

    # Operating point metrics
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Optimal F1 threshold
    opt_thresh, opt_f1, opt_p, opt_r = find_optimal_threshold(y_true, y_prob)

    # Calibration
    brier = float(brier_score_loss(y_true, y_prob))
    ece, bin_accs, bin_confs, bin_counts = compute_expected_calibration_error(y_true, y_prob)

    return {
        "prevalence": round(prevalence, 4),
        "n_ri_events": n_pos,
        "n_total": len(y_true),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        f"precision_at_{threshold:.2f}": round(float(p), 4),
        f"recall_at_{threshold:.2f}": round(float(r), 4),
        f"f1_at_{threshold:.2f}": round(float(f1), 4),
        "optimal_threshold": round(opt_thresh, 4),
        "optimal_f1": round(opt_f1, 4),
        "optimal_precision": round(opt_p, 4),
        "optimal_recall": round(opt_r, 4),
        "confusion_matrix": cm.tolist(),
        "calibration_bins": {
            "accuracies": bin_accs.tolist(),
            "confidences": bin_confs.tolist(),
            "counts": bin_counts.tolist(),
        },
    }


def compute_trend_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, any]:
    """Compute comprehensive 3-class Intensity Trend evaluation metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if class_names is None:
        class_names = {0: "WEAKENING", 1: "STABLE", 2: "INTENSIFYING"}

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec, rec, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    per_class = {}
    for idx, c_name in class_names.items():
        per_class[c_name] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1s[idx]), 4),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = (cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)).round(4)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def compute_stratified_evaluation(
    df: pd.DataFrame,
    trend_preds: np.ndarray,
    ri_probs: np.ndarray,
    ri_threshold: float = 0.5,
) -> Dict[str, any]:
    """Stratify evaluation across:
    1. Weakening / Stable / Intensifying subsets
    2. RI vs Non-RI subsets
    3. Saffir-Simpson intensity categories:
       - TD/TS (< 64 kt)
       - Cat 1-2 (64-95 kt)
       - Cat 3+ (>= 96 kt)
    """
    results = {}
    v_curr = df["vmax_curr"].values
    d24 = df["vmax_plus_24h"].values - v_curr

    # Ground truth
    y_trend = np.ones(len(d24), dtype=int)
    y_trend[d24 <= -10.0] = 0
    y_trend[d24 >= 10.0] = 2

    y_ri = (d24 >= 30.0).astype(int)

    # 1. Stratified by Event Type (Weakening, Stable, Intensifying)
    event_strata = {
        "weakening_events": y_trend == 0,
        "stable_events": y_trend == 1,
        "intensifying_events": y_trend == 2,
    }
    for e_name, mask in event_strata.items():
        n_sub = int(np.sum(mask))
        if n_sub > 0:
            trend_acc = float(accuracy_score(y_trend[mask], trend_preds[mask]))
            ri_mean_prob = float(np.mean(ri_probs[mask]))
            results[e_name] = {
                "n_samples": n_sub,
                "trend_accuracy": round(trend_acc, 4),
                "mean_ri_prob": round(ri_mean_prob, 4),
            }

    # 2. Stratified on RI Events vs Non-RI Events
    for is_ri, r_name in [(1, "ri_events"), (0, "non_ri_events")]:
        mask = y_ri == is_ri
        n_sub = int(np.sum(mask))
        if n_sub > 0:
            trend_acc = float(accuracy_score(y_trend[mask], trend_preds[mask]))
            mean_prob = float(np.mean(ri_probs[mask]))
            pred_ri_flags = (ri_probs[mask] >= ri_threshold).astype(int)
            ri_detected_fraction = float(np.mean(pred_ri_flags))
            results[r_name] = {
                "n_samples": n_sub,
                "trend_accuracy": round(trend_acc, 4),
                "mean_ri_prob": round(mean_prob, 4),
                "ri_detection_rate": round(ri_detected_fraction, 4),
            }

    # 3. Stratified by Saffir-Simpson Current Storm Intensity
    intensity_bins = {
        "Tropical Depression / Storm (< 64 kt)": v_curr < 64.0,
        "Category 1-2 Hurricane/Typhoon (64-95 kt)": (v_curr >= 64.0) & (v_curr <= 95.0),
        "Category 3+ Major Hurricane/Typhoon (>= 96 kt)": v_curr >= 96.0,
    }

    results["intensity_regimes"] = {}
    for regime_name, mask in intensity_bins.items():
        n_sub = int(np.sum(mask))
        if n_sub == 0:
            continue

        sub_trend_true = y_trend[mask]
        sub_trend_pred = trend_preds[mask]
        sub_ri_true = y_ri[mask]
        sub_ri_prob = ri_probs[mask]

        t_res = compute_trend_metrics(sub_trend_true, sub_trend_pred)
        r_res = compute_ri_metrics(sub_ri_true, sub_ri_prob, threshold=ri_threshold)

        results["intensity_regimes"][regime_name] = {
            "n_samples": n_sub,
            "trend_accuracy": t_res["accuracy"],
            "trend_macro_f1": t_res["macro_f1"],
            "ri_prevalence": r_res["prevalence"],
            "ri_pr_auc": r_res["pr_auc"],
            "ri_roc_auc": r_res["roc_auc"],
            "ri_f1": r_res[f"f1_at_{ri_threshold:.2f}"],
            "ri_recall": r_res[f"recall_at_{ri_threshold:.2f}"],
            "ri_precision": r_res[f"precision_at_{ri_threshold:.2f}"],
        }

    return results
