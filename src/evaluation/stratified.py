"""Regime-stratified evaluation and statistical paired comparison suite with bootstrap uncertainty."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence intervals.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    point_est = float(stat_fn(data))

    if n < 5:
        return point_est, point_est, point_est

    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([stat_fn(data[idx]) for idx in indices])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, alpha * 100.0))
    upper = float(np.percentile(boot_stats, (1.0 - alpha) * 100.0))
    return point_est, lower, upper


def compute_pr_auc_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval for PR-AUC (Average Precision).
    Returns: (point_pr_auc, ci_lower, ci_upper)
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    n = len(y_t)
    point_pr_auc = float(average_precision_score(y_t, y_p))

    if n < 10 or np.sum(y_t == 1) == 0 or np.sum(y_t == 0) == 0:
        return point_pr_auc, point_pr_auc, point_pr_auc

    rng = np.random.RandomState(seed)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample_t = y_t[idx]
        sample_p = y_p[idx]
        if np.sum(sample_t == 1) > 0 and np.sum(sample_t == 0) > 0:
            boot_scores.append(average_precision_score(sample_t, sample_p))

    if not boot_scores:
        return point_pr_auc, point_pr_auc, point_pr_auc

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_scores, alpha * 100.0))
    upper = float(np.percentile(boot_scores, (1.0 - alpha) * 100.0))
    return round(point_pr_auc, 4), round(lower, 4), round(upper, 4)



def compute_paired_comparison(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Union[float, str]]:
    """Paired comparison between Model A and Model B on the exact same test samples.
    Computes delta MAE = MAE(A) - MAE(B), bootstrap 95% CI, and paired Wilcoxon signed-rank test.
    """
    err_a = np.abs(preds_a - targets)
    err_b = np.abs(preds_b - targets)
    diff = err_a - err_b  # Positive when Model B has lower error (B is better)

    mean_diff, ci_low, ci_high = compute_bootstrap_ci(
        diff, stat_fn=np.mean, n_bootstrap=n_bootstrap, ci=ci, seed=seed
    )

    # Paired Wilcoxon signed-rank test
    try:
        w_stat, p_val = stats.wilcoxon(err_a.flatten(), err_b.flatten())
        p_val = float(p_val)
    except Exception:
        p_val = 1.0

    stat_sig = (ci_low > 0 or ci_high < 0) and (p_val < 0.05)

    return {
        "mae_a": round(float(np.mean(err_a)), 3),
        "mae_b": round(float(np.mean(err_b)), 3),
        "delta_mae_a_minus_b": round(mean_diff, 3),
        "ci_lower": round(ci_low, 3),
        "ci_upper": round(ci_high, 3),
        "p_value": float(f"{p_val:.2e}"),
        "statistically_significant": stat_sig,
        "interpretation": (
            "Model B is significantly better" if stat_sig and mean_diff > 0 else (
                "Model A is significantly better" if stat_sig and mean_diff < 0 else "Difference not statistically distinguishable from noise"
            )
        )
    }


def evaluate_regime_stratified(
    df: pd.DataFrame,
    preds: np.ndarray,
    targets: np.ndarray,
    ri_probs: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """Computes performance stratified by:
      - Initial intensity category (TD, TS, Cat 1-2, Cat 3-5)
      - Dynamic tendency (Weakening, Stable, Strengthening)
      - RI event cases vs Non-RI cases
      - Ocean basin (WP, AL, EP, IO, SH, CP)
    """
    df = df.reset_index(drop=True)
    v_curr = df["vmax_curr"].values
    horizons = ["+6h", "+12h", "+24h"]

    results = {
        "overall": {},
        "by_horizon": {},
        "by_intensity_regime": {},
        "by_tendency": {},
        "by_ri_occurrence": {},
        "by_basin": {},
    }

    # 1. Overall & By Horizon
    for h_idx, h_name in enumerate(horizons):
        p = preds[:, h_idx]
        t = targets[:, h_idx]
        err = p - t
        abs_err = np.abs(err)

        mae, mae_l, mae_u = compute_bootstrap_ci(abs_err, stat_fn=np.mean)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        results["by_horizon"][h_name] = {
            "mae": round(mae, 2),
            "mae_95ci": [round(mae_l, 2), round(mae_u, 2)],
            "rmse": round(rmse, 2),
            "bias": round(bias, 2),
            "n_samples": len(p),
        }

    # 2. Stratification by Initial Intensity
    regime_bins = {
        "TD (<34 kt)": v_curr < 34,
        "TS (34-63 kt)": (v_curr >= 34) & (v_curr < 64),
        "Cat 1-2 (64-95 kt)": (v_curr >= 64) & (v_curr < 96),
        "Cat 3-5 (>=96 kt)": v_curr >= 96,
    }

    for r_name, mask in regime_bins.items():
        if np.sum(mask) == 0:
            continue
        sub_preds = preds[mask]
        sub_targets = targets[mask]
        mae_24 = float(np.mean(np.abs(sub_preds[:, 2] - sub_targets[:, 2])))
        rmse_24 = float(np.sqrt(np.mean((sub_preds[:, 2] - sub_targets[:, 2]) ** 2)))

        results["by_intensity_regime"][r_name] = {
            "n_samples": int(np.sum(mask)),
            "pct_of_test": round(100.0 * np.sum(mask) / len(df), 1),
            "mae_plus_24h": round(mae_24, 2),
            "rmse_plus_24h": round(rmse_24, 2),
        }

    # 3. Stratification by 24h Tendency
    actual_delta_24 = targets[:, 2] - v_curr
    tendency_bins = {
        "Weakening (Delta<=-10kt)": actual_delta_24 <= -10.0,
        "Stable (-10<Delta<10kt)": (actual_delta_24 > -10.0) & (actual_delta_24 < 10.0),
        "Strengthening (Delta>=10kt)": actual_delta_24 >= 10.0,
    }

    for t_name, mask in tendency_bins.items():
        if np.sum(mask) == 0:
            continue
        sub_preds = preds[mask]
        sub_targets = targets[mask]
        results["by_tendency"][t_name] = {
            "n_samples": int(np.sum(mask)),
            "mae_plus_24h": round(float(np.mean(np.abs(sub_preds[:, 2] - sub_targets[:, 2]))), 2),
            "bias_plus_24h": round(float(np.mean(sub_preds[:, 2] - sub_targets[:, 2])), 2),
        }

    # 4. RI Occurrence vs Non-RI
    ri_mask = actual_delta_24 >= 30.0
    results["by_ri_occurrence"] = {
        "RI Events (Delta>=30kt)": {
            "n_samples": int(np.sum(ri_mask)),
            "mae_plus_24h": round(float(np.mean(np.abs(preds[ri_mask, 2] - targets[ri_mask, 2]))), 2) if np.sum(ri_mask) > 0 else None,
            "bias_plus_24h": round(float(np.mean(preds[ri_mask, 2] - targets[ri_mask, 2])), 2) if np.sum(ri_mask) > 0 else None,
        },
        "Non-RI Events (Delta<30kt)": {
            "n_samples": int(np.sum(~ri_mask)),
            "mae_plus_24h": round(float(np.mean(np.abs(preds[~ri_mask, 2] - targets[~ri_mask, 2]))), 2),
            "bias_plus_24h": round(float(np.mean(preds[~ri_mask, 2] - targets[~ri_mask, 2])), 2),
        }
    }

    # 5. Ocean Basin
    if "cyclone_id" in df.columns:
        df["basin"] = df["cyclone_id"].apply(lambda c: str(c)[-1] if len(str(c)) >= 7 else "UNK")
        basin_map = {"W": "West Pacific (WP)", "L": "Atlantic (AL)", "E": "East Pacific (EP)", "I": "Indian Ocean (IO)", "S": "Southern Hem (SH)", "C": "Central Pacific (CP)"}
        for b_char, b_name in basin_map.items():
            b_mask = (df["basin"] == b_char).values
            if np.sum(b_mask) >= 10:
                sub_p = preds[b_mask]
                sub_t = targets[b_mask]
                results["by_basin"][b_name] = {
                    "n_samples": int(np.sum(b_mask)),
                    "mae_plus_24h": round(float(np.mean(np.abs(sub_p[:, 2] - sub_t[:, 2]))), 2),
                    "rmse_plus_24h": round(float(np.sqrt(np.mean((sub_p[:, 2] - sub_t[:, 2]) ** 2))), 2),
                }

    return results
