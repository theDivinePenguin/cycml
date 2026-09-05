"""Physical Sanity Checks and Trajectory Coherence Diagnostic Suite.

STRICT PRINCIPLE: This system NEVER silently modifies, clips, overwrites, or repairs predictions.
All thresholds are diagnostic flags designed to identify model failures, trajectory anomalies,
and edge-case violations.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class TrajectoryEvaluator:
    """Comprehensive meteorological trajectory coherence and error evaluator.

    Evaluates:
      - Pointwise metrics: MAE, RMSE, Bias per horizon [+6h, +12h, +24h]
      - Delta-V metrics: MAE, RMSE, Bias on predicted intensity changes
      - Trajectory Roughness: First & second-difference errors, mean absolute curvature
      - False Dips & Peaks: Implausible reversals (e.g., 65 kt -> 43 kt -> 70 kt)
      - Jump Diagnostics: Max absolute 6h forecast changes, fraction exceeding configurable thresholds
      - Directional Accuracy: Concordance of predicted vs true intensification/steady/weakening
      - Horizon Consistency: Inversion rates and monotonic alignment
    """

    def __init__(
        self,
        step_thresholds_kt: Tuple[float, ...] = (20.0, 30.0, 45.0, 60.0),
        dip_dip_tolerance_kt: float = 5.0,
        steady_deadband_kt: float = 2.5,
    ):
        self.step_thresholds_kt = step_thresholds_kt
        self.dip_dip_tolerance_kt = dip_dip_tolerance_kt
        self.steady_deadband_kt = steady_deadband_kt

    def evaluate_trajectories(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        v_curr: np.ndarray,
    ) -> Dict[str, Union[float, int, Dict]]:
        """
        Args:
            predictions: (N, 3) predicted intensity at [+6h, +12h, +24h]
            targets: (N, 3) ground truth intensity at [+6h, +12h, +24h]
            v_curr: (N,) or (N, 1) current observed intensity at time t
        """
        preds = np.asarray(predictions, dtype=float)
        targs = np.asarray(targets, dtype=float)
        v0 = np.asarray(v_curr, dtype=float).reshape(-1, 1)
        n = len(preds)
        if n == 0:
            return {"total_samples": 0}

        # Full 4-step trajectory: [t0, +6h, +12h, +24h]
        pred_traj = np.hstack([v0, preds])  # (N, 4)
        true_traj = np.hstack([v0, targs])  # (N, 4)

        # 1. Pointwise intensity errors
        err = preds - targs
        mae_per_horizon = np.mean(np.abs(err), axis=0).tolist()
        rmse_per_horizon = np.sqrt(np.mean(err ** 2, axis=0)).tolist()
        bias_per_horizon = np.mean(err, axis=0).tolist()

        overall_mae = float(np.mean(np.abs(err)))
        overall_rmse = float(np.sqrt(np.mean(err ** 2)))
        overall_bias = float(np.mean(err))

        # 2. Delta-V errors (relative to v0)
        pred_delta = preds - v0
        true_delta = targs - v0
        delta_err = pred_delta - true_delta
        delta_mae = np.mean(np.abs(delta_err), axis=0).tolist()
        delta_rmse = np.sqrt(np.mean(delta_err ** 2, axis=0)).tolist()
        delta_bias = np.mean(delta_err, axis=0).tolist()

        # 3. First Differences: step 0->6h, 6->12h, and 12->24h (normalized to 6h rate)
        pred_d1_0_6 = pred_traj[:, 1] - pred_traj[:, 0]
        pred_d1_6_12 = pred_traj[:, 2] - pred_traj[:, 1]
        pred_d1_12_24_norm = (pred_traj[:, 3] - pred_traj[:, 2]) / 2.0  # 6h equivalent rate

        true_d1_0_6 = true_traj[:, 1] - true_traj[:, 0]
        true_d1_6_12 = true_traj[:, 2] - true_traj[:, 1]
        true_d1_12_24_norm = (true_traj[:, 3] - true_traj[:, 2]) / 2.0

        d1_err_0_6 = float(np.mean(np.abs(pred_d1_0_6 - true_d1_0_6)))
        d1_err_6_12 = float(np.mean(np.abs(pred_d1_6_12 - true_d1_6_12)))
        d1_err_12_24 = float(np.mean(np.abs(pred_d1_12_24_norm - true_d1_12_24_norm)))
        mean_first_diff_error = float(np.mean([d1_err_0_6, d1_err_6_12, d1_err_12_24]))

        # 4. Second Differences (Curvature / Roughness)
        # Curvature at 6h step: d1_6_12 - d1_0_6 = V(12) - 2*V(6) + V(0)
        pred_curv_6h = pred_d1_6_12 - pred_d1_0_6
        true_curv_6h = true_d1_6_12 - true_d1_0_6

        pred_roughness_6h = float(np.mean(np.abs(pred_curv_6h)))
        true_roughness_6h = float(np.mean(np.abs(true_curv_6h)))
        second_diff_error = float(np.mean(np.abs(pred_curv_6h - true_curv_6h)))
        roughness_ratio = round(pred_roughness_6h / max(true_roughness_6h, 1e-6), 3)

        # 5. False Dips and False Peaks Detection
        # False Dip: predicted dips at 6h then rises at 12h, while true trajectory did NOT dip
        tol = self.dip_dip_tolerance_kt
        pred_dips = (pred_traj[:, 1] < pred_traj[:, 0] - tol) & (pred_traj[:, 2] > pred_traj[:, 1] + tol)
        true_did_not_dip = ~((true_traj[:, 1] < true_traj[:, 0] - tol) & (true_traj[:, 2] > true_traj[:, 1] + tol))
        false_dip_mask = pred_dips & true_did_not_dip
        false_dip_count = int(np.sum(false_dip_mask))
        false_dip_rate_pct = round(100.0 * false_dip_count / max(n, 1), 3)

        # False Peak: predicted peaks at 6h then drops at 12h, while true trajectory did NOT peak
        pred_peaks = (pred_traj[:, 1] > pred_traj[:, 0] + tol) & (pred_traj[:, 2] < pred_traj[:, 1] - tol)
        true_did_not_peak = ~((true_traj[:, 1] > true_traj[:, 0] + tol) & (true_traj[:, 2] < true_traj[:, 1] - tol))
        false_peak_mask = pred_peaks & true_did_not_peak
        false_peak_count = int(np.sum(false_peak_mask))
        false_peak_rate_pct = round(100.0 * false_peak_count / max(n, 1), 3)

        # 6. Maximum Absolute 6h Forecast Change
        max_6h_changes = np.maximum(
            np.abs(pred_d1_0_6),
            np.maximum(np.abs(pred_d1_6_12), np.abs(pred_d1_12_24_norm))
        )
        max_6h_change_mean = float(np.mean(max_6h_changes))
        max_6h_change_p95 = float(np.percentile(max_6h_changes, 95))
        max_6h_change_p99 = float(np.percentile(max_6h_changes, 99))
        max_6h_change_max = float(np.max(max_6h_changes))

        # Fraction of changes exceeding thresholds
        threshold_exceedances = {}
        for thr in self.step_thresholds_kt:
            count = int(np.sum(max_6h_changes > thr))
            pct = round(100.0 * count / max(n, 1), 3)
            threshold_exceedances[f"pct_gt_{int(thr)}kt_per_6h"] = pct

        # 7. Directional Accuracy (Intensify vs Steady vs Weaken)
        deadband = self.steady_deadband_kt
        def get_direction(delta):
            # +1 for intensify, -1 for weaken, 0 for steady
            direction = np.zeros_like(delta, dtype=int)
            direction[delta > deadband] = 1
            direction[delta < -deadband] = -1
            return direction

        pred_dir = get_direction(pred_delta)
        true_dir = get_direction(true_delta)
        dir_acc_per_horizon = [
            round(100.0 * float(np.mean(pred_dir[:, i] == true_dir[:, i])), 2)
            for i in range(3)
        ]
        overall_dir_acc = round(100.0 * float(np.mean(pred_dir == true_dir)), 2)

        # 8. Horizon Inversion Rate (e.g. V(+6h) > V(+12h) when true V(+6h) <= V(+12h) or vice versa)
        pred_slope_6_12 = pred_traj[:, 2] - pred_traj[:, 1]
        true_slope_6_12 = true_traj[:, 2] - true_traj[:, 1]
        inversion_mask = (pred_slope_6_12 * true_slope_6_12) < - (deadband ** 2)
        inversion_count = int(np.sum(inversion_mask))
        inversion_rate_pct = round(100.0 * inversion_count / max(n, 1), 3)

        return {
            "total_samples": n,
            "overall_mae": round(overall_mae, 3),
            "overall_rmse": round(overall_rmse, 3),
            "overall_bias": round(overall_bias, 3),
            "mae_per_horizon": [round(x, 3) for x in mae_per_horizon],
            "rmse_per_horizon": [round(x, 3) for x in rmse_per_horizon],
            "bias_per_horizon": [round(x, 3) for x in bias_per_horizon],
            "delta_v_mae": [round(x, 3) for x in delta_mae],
            "delta_v_rmse": [round(x, 3) for x in delta_rmse],
            "delta_v_bias": [round(x, 3) for x in delta_bias],
            "mean_first_diff_error_kt": round(mean_first_diff_error, 3),
            "second_diff_error_kt": round(second_diff_error, 3),
            "predicted_trajectory_roughness": round(pred_roughness_6h, 3),
            "true_trajectory_roughness": round(true_roughness_6h, 3),
            "roughness_ratio_pred_vs_true": roughness_ratio,
            "false_dip_count": false_dip_count,
            "false_dip_rate_pct": false_dip_rate_pct,
            "false_peak_count": false_peak_count,
            "false_peak_rate_pct": false_peak_rate_pct,
            "max_6h_change_mean_kt": round(max_6h_change_mean, 2),
            "max_6h_change_p95_kt": round(max_6h_change_p95, 2),
            "max_6h_change_p99_kt": round(max_6h_change_p99, 2),
            "max_6h_change_max_kt": round(max_6h_change_max, 2),
            "threshold_exceedances": threshold_exceedances,
            "directional_accuracy_pct": overall_dir_acc,
            "directional_accuracy_per_horizon_pct": dir_acc_per_horizon,
            "trajectory_inversion_rate_pct": inversion_rate_pct,
        }


class PhysicalSanityChecker:
    """Diagnostic inspector for physical plausibility of tropical cyclone intensity forecasts.

    Inspects:
      - Negative predicted intensities (< min_intensity_kt, default 0 kt)
      - Implausibly high intensities (> max_plausible_kt, default 200 kt)
      - Extreme single-step rate of change (> large_step_change_kt, default 45 kt / 6h)
      - Extreme 24h intensity delta (> max_24h_delta_kt, default 80 kt)
      - Numerical anomalies (NaNs, infs)
    """

    def __init__(
        self,
        min_intensity_kt: float = 0.0,
        max_plausible_kt: float = 200.0,
        large_step_change_kt: float = 45.0,
        max_24h_delta_kt: float = 80.0,
    ):
        self.min_intensity_kt = min_intensity_kt
        self.max_plausible_kt = max_plausible_kt
        self.large_step_change_kt = large_step_change_kt
        self.max_24h_delta_kt = max_24h_delta_kt

    def inspect(
        self,
        predictions: np.ndarray,
        v_curr: Optional[np.ndarray] = None,
        return_flag_masks: bool = False,
    ) -> Dict[str, Union[int, float, Dict]]:
        """Inspects predictions without altering a single value.

        Args:
            predictions: (N, 3) array of forecasted intensities for [+6h, +12h, +24h]
            v_curr: Optional (N,) array of current intensities at time t
            return_flag_masks: Whether to return boolean mask arrays of flagged rows
        """
        preds = np.asarray(predictions, dtype=float)
        n_samples = len(preds)

        # 1. NaN and Inf checks
        nan_count = int(np.sum(np.isnan(preds)))
        inf_count = int(np.sum(np.isinf(preds)))

        # 2. Negative intensity flags
        neg_mask = np.any(preds < self.min_intensity_kt, axis=1)
        neg_count = int(np.sum(neg_mask))

        # 3. Super-plausible ceiling flags (> max_plausible_kt)
        high_mask = np.any(preds > self.max_plausible_kt, axis=1)
        high_count = int(np.sum(high_mask))

        # 4. Large step changes
        large_step_flags = np.zeros(n_samples, dtype=bool)
        if v_curr is not None:
            v_curr_arr = np.asarray(v_curr, dtype=float).reshape(-1)
            step_0_6 = np.abs(preds[:, 0] - v_curr_arr)
            large_step_flags |= (step_0_6 > self.large_step_change_kt)

        step_6_12 = np.abs(preds[:, 1] - preds[:, 0])
        large_step_flags |= (step_6_12 > self.large_step_change_kt)

        # Delta 12 to 24h: 12-hour span -> threshold scaled by 2
        step_12_24 = np.abs(preds[:, 2] - preds[:, 1])
        large_step_flags |= (step_12_24 > (self.large_step_change_kt * 2.0))

        large_step_count = int(np.sum(large_step_flags))

        # 5. Extreme 24h Delta
        large_24h_flags = np.zeros(n_samples, dtype=bool)
        if v_curr is not None:
            delta_24 = np.abs(preds[:, 2] - v_curr_arr)
            large_24h_flags = delta_24 > self.max_24h_delta_kt
        large_24h_count = int(np.sum(large_24h_flags))

        # Total unique samples flagged
        any_flag = neg_mask | high_mask | large_step_flags | large_24h_flags
        total_flagged = int(np.sum(any_flag))

        report = {
            "total_samples": n_samples,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "negative_intensity_count": neg_count,
            "negative_intensity_pct": round(100.0 * neg_count / max(n_samples, 1), 2),
            "exceeds_ceiling_count": high_count,
            "exceeds_ceiling_pct": round(100.0 * high_count / max(n_samples, 1), 2),
            "large_single_step_count": large_step_count,
            "large_single_step_pct": round(100.0 * large_step_count / max(n_samples, 1), 2),
            "extreme_24h_delta_count": large_24h_count,
            "extreme_24h_delta_pct": round(100.0 * large_24h_count / max(n_samples, 1), 2),
            "total_flagged_samples": total_flagged,
            "total_flagged_pct": round(100.0 * total_flagged / max(n_samples, 1), 2),
            "status": "CLEAN" if total_flagged == 0 and nan_count == 0 else "ANOMALIES_DETECTED",
        }

        if return_flag_masks:
            report["flag_masks"] = {
                "negative": neg_mask,
                "high": high_mask,
                "large_step": large_step_flags,
                "extreme_24h": large_24h_flags,
                "any": any_flag,
            }

        return report

