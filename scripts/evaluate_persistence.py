"""Evaluate Oracle Persistence and Current-CNN Hold-Forward Baselines for Multi-Horizon Forecasting."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import torch

from src.models.resnet import CycloneResNet


def compute_regression_metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    errors = preds - actuals
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))
    median_ae = float(np.median(abs_errors))
    mean_bias = float(np.mean(errors))
    
    if len(preds) > 1 and np.std(preds) > 1e-6 and np.std(actuals) > 1e-6:
        pearson_r, _ = stats.pearsonr(preds, actuals)
        pearson_r = float(pearson_r)
    else:
        pearson_r = 0.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "pearson_r": round(pearson_r, 4),
        "median_ae": round(median_ae, 3),
        "mean_bias": round(mean_bias, 3),
        "n_samples": int(len(preds))
    }


def paired_cyclone_block_bootstrap(seq_df: pd.DataFrame, preds_dict: dict, n_bootstraps: int = 1000) -> dict:
    """Run 1,000-iteration cyclone-level block bootstrap for confidence intervals."""
    cyclone_ids = seq_df["cyclone_id"].values
    unique_cyclones = np.unique(cyclone_ids)
    n_cyclones = len(unique_cyclones)
    
    cyclone_to_indices = {cid: np.where(cyclone_ids == cid)[0] for cid in unique_cyclones}
    
    bootstrap_results = {}
    
    for model_name, preds_arr in preds_dict.items():
        # preds_arr: shape (N, 3) for +6h, +12h, +24h
        bootstrap_results[model_name] = {}
        for h_idx, h_name in enumerate(["+6h", "+12h", "+24h"]):
            target_col = f"vmax_plus_{h_name[1:]}"
            actuals = seq_df[target_col].values
            p = preds_arr[:, h_idx]
            
            mae_samples = []
            for _ in range(n_bootstraps):
                sampled_cids = np.random.choice(unique_cyclones, size=n_cyclones, replace=True)
                sample_idx = np.concatenate([cyclone_to_indices[cid] for cid in sampled_cids])
                b_mae = np.mean(np.abs(p[sample_idx] - actuals[sample_idx]))
                mae_samples.append(b_mae)
            
            ci_low = float(np.percentile(mae_samples, 2.5))
            ci_high = float(np.percentile(mae_samples, 97.5))
            bootstrap_results[model_name][h_name] = {
                "ci95_low": round(ci_low, 3),
                "ci95_high": round(ci_high, 3)
            }
            
    return bootstrap_results


def evaluate_baselines():
    meta_dir = Path("data/metadata")
    test_seq_path = meta_dir / "forecast_test_sequences_k5.csv"
    assert test_seq_path.exists(), "Test sequences not found!"
    test_seq_df = pd.read_csv(test_seq_path)

    print("=" * 90)
    print(f"EVALUATING FORECASTING BASELINES ON {len(test_seq_df):,} TEST SEQUENCES (191 CYCLONES)")
    print("=" * 90)

    # 1. Oracle Persistence: V(t+Δt) = V(t)
    v_curr = test_seq_df["vmax_curr"].values
    oracle_preds = np.stack([v_curr, v_curr, v_curr], axis=1)  # (N, 3)

    # 2. Current-CNN Hold-Forward: V(t+Δt) = \hat{V}(t)
    # Load current ResNet18 model to predict on frame t
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_ckpt_path = Path("experiments/modality_ablation/ir1_wv_vis/best.pt")
    
    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    mean = np.array([norm_stats["mean"][c] for c in [0, 1, 2]], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([norm_stats["std"][c] for c in [0, 1, 2]], dtype=np.float32).reshape(1, 3, 1, 1)

    print(f"\n[Current-CNN Hold-Forward] Loading ResNet18 from {cnn_ckpt_path}...")
    cnn_model = CycloneResNet(architecture="resnet18", in_channels=3)
    checkpoint = torch.load(cnn_ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    cnn_model.load_state_dict(state_dict)
    cnn_model.to(device).eval()

    # Predict V_hat(t) for all test sequences
    cnn_v_curr_preds = []
    
    # Open HDF5 files lazily
    h5_handles = {}
    def get_h5(p):
        if p not in h5_handles:
            h5_handles[p] = h5py.File(p, "r", swmr=True)
        return h5_handles[p]

    batch_frames = []
    with torch.no_grad():
        for idx, row in test_seq_df.iterrows():
            hist_files = json.loads(row["history_h5_files"])
            hist_rows = json.loads(row["history_h5_rows"])
            # Frame t is the last frame (index -1)
            f_file, f_row = hist_files[-1], hist_rows[-1]
            raw = get_h5(f_file)["matrix"][f_row, :, :, [0, 1, 2]].astype(np.float32)
            # Clean missing values
            raw[np.isnan(raw) | np.isinf(raw) | (raw > 1e20) | (raw < -1e20)] = 0.0
            frame_t = np.transpose(raw, (2, 0, 1))  # (3, H, W)
            frame_t = (frame_t - mean[0]) / (std[0] + 1e-7)
            batch_frames.append(frame_t)

            if len(batch_frames) == 64 or idx == len(test_seq_df) - 1:
                b_tensor = torch.from_numpy(np.stack(batch_frames, axis=0)).float().to(device)
                out = cnn_model(b_tensor).cpu().numpy().flatten()
                cnn_v_curr_preds.extend(out.tolist())
                batch_frames = []

    for h in h5_handles.values():
        h.close()

    cnn_v_curr_arr = np.array(cnn_v_curr_preds, dtype=np.float32)
    cnn_hold_preds = np.stack([cnn_v_curr_arr, cnn_v_curr_arr, cnn_v_curr_arr], axis=1)  # (N, 3)

    # Compute metrics for both baselines
    results = {
        "Oracle Persistence": {},
        "Current-CNN Hold-Forward": {}
    }

    preds_dict = {
        "Oracle Persistence": oracle_preds,
        "Current-CNN Hold-Forward": cnn_hold_preds
    }

    for model_name, p_arr in preds_dict.items():
        for h_idx, h_name in enumerate(["+6h", "+12h", "+24h"]):
            target_col = f"vmax_plus_{h_name[1:]}"
            actuals = test_seq_df[target_col].values
            m = compute_regression_metrics(p_arr[:, h_idx], actuals)
            results[model_name][h_name] = m

    # Compute 1,000-sample bootstrap CIs
    print("\nRunning 1,000-Iteration Cyclone-Level Paired Block Bootstrap for Baselines...")
    np.random.seed(42)
    boot_res = paired_cyclone_block_bootstrap(test_seq_df, preds_dict, n_bootstraps=1000)
    for model_name in results:
        for h_name in ["+6h", "+12h", "+24h"]:
            results[model_name][h_name]["ci95"] = [
                boot_res[model_name][h_name]["ci95_low"],
                boot_res[model_name][h_name]["ci95_high"]
            ]

    print("\n" + "=" * 90)
    print("BASELINE FORECASTING RESULTS (TEST SET, 8,279 SEQUENCES)")
    print("=" * 90)
    for model_name, h_dict in results.items():
        print(f"\n[{model_name.upper()}]:")
        for h_name, m in h_dict.items():
            print(f"  • {h_name:5s} -> MAE: {m['mae']:5.3f} kt (95% CI: [{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]) | "
                  f"RMSE: {m['rmse']:5.3f} kt | R²: {m['r2']:6.3f} | Pearson r: {m['pearson_r']:5.3f} | Bias: {m['mean_bias']:+5.2f} kt")

    # Save baseline metrics
    out_dir = Path("experiments/forecasting/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved Baseline Metrics] -> {out_path}")

    # Save baseline predictions CSV
    pred_df = pd.DataFrame({
        "cyclone_id": test_seq_df["cyclone_id"],
        "target_t_timestamp": test_seq_df["target_t_timestamp"],
        "vmax_curr": test_seq_df["vmax_curr"],
        "actual_plus_6h": test_seq_df["vmax_plus_6h"],
        "actual_plus_12h": test_seq_df["vmax_plus_12h"],
        "actual_plus_24h": test_seq_df["vmax_plus_24h"],
        "oracle_plus_6h": oracle_preds[:, 0],
        "oracle_plus_12h": oracle_preds[:, 1],
        "oracle_plus_24h": oracle_preds[:, 2],
        "cnn_hold_plus_6h": cnn_hold_preds[:, 0],
        "cnn_hold_plus_12h": cnn_hold_preds[:, 1],
        "cnn_hold_plus_24h": cnn_hold_preds[:, 2],
    })
    pred_csv_path = out_dir / "baseline_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[Saved Baseline Predictions] -> {pred_csv_path}")

    return results


if __name__ == "__main__":
    evaluate_baselines()
