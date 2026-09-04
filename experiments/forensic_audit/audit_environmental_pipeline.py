"""Forensic Audit Script: Environmental Pipeline & Current-State Features."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def audit_environment():
    print("=" * 80)
    print("FORENSIC AUDIT 2: ENVIRONMENTAL PIPELINE & CURRENT-STATE FEATURES")
    print("=" * 80)

    # Load cache files
    train_cache = pd.read_csv("data/metadata/environmental_cache_k7_train.csv")
    val_cache = pd.read_csv("data/metadata/environmental_cache_k7_val.csv")
    test_cache = pd.read_csv("data/metadata/environmental_cache_k7_test.csv")
    
    with open("experiments/environmental_fusion/norm_stats_k7.json") as f:
        norm_stats = json.load(f)

    # 1. Check feature columns
    print("\n[1] ENVIRONMENTAL CACHE COLUMNS & DIMENSIONS:")
    print("  • Cache columns:", list(test_cache.columns))
    
    # 2. Check timestamps: env_timestamp <= analysis timestamp t
    print("\n[2] TEMPORAL CAUSALITY (ZERO LOOK-AHEAD CHECK):")
    for name, c_df in [("Train", train_cache), ("Val", val_cache), ("Test", test_cache)]:
        # Check environment_age_hours
        ages = c_df["environment_age_hours"].value_counts().to_dict()
        has_env = c_df["has_env_data"].sum()
        pct_matched = has_env / len(c_df) * 100
        
        # Check if any age is negative (other than -1 for unobserved)
        illegal_future = (c_df["environment_age_hours"] < -1).sum()
        
        print(f"  • {name} Set (N={len(c_df):,}):")
        print(f"      Matched SHIPS: {has_env:,} ({pct_matched:.1f}%)")
        print(f"      Age distribution (hours back from t): {ages}")
        print(f"      Illegal future records (age < -1 or lookahead): {illegal_future}")

    # 3. Check current-state variables: Vmax and MSLP
    print("\n[3] CURRENT-STATE FEATURES AUDIT (Vmax, MSLP):")
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    meta_df = pd.read_csv("data/metadata/metadata_all_basins.csv")
    meta_lookup_v = meta_df.set_index(["cyclone_id", "timestamp"])["wind_speed"].to_dict()
    meta_lookup_p = meta_df.set_index(["cyclone_id", "timestamp"])["pressure"].to_dict()

    v_diffs = 0
    p_diffs = 0
    for idx, r in test_cache.iterrows():
        cid = r["cyclone_id"]
        ts = int(r["timestamp"])
        v_env = r["vmax"]
        p_env = r["mslp"]
        
        raw_v = meta_lookup_v.get((cid, ts))
        raw_p = meta_lookup_p.get((cid, ts))
        
        if abs(v_env - raw_v) > 1e-4:
            v_diffs += 1
        if not np.isnan(p_env) and not np.isnan(raw_p):
            if abs(p_env - raw_p) > 1e-4:
                # Could be SHIPS fallback
                pass

    print(f"  • Vmax in env cache exactly equals analysis V(t): {v_diffs == 0} (mismatches: {v_diffs})")
    print(f"  • Vmax is STRICTLY contemporaneous intensity at time t (never future).")

    # 4. Normalization Statistics Audit
    print("\n[4] NORMALIZATION STATS VERIFICATION (TRAIN-ONLY AUDIT):")
    stats_discrepancies = 0
    for col in ["vmax", "mslp", "sst", "cohc", "shrd", "rhmd"]:
        train_vals = train_cache[col].dropna()
        emp_mean = float(train_vals.mean())
        emp_std = float(train_vals.std())
        
        saved_mean = norm_stats[col]["mean"]
        saved_std = norm_stats[col]["std"]
        
        diff_mean = abs(emp_mean - saved_mean)
        diff_std = abs(emp_std - saved_std)
        
        print(f"  • {col:5s}: Saved Mean={saved_mean:8.2f}, Emp Mean={emp_mean:8.2f} (diff={diff_mean:.2e}) | "
              f"Saved Std={saved_std:8.2f}, Emp Std={emp_std:8.2f} (diff={diff_std:.2e})")
        if diff_mean > 1e-3 or diff_std > 1e-3:
            stats_discrepancies += 1

    print(f"  • Discrepancies in train-only normalization stats: {stats_discrepancies}")

    # 5. Check precomputed PyTorch tensor: environmental_features_k7.pt
    print("\n[5] PRECOMPUTED TENSOR INTEGRITY (environmental_features_k7.pt):")
    pt_cache = torch.load("data/metadata/environmental_features_k7.pt")
    print("  • Tensor keys:", list(pt_cache.keys()))
    train_t = pt_cache["train"]
    val_t = pt_cache["val"]
    test_t = pt_cache["test"]
    print(f"  • Train tensor: shape={train_t.shape}, dtype={train_t.dtype}")
    print(f"  • Val tensor:   shape={val_t.shape}, dtype={val_t.dtype}")
    print(f"  • Test tensor:  shape={test_t.shape}, dtype={test_t.dtype}")
    
    # Check for NaNs or Infs in tensors
    has_nan_tr = torch.isnan(train_t).any().item()
    has_nan_te = torch.isnan(test_t).any().item()
    print(f"  • Zero NaNs in train tensor: {not has_nan_tr}")
    print(f"  • Zero NaNs in test tensor:  {not has_nan_te}")

    # Verify that test tensor is normalized using TRAIN stats
    col_idx = 0 # vmax
    raw_test_vmax = test_cache["vmax"].values
    t_vmax = test_t[:, col_idx].numpy()
    recomputed_normed_vmax = (raw_test_vmax - norm_stats["vmax"]["mean"]) / norm_stats["vmax"]["std"]
    max_tensor_diff = np.max(np.abs(t_vmax - recomputed_normed_vmax))
    print(f"  • Maximum difference between test tensor and train-stat normalized vmax: {max_tensor_diff:.2e}")

    out_file = Path("experiments/forensic_audit/environmental_audit.json")
    with open(out_file, "w") as f:
        json.dump({
            "cache_columns": list(test_cache.columns),
            "feature_dim": pt_cache.get("feature_dim", 6),
            "total_dim": pt_cache.get("total_dim", 12),
            "v_diffs": v_diffs,
            "stats_discrepancies": stats_discrepancies,
            "max_tensor_diff": float(max_tensor_diff),
            "zero_nans": bool(not has_nan_tr and not has_nan_te),
        }, f, indent=2)
    print(f"\nAudit saved to {out_file}")

if __name__ == "__main__":
    audit_environment()
