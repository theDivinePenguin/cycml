"""Compute multi-channel normalization statistics strictly on the training set split.

Zero leakage protocol: Only frames listed in data/metadata/train_metadata_all_basins.csv
are accessed. Output is saved to data/metadata/normalization_stats_multichannel.json.
"""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def compute_multichannel_normalization_stats(
    train_metadata_path: str | Path = "data/metadata/train_metadata_all_basins.csv",
    save_path: str | Path = "data/metadata/normalization_stats_multichannel.json",
    batch_size: int = 1000
) -> dict:
    """Compute mean, std, min, and max for each channel strictly on the training set."""
    train_df = pd.read_csv(train_metadata_path)
    print(f"[Stats] Computing multi-channel training normalization stats over {len(train_df):,} training frames...")
    print(f"  • Zero-leakage verification: Only training split frames are accessed.")
    
    channel_names = ["IR1", "WV", "VIS", "PMW"]
    num_channels = 4
    
    # Accumulators for streaming mean and variance (Welford's / sum-of-squares)
    pixel_counts = [0] * num_channels
    sums = [0.0] * num_channels
    sum_sqs = [0.0] * num_channels
    mins = [float("inf")] * num_channels
    maxs = [float("-inf")] * num_channels
    
    grouped = train_df.groupby("h5_file")
    
    for h5_file, group_df in grouped:
        h5_p = Path(h5_file)
        if not h5_p.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_p}")
        
        row_indices = group_df["h5_row_index"].astype(int).tolist()
        print(f"  • Reading {len(row_indices):,} training frames from {h5_p.name}...")
        
        with h5py.File(h5_p, "r") as hf:
            matrix_ds = hf["matrix"]
            
            for start_i in range(0, len(row_indices), batch_size):
                chunk_indices = sorted(row_indices[start_i:start_i + batch_size])
                chunk = matrix_ds[chunk_indices]  # Shape: (B, 201, 201, 4)
                
                for c in range(num_channels):
                    ch_data = np.copy(chunk[:, :, :, c])
                    
                    # Preprocessing / Imputation for stats:
                    if c == 3:
                        # PMW: remove huge fill values (>1e20) and NaNs, impute with 0.0
                        ch_data[(ch_data > 1e20) | (ch_data < -100) | np.isnan(ch_data)] = 0.0
                    elif c == 2:
                        # VIS: nighttime NaNs -> 0.0
                        ch_data[np.isnan(ch_data)] = 0.0
                    else:
                        # IR1 and WV: replace isolated NaNs with valid channel mean estimate
                        valid_mask = ~np.isnan(ch_data)
                        fill_val = np.mean(ch_data[valid_mask]) if np.any(valid_mask) else (267.0 if c == 0 else 236.0)
                        ch_data[~valid_mask] = fill_val
                        
                    n_pix = ch_data.size
                    pixel_counts[c] += n_pix
                    sums[c] += float(np.sum(ch_data))
                    sum_sqs[c] += float(np.sum(ch_data ** 2))
                    mins[c] = min(mins[c], float(np.min(ch_data)))
                    maxs[c] = max(maxs[c], float(np.max(ch_data)))
    
    stats = {
        "n_training_samples": len(train_df),
        "total_pixels_per_channel": pixel_counts[0],
        "channels": {}
    }
    
    means = []
    stds = []
    
    print("\n" + "=" * 70)
    print("TRAINING SET MULTI-CHANNEL NORMALIZATION STATISTICS (NO LEAKAGE)")
    print("=" * 70)
    
    for c in range(num_channels):
        mean_c = sums[c] / pixel_counts[c]
        variance_c = (sum_sqs[c] / pixel_counts[c]) - (mean_c ** 2)
        std_c = float(np.sqrt(max(variance_c, 1e-6)))
        
        means.append(round(mean_c, 4))
        stds.append(round(std_c, 4))
        
        ch_dict = {
            "channel_idx": c,
            "name": channel_names[c],
            "mean": round(mean_c, 4),
            "std": round(std_c, 4),
            "min": round(mins[c], 4),
            "max": round(maxs[c], 4)
        }
        stats["channels"][str(c)] = ch_dict
        
        print(f"Channel {c} ({channel_names[c]:3s}): Mean = {mean_c:9.4f} | Std = {std_c:9.4f} | Range = [{mins[c]:7.2f}, {maxs[c]:7.2f}]")
    
    stats["mean"] = means
    stats["std"] = stds
    
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    with open(save_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Stats] Saved multi-channel normalization statistics to: {save_p}")
    print("=" * 70)
    
    return stats


if __name__ == "__main__":
    compute_multichannel_normalization_stats()
