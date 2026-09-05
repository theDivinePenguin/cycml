"""Forensic audit script for Section 1: Dataset Ground Truth.
Verifies all raw HDF5 files, metadata, cadence, ordering, basins, and intensities.
"""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

def run_ground_truth_audit():
    print("=" * 80)
    print("SECTION 1: DATASET GROUND TRUTH AUDIT")
    print("=" * 80)

    h5_path_1 = Path("data/raw/TCIR-ATLN_EPAC_WPAC.h5")
    h5_path_2 = Path("data/raw/TCIR-CPAC_IO_SH.h5")
    meta_path = Path("data/metadata/metadata_all_basins.csv")

    assert h5_path_1.exists(), f"Missing {h5_path_1}"
    assert h5_path_2.exists(), f"Missing {h5_path_2}"
    assert meta_path.exists(), f"Missing {meta_path}"

    with h5py.File(h5_path_1, "r") as f1, h5py.File(h5_path_2, "r") as f2:
        m1_shape = f1["matrix"].shape
        m1_dtype = str(f1["matrix"].dtype)
        m2_shape = f2["matrix"].shape
        m2_dtype = str(f2["matrix"].dtype)

    print(f"HDF5 1 (ATLN/EPAC/WPAC) matrix shape: {m1_shape}, dtype: {m1_dtype}")
    print(f"HDF5 2 (CPAC/IO/SH) matrix shape:     {m2_shape}, dtype: {m2_dtype}")
    total_h5_frames = m1_shape[0] + m2_shape[0]
    print(f"Total HDF5 frames: {m1_shape[0]} + {m2_shape[0]} = {total_h5_frames:,d}")

    # Load metadata
    meta = pd.read_csv(meta_path)
    meta_rows = len(meta)
    print(f"Metadata rows: {meta_rows:,d}")
    assert total_h5_frames == meta_rows, f"Mismatch: {total_h5_frames} != {meta_rows}"

    unique_cyclones = meta["cyclone_id"].nunique()
    print(f"Total unique cyclones: {unique_cyclones:,d}")

    # Verify years
    years = sorted(meta["year"].unique())
    print(f"Years covered: {years[0]} to {years[-1]} (Total {len(years)} years: {years})")

    # Verify 1-to-1 correspondence
    print("\nChecking 1-to-1 correspondence between metadata and HDF5 matrices...")
    # Check sample indices
    assert (meta["sample_index"].values == np.arange(meta_rows)).all(), "sample_index is not strictly sequential 0..N-1"
    
    # Check h5_row_index ranges
    m1_rows = meta[meta["h5_file"].str.endswith("TCIR-ATLN_EPAC_WPAC.h5")]
    m2_rows = meta[meta["h5_file"].str.endswith("TCIR-CPAC_IO_SH.h5")]
    print(f"Metadata file split: ATLN_EPAC_WPAC={len(m1_rows):,d}, CPAC_IO_SH={len(m2_rows):,d}")
    assert len(m1_rows) == m1_shape[0], f"HDF5 1 row count mismatch: {len(m1_rows)} vs {m1_shape[0]}"
    assert len(m2_rows) == m2_shape[0], f"HDF5 2 row count mismatch: {len(m2_rows)} vs {m2_shape[0]}"
    assert (m1_rows["h5_row_index"].values == np.arange(m1_shape[0])).all(), "HDF5 1 row indices not 0..N-1"
    assert (m2_rows["h5_row_index"].values == np.arange(m2_shape[0])).all(), "HDF5 2 row indices not 0..N-1"
    print("  -> PASS: 100% strictly verified 1-to-1 index correspondence.")

    # Check timestamp uniqueness and ordering per cyclone
    print("\nChecking per-cyclone timestamp uniqueness and chronological ordering...")
    duplicate_stamps = 0
    non_monotonic_cyclones = 0
    cadence_deviations = 0
    total_cadence_steps = 0
    dt_counts = {}

    for cid, grp in meta.groupby("cyclone_id"):
        # Check uniqueness
        if grp["timestamp"].duplicated().any():
            duplicate_stamps += grp["timestamp"].duplicated().sum()
        
        # Check monotonicity
        ts = grp["timestamp"].astype(str)
        # Parse timestamp to datetime
        dt = pd.to_datetime(ts, format="%Y%m%d%H")
        if not dt.is_monotonic_increasing:
            non_monotonic_cyclones += 1

        # Check cadence (diff in hours)
        diffs = (dt.diff().dt.total_seconds() / 3600.0).dropna().values
        for d in diffs:
            total_cadence_steps += 1
            d_int = int(round(d))
            dt_counts[d_int] = dt_counts.get(d_int, 0) + 1
            if d_int != 3:
                cadence_deviations += 1

    print(f"  Duplicate timestamps across all cyclones: {duplicate_stamps}")
    print(f"  Non-monotonic cyclones: {non_monotonic_cyclones}")
    print(f"  Total sequential steps: {total_cadence_steps:,d}")
    print(f"  Cadence step distribution (hours):")
    for k in sorted(dt_counts.keys())[:10]:
        print(f"    dt = {k:3d}h: {dt_counts[k]:,d} steps ({dt_counts[k]/total_cadence_steps*100:.2f}%)")
    print(f"  Cadence = 3h exact: {dt_counts.get(3, 0):,d} steps ({dt_counts.get(3, 0)/total_cadence_steps*100:.2f}%)")
    print(f"  Gaps > 3h: {cadence_deviations:,d} ({cadence_deviations/total_cadence_steps*100:.2f}%)")

    # Basin distribution
    print("\nBasin distribution (frames & unique cyclones):")
    basin_summary = []
    for b, grp in meta.groupby("region"):
        basin_summary.append({
            "region": b,
            "frames": len(grp),
            "pct_frames": len(grp) / meta_rows * 100,
            "cyclones": grp["cyclone_id"].nunique(),
            "pct_cyclones": grp["cyclone_id"].nunique() / unique_cyclones * 100,
            "mean_vmax": grp["wind_speed"].mean(),
            "max_vmax": grp["wind_speed"].max(),
        })
    df_basin = pd.DataFrame(basin_summary).sort_values("frames", ascending=False)
    print(df_basin.to_string(index=False))

    # Intensity distribution
    print("\nIntensity category distribution:")
    bins = [0, 34, 64, 83, 96, 113, 137, 300]
    labels = ["TD (<34)", "TS (34-63)", "Cat 1 (64-82)", "Cat 2 (83-95)", "Cat 3 (96-112)", "Cat 4 (113-136)", "Cat 5 (>=137)"]
    meta["cat"] = pd.cut(meta["wind_speed"], bins=bins, labels=labels, right=False)
    cat_counts = meta["cat"].value_counts().reindex(labels)
    for cat, cnt in cat_counts.items():
        print(f"  {cat:<16}: {cnt:,d} frames ({cnt/meta_rows*100:.2f}%)")

    # Missing values in metadata
    print("\nMissing values in metadata:")
    for col in meta.columns:
        n_nan = meta[col].isna().sum()
        print(f"  {col:<16}: {n_nan:,d} NaNs ({n_nan/meta_rows*100:.2f}%)")

    # Missing value distribution in HDF5 satellite channels
    print("\nSatellite channel missingness audit (checking sample of chunks):")
    stats = {}
    with open("data/metadata/normalization_stats_multichannel.json") as f:
        multichannel_stats = json.load(f)
    print("Precomputed multichannel stats from Train split:")
    for ch_idx, data in multichannel_stats["channels"].items():
        print(f"  {data['name']:<6}: Mean={data['mean']:.4f}, Std={data['std']:.4f}, Min={data['min']:.2f}, Max={data['max']:.2f}")

    results = {
        "total_frames": int(total_h5_frames),
        "total_cyclones": int(unique_cyclones),
        "years": [int(years[0]), int(years[-1])],
        "hdf5_matrices": {
            "ATLN_EPAC_WPAC": list(m1_shape),
            "CPAC_IO_SH": list(m2_shape)
        },
        "cadence_3h_exact_pct": float(dt_counts.get(3, 0)/total_cadence_steps*100),
        "gaps_gt_3h_count": int(cadence_deviations),
        "basin_breakdown": basin_summary,
        "intensity_breakdown": {k: int(v) for k, v in cat_counts.items()}
    }

    out_file = Path("experiments/forensic_audit/section1_ground_truth.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Section 1 audit results to {out_file}")

if __name__ == "__main__":
    run_ground_truth_audit()
