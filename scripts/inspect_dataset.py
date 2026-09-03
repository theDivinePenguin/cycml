"""Script to inspect TCIR HDF5 dataset structure and report authoritative statistics."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from src.data.downloader import download_tcir_archive
from src.data.metadata import load_tcir_info_table, parse_and_normalize_metadata


def inspect_hdf5(h5_path: Path) -> dict:
    """Inspect the internal structure of TCIR HDF5 file."""
    print("=" * 70)
    print(f"Authoritative Dataset Inspection: {h5_path.name}")
    print("=" * 70)

    file_size_gb = h5_path.stat().st_size / 1e9
    print(f"File Size: {file_size_gb:.2f} GB ({h5_path.stat().st_size:,} bytes)")

    stats = {}
    with h5py.File(h5_path, "r") as hf:
        print("\nHDF5 Root Keys:")
        for k in hf.keys():
            item = hf[k]
            if isinstance(item, h5py.Dataset):
                print(f"  • Dataset '{k}': shape={item.shape}, dtype={item.dtype}, chunks={item.chunks}")
                stats[f"{k}_shape"] = item.shape
                stats[f"{k}_dtype"] = str(item.dtype)
            elif isinstance(item, h5py.Group):
                print(f"  • Group '{k}': keys={list(item.keys())}")
                stats[f"{k}_group_keys"] = list(item.keys())

        # Inspect matrix dataset in detail
        if "matrix" in hf:
            matrix_ds = hf["matrix"]
            n_samples, h, w, n_channels = matrix_ds.shape
            print(f"\nImage Tensor Specification:")
            print(f"  • Total frames (N):      {n_samples:,}")
            print(f"  • Spatial dimensions:    {h} × {w} pixels")
            print(f"  • Channels (C):          {n_channels}")

            # Inspect a small sample of channel values
            sample_slice = matrix_ds[:100]  # Shape (100, 201, 201, 4)
            print("\nChannel Sample Physical Value Ranges (first 100 frames):")
            channel_names = ["Channel 0 (IR1 - Infrared)", "Channel 1 (WV - Water Vapor)", "Channel 2 (VIS - Visible)", "Channel 3 (PMW - Passive Microwave)"]
            for c in range(min(n_channels, 4)):
                c_data = sample_slice[:, :, :, c]
                nan_count = np.isnan(c_data).sum()
                valid_data = c_data[~np.isnan(c_data)]
                if len(valid_data) > 0:
                    print(f"  • {channel_names[c]}: min={valid_data.min():.2f}, max={valid_data.max():.2f}, mean={valid_data.mean():.2f}, std={valid_data.std():.2f}, NaN count={nan_count}")
                else:
                    print(f"  • {channel_names[c]}: All NaNs")

    # Inspect info table
    df = parse_and_normalize_metadata(h5_path)
    print("\nMetadata Table Summary:")
    print(f"  • Total valid samples:    {len(df):,}")
    print(f"  • Unique cyclone IDs:     {df['cyclone_id'].nunique():,}")
    print(f"  • Label column:           'wind_speed' (vmax in knots)")
    print(f"  • Wind speed range:       {df['wind_speed'].min():.1f} kt to {df['wind_speed'].max():.1f} kt (mean: {df['wind_speed'].mean():.1f} kt, median: {df['wind_speed'].median():.1f} kt)")
    if "pressure" in df.columns:
        valid_pres = df["pressure"].dropna()
        if len(valid_pres) > 0:
            print(f"  • Pressure range:         {valid_pres.min():.1f} mb to {valid_pres.max():.1f} mb")

    # Years covered
    if "year" in df.columns and (df["year"] > 0).any():
        valid_years = df[df["year"] > 0]["year"]
        print(f"  • Years covered:          {valid_years.min()} – {valid_years.max()}")

    # Regional breakdown
    print("\nRegional Distribution:")
    for region, count in df["region"].value_counts().items():
        n_cyclones = df[df["region"] == region]["cyclone_id"].nunique()
        print(f"  • Region {region:6s}: {count:6,d} frames ({count/len(df)*100:5.1f}%) | {n_cyclones:4d} unique cyclones")

    # Cyclones summary
    frames_per_cyclone = df.groupby("cyclone_id")["sample_index"].count()
    print(f"\nFrames per Cyclone Distribution:")
    print(f"  • Min frames/cyclone:    {frames_per_cyclone.min()}")
    print(f"  • Max frames/cyclone:    {frames_per_cyclone.max()}")
    print(f"  • Mean frames/cyclone:   {frames_per_cyclone.mean():.1f}")
    print(f"  • Median frames/cyclone: {frames_per_cyclone.median():.1f}")

    print("=" * 70)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Inspect TCIR HDF5 dataset metadata and statistics.")
    parser.add_argument("--key", type=str, default="CPAC_IO_SH", help="TCIR archive key (e.g. CPAC_IO_SH, ATLN_EPAC_WPAC)")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw HDF5 files")
    args = parser.parse_args()

    h5_path = download_tcir_archive(key=args.key, destination_dir=args.raw_dir, extract=True)
    inspect_hdf5(h5_path)


if __name__ == "__main__":
    main()
