"""Metadata extraction, parsing, and normalization for TCIR."""
from pathlib import Path
from typing import List, Optional, Tuple
import h5py
import numpy as np
import pandas as pd


def infer_region_from_storm_id(storm_id: str, lat: float, lon: float) -> str:
    """Infer oceanic basin/region from storm ID and coordinates.

    TCIR storm IDs typically follow the international convention:
    - S, P, U: Southern Hemisphere (SH / South Pacific / South Indian)
    - A, B: North Indian Ocean (IO - Arabian Sea / Bay of Bengal)
    - C: Central Pacific (CPAC)
    - E: Eastern North Pacific (EPAC)
    - W: Western North Pacific (WPAC)
    - L: North Atlantic (ATLN)
    """
    sid = str(storm_id).strip().upper()
    if sid.endswith(("S", "P", "U")) or lat < 0:
        return "SH"
    elif sid.endswith(("A", "B")) or (0 <= lat <= 35 and 45 <= lon <= 100):
        return "IO"
    elif sid.endswith("C") or (0 <= lat <= 40 and 180 <= lon <= 220):
        return "CPAC"
    elif sid.endswith("E") or (0 <= lat <= 40 and 220 <= lon <= 280):
        return "EPAC"
    elif sid.endswith("W") or (0 <= lat <= 45 and 100 <= lon <= 180):
        return "WPAC"
    elif sid.endswith("L") or (0 <= lat <= 60 and (lon > 280 or lon < 0)):
        return "ATLN"
    return "UNKNOWN"


def load_tcir_info_table(h5_path: str | Path) -> pd.DataFrame:
    """Load the metadata table 'info' from a TCIR HDF5 file.

    Supports direct PyTables block reconstruction to ensure compatibility
    across all Pandas/PyTables versions.

    Args:
        h5_path: Path to HDF5 file.

    Returns:
        pandas DataFrame containing the original info table.
    """
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"TCIR HDF5 file not found at: {path}")

    import pickle

    with h5py.File(path, "r") as hf:
        if "info" not in hf:
            raise KeyError(f"Key 'info' not found in HDF5 file {path}")
        info = hf["info"]

        # Check if PyTables block format
        if "block0_items" in info and "block0_values" in info:
            data_dict = {}
            # Float block
            b0_items = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in info["block0_items"][:]]
            b0_values = info["block0_values"][:]
            for i, col in enumerate(b0_items):
                data_dict[col] = b0_values[:, i]

            # String/Object block (data_set, ID, time)
            if "block1_items" in info and "block1_values" in info:
                b1_items = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in info["block1_items"][:]]
                b1_raw = info["block1_values"][0]
                b1_data = pickle.loads(b1_raw.tobytes() if hasattr(b1_raw, "tobytes") else bytes(b1_raw))
                for i, col in enumerate(b1_items):
                    data_dict[col] = b1_data[:, i]

            df = pd.DataFrame(data_dict)
            return df
        else:
            # Fallback simple group read
            data_dict = {k: info[k][:] for k in info.keys()}
            return pd.DataFrame(data_dict)


def parse_and_normalize_metadata(
    h5_path: str | Path,
    target_regions: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """Parse, clean, and normalize TCIR metadata.

    Args:
        h5_path: Path to raw HDF5 file.
        target_regions: List of target regions to filter by (e.g. ['CPAC', 'IO', 'SH']).
        save_path: Optional destination CSV path.

    Returns:
        Cleaned, normalized pandas DataFrame.
    """
    df_raw = load_tcir_info_table(h5_path)
    
    # Ensure sample_index represents the exact row in HDF5 matrix
    df_raw = df_raw.reset_index(drop=True)
    df_raw["sample_index"] = df_raw.index

    # Standardize column names
    col_map = {
        "ID": "cyclone_id",
        "id": "cyclone_id",
        "time": "timestamp",
        "lat": "latitude",
        "Latitude": "latitude",
        "lon": "longitude",
        "Longitude": "longitude",
        "Vmax": "wind_speed",
        "vmax": "wind_speed",
        "MSLP": "pressure",
        "mslp": "pressure",
        "pres": "pressure",
        "data_set": "source_dataset",
        "R35_4qAVG": "size_radius"
    }
    df = df_raw.rename(columns=col_map).copy()

    # Decode bytes to string if needed
    for col in ["cyclone_id", "timestamp"]:
        if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], (bytes, bytearray)):
            df[col] = df[col].str.decode("utf-8")
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Extract year from timestamp (format YYYYMMDDHH or YYYYMMDD)
    if "timestamp" in df.columns:
        df["year"] = df["timestamp"].apply(lambda t: int(str(t)[:4]) if len(str(t)) >= 4 and str(t)[:4].isdigit() else -1)

    # Infer regions
    regions = []
    for _, row in df.iterrows():
        r = infer_region_from_storm_id(
            storm_id=row.get("cyclone_id", ""),
            lat=float(row.get("latitude", 0.0)),
            lon=float(row.get("longitude", 0.0))
        )
        regions.append(r)
    df["region"] = regions

    # Data Quality Validation: Remove NaN or non-positive wind speeds
    initial_count = len(df)
    df = df.dropna(subset=["wind_speed", "cyclone_id"]).copy()
    df = df[df["wind_speed"] > 0].copy()
    valid_count = len(df)

    if valid_count < initial_count:
        print(f"[Metadata] Removed {initial_count - valid_count} invalid/missing samples. Remaining: {valid_count}")

    # Regional filtering if specified
    if target_regions:
        target_set = set(r.upper() for r in target_regions)
        df = df[df["region"].isin(target_set)].copy()
        print(f"[Metadata] Filtered for regions {target_regions}: {len(df)} samples remaining across {df['cyclone_id'].nunique()} cyclones.")

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        print(f"[Metadata] Saved normalized metadata to: {out_p}")

    return df


def build_unified_multi_hdf5_metadata(
    h5_paths: List[str | Path],
    save_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """Load, parse, normalize, and concatenate metadata from multiple TCIR HDF5 archives.

    Each row is annotated with:
    - 'h5_file': absolute or relative path to the specific source HDF5 archive
    - 'h5_row_index': original row index within that specific HDF5 matrix
    - 'sample_index': unique sequential global index (0 to N_total - 1)

    Args:
        h5_paths: List of paths to HDF5 archives (e.g. [TCIR-CPAC_IO_SH.h5, TCIR-ATLN_EPAC_WPAC.h5])
        save_path: Destination path for normalized metadata CSV.

    Returns:
        Unified pandas DataFrame containing all combined observations.
    """
    dfs = []
    global_idx = 0

    for path_item in h5_paths:
        p = Path(path_item).resolve()
        if not p.exists():
            raise FileNotFoundError(f"HDF5 file does not exist: {p}")

        print(f"[Metadata Builder] Parsing {p.name}...")
        df_single = parse_and_normalize_metadata(p)
        df_single["h5_file"] = str(p)
        df_single["h5_row_index"] = df_single["sample_index"]
        dfs.append(df_single)

    df_unified = pd.concat(dfs, ignore_index=True)
    df_unified["sample_index"] = df_unified.index

    print(f"[Metadata Builder] Unified metadata complete: {len(df_unified):,} frames across {df_unified['cyclone_id'].nunique()} cyclones.")
    print(f"[Metadata Builder] Regional breakdown:")
    for region, count in df_unified["region"].value_counts().items():
        print(f"  • {region:<6}: {count:,d} frames ({count / len(df_unified) * 100:.1f}%)")

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_unified.to_csv(out_p, index=False)
        print(f"[Metadata Builder] Saved unified metadata to: {out_p}")

    return df_unified
