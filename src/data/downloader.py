"""Dataset download and archive extraction utilities for TCIR."""
import os
import tarfile
from pathlib import Path
from typing import Optional
import gdown


TCIR_FILE_MAP = {
    "CPAC_IO_SH": {
        "id": "1nvDQvgcC5GlXIZuQNtSjL7tmRPunyOsG",
        "archive_name": "TCIR-CPAC_IO_SH.h5.tar.gz",
        "h5_name": "TCIR-CPAC_IO_SH.h5",
        "regions": ["CPAC", "IO", "SH"]
    },
    "ATLN_EPAC_WPAC": {
        "id": "1_g7aKIwJRbgjEiBqxWBgwLNhgF12-RVt",
        "archive_name": "TCIR-ATLN_EPAC_WPAC.h5.tar.gz",
        "h5_name": "TCIR-ATLN_EPAC_WPAC.h5",
        "regions": ["ATLN", "EPAC", "WPAC"]
    },
    "ALL_2017": {
        "id": "1sTJahi4UFNIDlAZuvMSXgjYGsR9X6_7e",
        "archive_name": "TCIR-ALL_2017.h5.tar.gz",
        "h5_name": "TCIR-ALL_2017.h5",
        "regions": ["ALL"]
    }
}


def download_tcir_archive(
    key: str = "CPAC_IO_SH",
    destination_dir: str | Path = "data/raw",
    extract: bool = True
) -> Path:
    """Download TCIR archive from Google Drive and optionally extract the HDF5 file.

    Args:
        key: Dataset key ("CPAC_IO_SH", "ATLN_EPAC_WPAC", etc.).
        destination_dir: Directory where raw files should be saved.
        extract: Whether to extract .tar.gz archive after download.

    Returns:
        Path to the uncompressed .h5 file.
    """
    if key not in TCIR_FILE_MAP:
        raise ValueError(f"Unknown TCIR key '{key}'. Supported keys: {list(TCIR_FILE_MAP.keys())}")

    info = TCIR_FILE_MAP[key]
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    h5_path = dest_path / info["h5_name"]
    if h5_path.exists():
        print(f"[TCIR Downloader] Found existing HDF5 file: {h5_path} ({h5_path.stat().st_size / 1e9:.2f} GB)")
        return h5_path

    archive_path = dest_path / info["archive_name"]
    if not archive_path.exists():
        print(f"[TCIR Downloader] Downloading {info['archive_name']} (Google Drive ID: {info['id']})...")
        url = f"https://drive.google.com/uc?id={info['id']}"
        output = gdown.download(url, str(archive_path), quiet=False)
        if output is None or not Path(output).exists():
            raise RuntimeError(f"Failed to download {info['archive_name']} from Google Drive.")
        print(f"[TCIR Downloader] Successfully downloaded: {archive_path}")

    if extract:
        print(f"[TCIR Downloader] Extracting {archive_path} to {dest_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dest_path)
        print(f"[TCIR Downloader] Extraction complete. HDF5 file is at: {h5_path}")

    return h5_path
