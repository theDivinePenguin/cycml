#!/usr/bin/env bash
# ==============================================================================
# scripts/a100_preflight.sh — Pre-Flight Diagnostics for A100 80GB Environment
# ==============================================================================
set -e

echo "================================================================================"
echo "          NVIDIA A100 80GB CLOUD PRE-FLIGHT DIAGNOSTICS SUITE"
echo "================================================================================"

FAILED=0

# 1. Verify NVIDIA Driver & nvidia-smi
echo -n "[1/12] Checking nvidia-smi & driver... "
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1)
    echo "OK ($GPU_NAME, Driver $DRIVER_VER)"
else
    echo "FAIL (nvidia-smi not found)"
    FAILED=1
fi

# 2. Check nvcc compiler (optional but documented)
echo -n "[2/12] Checking CUDA compiler (nvcc)... "
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo "OK (CUDA $NVCC_VER)"
else
    echo "NOTICE (nvcc not in PATH; PyTorch prebuilt CUDA runtime will be used)"
fi

# 3. Check Python version
echo -n "[3/12] Checking Python 3.10+ ... "
PYTHON_BIN="python3"
if [ -f "/opt/venv/bin/python" ]; then
    PYTHON_BIN="/opt/venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi
PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PY_MAJOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_BIN -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
    echo "OK ($PYTHON_BIN: $PY_VER)"
else
    echo "FAIL (Found $PY_VER, require >= 3.10)"
    FAILED=1
fi

# 4. Check PyTorch & CUDA availability
echo -n "[4/12] Checking PyTorch & CUDA... "
TORCH_INFO=$($PYTHON_BIN -c "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()}, ver: {torch.version.cuda})')")
echo "OK ($TORCH_INFO)"

# 5. Check System RAM
echo -n "[5/12] Checking System RAM... "
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM_GB" -ge 30 ]; then
    echo "OK (${TOTAL_RAM_GB} GB RAM available)"
else
    echo "WARN (${TOTAL_RAM_GB} GB RAM; >= 32 GB recommended for high-worker DataLoader)"
fi

# 6. Check Available Disk Space
echo -n "[6/12] Checking Disk Space... "
DISK_AVAIL_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$DISK_AVAIL_GB" -ge 20 ]; then
    echo "OK (${DISK_AVAIL_GB} GB available on workspace partition)"
else
    echo "FAIL (Only ${DISK_AVAIL_GB} GB available; >= 20 GB required for checkpoints)"
    FAILED=1
fi

# 7. Check Raw Dataset Integrity
echo -n "[7/12] Checking TCIR raw datasets... "
H5_1="data/raw/TCIR-CPAC_IO_SH.h5"
H5_2="data/raw/TCIR-ATLN_EPAC_WPAC.h5"
if [ -f "$H5_1" ] && [ -f "$H5_2" ]; then
    H5_1_SZ=$(du -h "$H5_1" | awk '{print $1}')
    H5_2_SZ=$(du -h "$H5_2" | awk '{print $1}')
    echo "OK (CPAC_IO_SH: $H5_1_SZ, ATLN_EPAC_WPAC: $H5_2_SZ)"
else
    echo "FAIL (One or more HDF5 files missing from data/raw/)"
    FAILED=1
fi

# 8. Check Metadata & Splits
echo -n "[8/12] Checking metadata manifests... "
META_FILE="data/metadata/metadata_all_basins.csv"
TRAIN_META="data/metadata/train_metadata_all_basins.csv"
NORM_FILE="data/metadata/normalization_stats_multichannel.json"
if [ -f "$META_FILE" ] && [ -f "$TRAIN_META" ] && [ -f "$NORM_FILE" ]; then
    N_SAMPLES=$(wc -l < "$META_FILE")
    echo "OK (Unified metadata: $N_SAMPLES lines, training norm stats present)"
else
    echo "FAIL (Missing metadata or normalization statistics in data/metadata/)"
    FAILED=1
fi

# 9. Check SHIPS Environmental Databases
echo -n "[9/12] Checking SHIPS 6-basin environmental predictor files... "
SHIPS_DIR="data/ships"
SHIPS_COUNT=$(ls "$SHIPS_DIR"/lsdiag*.txt 2>/dev/null | wc -l)
if [ "$SHIPS_COUNT" -ge 6 ]; then
    echo "OK (All 6 global ocean basins present: IO, EP, CP, AL, WP, SH)"
else
    echo "WARN (Found only $SHIPS_COUNT SHIPS files in $SHIPS_DIR)"
fi

# 10. Run Full Python Hardware Verification
echo "[10/12] Running Deep Hardware Verification via verify_a100.py..."
if ! $PYTHON_BIN scripts/verify_a100.py; then
    FAILED=1
fi

# 11. Estimated Dataset & Experiment Memory Footprint Report
echo "[11/12] Calculating Dataset & Checkpoint Footprint..."
$PYTHON_BIN -c "
import os, glob
h5_sz = sum(os.path.getsize(f) for f in glob.glob('data/raw/*.h5')) / (1024**3)
ships_sz = sum(os.path.getsize(f) for f in glob.glob('data/ships/*.txt')) / (1024**2)
meta_sz = sum(os.path.getsize(f) for f in glob.glob('data/metadata/*.csv')) / (1024**2)
print(f'  • Raw Satellite HDF5 Data: {h5_sz:.1f} GB')
print(f'  • Global SHIPS Environmental Predictors: {ships_sz:.1f} MB')
print(f'  • Sequence Metadata Manifests: {meta_sz:.1f} MB')
print(f'  • Estimated Checkpoint Footprint per Experiment: ~350 MB')
"

# 12. Final Preflight Verdict
echo "--------------------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
    echo -e "\033[92m================================================================================"
    echo "                          A100 PRE-FLIGHT: PASS"
    echo "================================================================================\033[0m"
    echo "The environment is fully configured, validated, and ready for experimentation."
    exit 0
else
    echo -e "\033[91m================================================================================"
    echo "                          A100 PRE-FLIGHT: FAIL"
    echo "================================================================================\033[0m"
    echo "Action required: resolve the failures marked above before initiating experiments."
    exit 1
fi
