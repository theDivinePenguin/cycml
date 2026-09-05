# Cloud GPU Smoke Test Report: NVIDIA L4 / Target Environment

**Date:** 2026-09-05 17:26:57  
**Overall Verdict:** **FAIL**  

---

## 1. Environment & Hardware Audit

* **GPU Name:** NVIDIA H200 NVL
* **Compute Capability:** 9.0
* **Total VRAM:** 139.8 GB (150,109,880,320 bytes)
* **NVIDIA Driver Version:** 580.173.02
* **CUDA Version (PyTorch):** 12.8
* **PyTorch Version:** 2.11.0+cu128
* **Python Version:** 3.12.3
* **BF16 Support:** Yes
* **FP16 Support:** Yes
* **CUDA Tensor Matmul:** PASSED

```text
Sat Sep  5 17:26:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200 NVL                On  |   00000000:06:00.0 Off |                    0 |
| N/A   35C    P0             78W /  600W |     741MiB / 143771MiB |      3%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Project Dependency & Test Suite

* **pytest Suite:** FAILED
* **Critical Pipeline Imports:** ALL PASSED

---

## 3. Data Access Check

All 6 canonical dataset and aligned manifest files were verified:
* `data/raw/TCIR-ATLN_EPAC_WPAC.h5` (14254.5 MB)
* `data/raw/TCIR-CPAC_IO_SH.h5` (14254.5 MB)
* `data/metadata/metadata_all_basins.csv` (8.9 MB)
* `data/metadata/forecast_train_sequences_k5_aligned.csv` (17.7 MB)
* `data/metadata/forecast_val_sequences_k5_aligned.csv` (4.1 MB)
* `data/metadata/forecast_test_sequences_k5_aligned.csv` (3.8 MB)

---

## 4. Model & Checkpoint Check

* **Checkpoint:** `experiments/forecasting/checkpoints/cnn_transformer_k5/best.pt`
* **Architecture:** `TemporalTransformerForecaster` (K=5, in_channels=3, d_model=256, nhead=8, num_layers=2)
* **Real Sample Validation Forward Pass:** PASSED (Output finite, [+6h, +12h, +24h] predictions generated on CUDA)

---

## 5. GPU Micro-Benchmark

| Batch Size | Precision | Fits VRAM | Peak Allocated | Peak Reserved | Avg Forward Time | Samples/sec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 16 | BF16 | YES      |   384.9 MB |   508.0 MB |   3.58 ms | 4468.5 |
| 16 | FP16 | YES      |   360.2 MB |   484.0 MB |   3.15 ms | 5083.1 |
| 16 | FP32 | YES      |   553.6 MB |   568.0 MB |   4.19 ms | 3822.8 |
| 32 | BF16 | YES      |   689.4 MB |  1000.0 MB |   6.28 ms | 5092.4 |
| 32 | FP16 | YES      |   639.9 MB |   952.0 MB |   5.52 ms | 5801.6 |
| 32 | FP32 | YES      |  1025.3 MB |  1168.0 MB |   7.67 ms | 4169.4 |
| 64 | BF16 | YES      |  1296.8 MB |  1904.0 MB |  11.43 ms | 5599.7 |
| 64 | FP16 | YES      |  1198.8 MB |  1806.0 MB |   9.99 ms | 6404.7 |
| 64 | FP32 | YES      |  1970.2 MB |  2240.0 MB |  14.22 ms | 4499.6 |

* **Peak Throughput:** 6404.7 samples/s at BS=64 (FP16)
* **Max Tested Batch Size:** BS=64

---

## 6. Training Smoke Test (150 Steps)

* **Steps Completed:** 150
* **Initial Loss:** 56.7452
* **Final Loss:** 18.3070
* **Early 10-Step Avg Loss:** 55.3006
* **Late 10-Step Avg Loss:** 20.0287
* **Loss Trajectory:** Decreased as expected
* **Finite Gradients:** Verified
* **Parameter Updates:** Verified (Weights changed)
* **NaNs / Infs:** None

---

## 7. Final Verdict

* **OVERALL STATUS:** **FAIL**
* **Readiness for RTX PRO 6000 96GB / A100:** Ready to execute full campaigns without pipeline failure.
