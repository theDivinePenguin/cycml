#!/usr/bin/env python3
"""Hardware & software environment verification script for NVIDIA A100 80GB GPU.

Validates:
  1. PyTorch version >= 2.0
  2. CUDA availability & runtime version
  3. GPU identity and SM architecture (A100 requires sm_80)
  4. VRAM capacity (>= 75 GiB for A100 80GB; warns gracefully if run locally)
  5. cuDNN availability and version
  6. AMP (Automatic Mixed Precision) functionality
  7. BF16 hardware support (native on Ampere A100)
  8. FP16 tensor core support
  9. TF32 tensor core configuration
  10. Multi-worker DataLoader functionality with pinned memory
  11. End-to-end forward and backward passes with gradient update
"""
import sys
import os
from pathlib import Path
import tempfile
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def log_step(name: str, status: str, details: str = ""):
    color = "\033[92m" if status == "PASS" else ("\033[93m" if status == "WARN" else "\033[91m")
    reset = "\033[0m"
    detail_str = f" - {details}" if details else ""
    print(f"[{color}{status:4s}{reset}] {name}{detail_str}")


def run_verification(strict_a100: bool = False) -> bool:
    print("=" * 80)
    print("NVIDIA A100 80GB ENVIRONMENT VERIFICATION")
    print(f"Mode: {'STRICT A100 ENFORCEMENT' if strict_a100 else 'HARDWARE DISCOVERY & SANITY'}")
    print("=" * 80)

    all_passed = True
    warnings = []

    # 1. PyTorch Version
    pt_ver = torch.__version__
    pt_major = int(pt_ver.split(".")[0])
    if pt_major >= 2:
        log_step("PyTorch Version", "PASS", f"Detected {pt_ver}")
    else:
        log_step("PyTorch Version", "FAIL", f"Found {pt_ver}, expected PyTorch >= 2.0")
        all_passed = False

    # 2. CUDA Availability
    cuda_avail = torch.cuda.is_available()
    if not cuda_avail:
        log_step("CUDA Availability", "FAIL", "torch.cuda.is_available() is False. No CUDA runtime detected.")
        return False
    
    cuda_ver = torch.version.cuda
    log_step("CUDA Availability", "PASS", f"CUDA available, runtime version: {cuda_ver}")

    # 3. GPU Identification & Compute Capability
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    compute_cap = f"sm_{major}{minor}"

    is_a100 = "A100" in device_name.upper()
    is_ampere = (major == 8 and minor == 0)

    if is_a100 and is_ampere:
        log_step("GPU Architecture", "PASS", f"{device_name} (Capability: {compute_cap})")
    elif major >= 8:
        status = "WARN" if not strict_a100 else "FAIL"
        log_step("GPU Architecture", status, f"{device_name} (Capability: {compute_cap} >= sm_80 compatible)")
        if strict_a100:
            all_passed = False
        else:
            warnings.append(f"Running on non-A100 GPU: {device_name}")
    else:
        status = "WARN" if not strict_a100 else "FAIL"
        log_step("GPU Architecture", status, f"{device_name} (Capability: {compute_cap})")
        if strict_a100:
            all_passed = False
        else:
            warnings.append(f"GPU {device_name} compute capability is {compute_cap} (< sm_80)")

    # 4. VRAM Capacity
    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    if total_vram_gb >= 75.0:
        log_step("VRAM Capacity", "PASS", f"{total_vram_gb:.1f} GiB (A100 80GB class)")
    else:
        status = "WARN" if not strict_a100 else "FAIL"
        log_step("VRAM Capacity", status, f"{total_vram_gb:.1f} GiB (< 75 GiB requirement for A100 80GB)")
        if strict_a100:
            all_passed = False
        else:
            warnings.append(f"VRAM is {total_vram_gb:.1f} GiB (local dev GPU)")

    # 5. cuDNN
    cudnn_avail = torch.backends.cudnn.is_available()
    if cudnn_avail:
        cudnn_ver = torch.backends.cudnn.version()
        log_step("cuDNN", "PASS", f"Available, version {cudnn_ver}")
    else:
        log_step("cuDNN", "FAIL", "cuDNN is not available")
        all_passed = False

    # 6. BF16 Support
    bf16_supported = torch.cuda.is_bf16_supported()
    if bf16_supported:
        log_step("BF16 Hardware Support", "PASS", "Native bfloat16 supported by hardware")
    else:
        status = "WARN" if not strict_a100 else "FAIL"
        log_step("BF16 Hardware Support", status, "bfloat16 not natively supported by this GPU")
        if strict_a100:
            all_passed = False

    # 7. TF32 Support
    tf32_supported = torch.cuda.get_device_capability()[0] >= 8
    if tf32_supported:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_step("TF32 Tensor Cores", "PASS", "Ampere TF32 enabled for matmul and cuDNN")
    else:
        log_step("TF32 Tensor Cores", "WARN", "TF32 requires compute capability >= 8.0")

    # 8. FP16 AMP Sanity
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            a = torch.randn(64, 64, device="cuda")
            b = torch.randn(64, 64, device="cuda")
            c = torch.matmul(a, b)
        assert c.dtype == torch.float16
        log_step("FP16 AMP", "PASS", "Autocast FP16 matmul verified")
    except Exception as e:
        log_step("FP16 AMP", "FAIL", str(e))
        all_passed = False

    # 9. BF16 AMP Sanity
    if bf16_supported:
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
                c = torch.matmul(a, b)
            assert c.dtype == torch.bfloat16
            log_step("BF16 AMP", "PASS", "Autocast BF16 matmul verified")
        except Exception as e:
            log_step("BF16 AMP", "FAIL", str(e))
            all_passed = False

    # 10. Multi-Worker DataLoader Check
    try:
        x_dummy = torch.randn(256, 16)
        y_dummy = torch.randn(256, 1)
        ds = TensorDataset(x_dummy, y_dummy)
        loader = DataLoader(
            ds,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
        batch_count = 0
        for bx, by in loader:
            bx_gpu = bx.to("cuda", non_blocking=True)
            by_gpu = by.to("cuda", non_blocking=True)
            batch_count += 1
        assert batch_count == 8
        log_step("DataLoader Pipeline", "PASS", "Multi-worker pinned memory iteration verified")
    except Exception as e:
        log_step("DataLoader Pipeline", "FAIL", str(e))
        all_passed = False

    # 11. End-to-End Forward + Backward + Optimizer Pass
    try:
        model = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        ).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        inputs = torch.randn(32, 64, device="cuda")
        targets = torch.randn(32, 1, device="cuda")

        optimizer.zero_grad()
        use_dtype = torch.bfloat16 if bf16_supported else torch.float16
        with torch.amp.autocast("cuda", dtype=use_dtype):
            preds = model(inputs)
            loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss).item()
        log_step("Model Forward/Backward", "PASS", f"End-to-end gradient pass verified ({use_dtype})")
    except Exception as e:
        log_step("Model Forward/Backward", "FAIL", str(e))
        all_passed = False

    # 12. Checkpoint Disk I/O
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
            torch.save({"dummy_weights": model.state_dict()}, tmp.name)
            loaded = torch.load(tmp.name, map_location="cpu")
            assert "dummy_weights" in loaded
        log_step("Checkpoint I/O", "PASS", "Atomic checkpoint save and reload verified")
    except Exception as e:
        log_step("Checkpoint I/O", "FAIL", str(e))
        all_passed = False

    print("-" * 80)
    if all_passed:
        if warnings:
            print("\033[93mVERIFICATION PASSED WITH WARNINGS (Local/Dev GPU detected):\033[0m")
            for w in warnings:
                print(f"  • {w}")
            print("To strictly enforce A100 80GB, run: python scripts/verify_a100.py --strict")
        else:
            print("\033[92m[VERIFICATION SUCCESS] NVIDIA A100 80GB fully verified and experiment-ready!\033[0m")
        return True
    else:
        print("\033[91m[VERIFICATION FAILED] Hardware/software environment is incompatible.\033[0m")
        return False


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    success = run_verification(strict_a100=strict)
    sys.exit(0 if success else 1)
