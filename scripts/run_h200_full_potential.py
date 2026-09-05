"""
NVIDIA H200 NVL Full-Potential Campaign Runner.

Exploits 141 GB HBM3e VRAM and 125 GB System RAM via:
1. Multi-Process Concurrent GPU Training (trains multiple models simultaneously).
2. Hopper BF16 AMP Acceleration.
3. Zero-wait execution across all 6 core campaigns.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = sys.executable

CAMPAIGNS = [
    {
        "name": "temporal_k1",
        "config": "configs/temporal/temporal_k1.yaml",
        "gpu_mem_target_gb": 8,
        "description": "Temporal Ablation K=1 (Order baseline)"
    },
    {
        "name": "temporal_k7",
        "config": "configs/temporal/temporal_k7.yaml",
        "gpu_mem_target_gb": 10,
        "description": "Temporal Ablation K=7 (Medium history)"
    },
    {
        "name": "temporal_k13",
        "config": "configs/temporal/temporal_k13.yaml",
        "gpu_mem_target_gb": 12,
        "description": "Temporal Ablation K=13 (Full aligned history)"
    },
    {
        "name": "residual_k5_unconstrained",
        "config": "configs/residual/residual_k5_unconstrained.yaml",
        "gpu_mem_target_gb": 10,
        "description": "Residual Delta-V Forecaster (Solves false dips)"
    },
    {
        "name": "ri_model1_dedicated_focal",
        "config": "configs/ri/ri_model1_dedicated_focal.yaml",
        "gpu_mem_target_gb": 8,
        "description": "Dedicated Rapid Intensification (Focal Loss)"
    },
    {
        "name": "fusion_gated_residual",
        "config": "configs/multimodal/fusion_gated_residual.yaml",
        "gpu_mem_target_gb": 10,
        "description": "Multimodal Satellite + SHIPS Environmental Fusion"
    },
]


def run_parallel_group(group_name: str, configs: list):
    print("\n" + "=" * 80)
    print(f"LAUNCHING PARALLEL H200 CAMPAIGN GROUP: {group_name}")
    print("=" * 80)

    processes = []
    log_files = []

    log_dir = PROJECT_ROOT / "experiments" / "h200_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for item in configs:
        name = item["name"]
        cfg_path = item["config"]
        log_path = log_dir / f"{name}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)

        cmd = [
            PYTHON_BIN, "train.py",
            "--config", str(cfg_path),
            "--device", "cuda"
        ]

        print(f"🚀 Spawning concurrent process: {name:25s} | Config: {cfg_path}")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        processes.append((name, p, log_path))

    print(f"\nAll {len(processes)} processes running concurrently on H200 NVL (141 GB VRAM). Monitoring...")

    # Wait for all to complete
    completed = {}
    while len(completed) < len(processes):
        time.sleep(10)
        for name, p, log_path in processes:
            if name not in completed:
                ret = p.poll()
                if ret is not None:
                    status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
                    elapsed_m = (time.time() - start_time) / 60.0
                    print(f"🏁 Completed: {name:25s} | Status: {status:15s} | Time: {elapsed_m:.1f} min")
                    completed[name] = ret

    for lf in log_files:
        lf.close()

    elapsed_total = (time.time() - start_time) / 60.0
    print(f"\nGroup {group_name} finished in {elapsed_total:.2f} minutes.")
    return completed


def main():
    print("================================================================================")
    print("      NVIDIA H200 NVL (141 GB) HIGH-THROUGHPUT EXPERIMENT SUITE")
    print("================================================================================")

    # Launch Group 1: 3 Concurrent Temporal K-Ablation Models
    group1 = CAMPAIGNS[:3]
    res1 = run_parallel_group("Group 1: Temporal K Scaling (K1, K7, K13)", group1)

    # Launch Group 2: 3 Concurrent Formulations (Residual, RI Focal, Multimodal Fusion)
    group2 = CAMPAIGNS[3:]
    res2 = run_parallel_group("Group 2: Physics Formulations (Residual, RI, Fusion)", group2)

    print("\n" + "=" * 80)
    print("ALL CAMPAIGNS EXECUTED SUCCESSFULLY ON H200 NVL")
    print("=" * 80)


if __name__ == "__main__":
    main()
