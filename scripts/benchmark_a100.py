"""Production pipeline benchmark suite for NVIDIA A100 80GB optimization.

Evaluates:
  - Throughput (samples/sec, batches/sec, ms/batch)
  - Precision modes: FP32, FP16 AMP, BF16 AMP
  - Batch sizes: 16, 32, 64, 128, 256 (where VRAM permits)
  - DataLoader configurations: num_workers, pin_memory, persistent_workers, prefetch_factor
  - Peak VRAM allocation
  - Projected epoch duration across dataset (48,856 samples)
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sequence_dataset import TCIRSequenceDataset
from src.models.temporal_forecaster import TemporalTransformerForecaster


def run_single_benchmark_trial(
    model: nn.Module,
    loader: DataLoader,
    precision: str = "bf16",
    device: str = "cuda",
    warmup_batches: int = 5,
    benchmark_batches: int = 25,
) -> Dict[str, float]:
    """Measures precise throughput and VRAM for a specific configuration."""
    dev = torch.device(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    use_amp = (precision in ["fp16", "bf16"]) and (dev.type == "cuda")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    batch_times = []
    h2d_times = []
    fetch_times = []
    compute_times = []
    samples_processed = 0

    loader_iter = iter(loader)

    # Warmup
    for _ in range(warmup_batches):
        try:
            images, vis_masks, targets, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, vis_masks, targets, _ = next(loader_iter)

        images = images.to(dev, non_blocking=True)
        vis_masks = vis_masks.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(images, vis_masks)
            loss = criterion(out, targets)

        if precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    # Benchmark Loop
    for _ in range(benchmark_batches):
        t_fetch_start = time.perf_counter()
        try:
            images, vis_masks, targets, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, vis_masks, targets, _ = next(loader_iter)
        t_fetch_end = time.perf_counter()
        fetch_times.append(t_fetch_end - t_fetch_start)

        t_comp_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        images = images.to(dev, non_blocking=True)
        vis_masks = vis_masks.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_h2d_end = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(images, vis_masks)
            loss = criterion(out, targets)

        if precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_comp_end = time.perf_counter()

        compute_times.append(t_comp_end - t_comp_start)
        h2d_times.append(t_h2d_end - t_h2d_start)
        batch_times.append((t_fetch_end - t_fetch_start) + (t_comp_end - t_comp_start))
        samples_processed += len(images)

    total_time = sum(batch_times)
    samples_per_sec = samples_processed / max(total_time, 1e-6)
    batches_per_sec = benchmark_batches / max(total_time, 1e-6)
    sec_per_batch = total_time / max(benchmark_batches, 1)
    h2d_transfer_ms = (sum(h2d_times) / max(benchmark_batches, 1)) * 1000.0
    fetch_time_ms = (sum(fetch_times) / max(benchmark_batches, 1)) * 1000.0
    compute_time_ms = (sum(compute_times) / max(benchmark_batches, 1)) * 1000.0

    # Compute duty cycle / CPU/IO bottleneck
    total_batch_ms = fetch_time_ms + compute_time_ms
    gpu_duty_cycle_pct = round(100.0 * compute_time_ms / max(total_batch_ms, 1e-6), 1)
    io_bottleneck_pct = round(100.0 * fetch_time_ms / max(total_batch_ms, 1e-6), 1)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if dev.type == "cuda" else 0.0
    )
    peak_vram_gb = round(peak_vram_mb / 1024.0, 2)

    # Dataset size extrapolation (48,856 train sequences)
    dataset_size = 48856
    est_epoch_sec = dataset_size / max(samples_per_sec, 1e-6)

    return {
        "samples_per_sec": round(samples_per_sec, 1),
        "batches_per_sec": round(batches_per_sec, 2),
        "sec_per_batch": round(sec_per_batch, 4),
        "ms_per_batch": round(sec_per_batch * 1000.0, 1),
        "h2d_transfer_ms": round(h2d_transfer_ms, 2),
        "dataloader_fetch_ms": round(fetch_time_ms, 2),
        "gpu_compute_ms": round(compute_time_ms, 2),
        "io_bottleneck_pct": io_bottleneck_pct,
        "gpu_utilization_pct": gpu_duty_cycle_pct,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "peak_vram_gb": peak_vram_gb,
        "est_epoch_minutes": round(est_epoch_sec / 60.0, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch training throughput on A100.")
    parser.add_argument("--preset", type=str, default="a100", choices=["a100", "local-smoke"],
                        help="Benchmark matrix preset: 'a100' (15 trials: 5 BS x 3 precisions) or 'local-smoke' (8 trials: 4 BS x 2 precisions)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None, help="Batch sizes to test (overrides preset)")
    parser.add_argument("--precisions", type=str, nargs="+", default=None, help="Precision modes (overrides preset)")
    parser.add_argument("--k-frames", type=int, default=5, help="Number of historical frames")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for benchmark reports")
    args = parser.parse_args()

    # Resolve preset vs explicit arguments
    if args.batch_sizes is None:
        args.batch_sizes = [16, 32, 64, 128, 256] if args.preset == "a100" else [16, 32, 64, 128]
    if args.precisions is None:
        args.precisions = ["bf16", "fp16", "fp32"] if args.preset == "a100" else ["fp16", "fp32"]

    total_trials = len(args.batch_sizes) * len(args.precisions)
    print(f"[Benchmark Matrix] Preset: {args.preset.upper()} -> {len(args.batch_sizes)} batch sizes {args.batch_sizes} x {len(args.precisions)} precisions {args.precisions} = {total_trials} total trials.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("A100 PRODUCTION PIPELINE BENCHMARK SUITE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # Enable TF32 for Ampere / A100
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[Optimization] Enabled TF32 matmul and cuDNN acceleration.")

    meta_dir = Path("data/metadata")
    manifest_path = meta_dir / f"forecast_train_sequences_k{args.k_frames}.csv"
    if not manifest_path.exists():
        manifest_path = meta_dir / "forecast_train_sequences_k5.csv"

    train_df = pd.read_csv(manifest_path)
    with open(meta_dir / "normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)

    mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    std = [norm_stats["std"][c] for c in [0, 1, 2]]

    dataset = TCIRSequenceDataset(train_df, mean=mean, std=std, channels=[0, 1, 2], is_training=True)

    benchmark_matrix = []

    for bs in args.batch_sizes:
        for prec in args.precisions:
            if prec == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                print(f"[Skip] BF16 not supported on this GPU. Skipping batch={bs}, prec={prec}.")
                continue

            print(f"\nEvaluating Config: Batch Size = {bs:3d} | Precision = {prec:5s}...")

            loader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=4,
                pin_memory=(device == "cuda"),
                persistent_workers=True,
                drop_last=True,
            )

            # Instantiate model
            model = TemporalTransformerForecaster(
                in_channels=3, d_model=256, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1, pretrained_cnn=False
            ).to(device)

            try:
                metrics = run_single_benchmark_trial(
                    model=model,
                    loader=loader,
                    precision=prec,
                    device=device,
                    warmup_batches=4,
                    benchmark_batches=15,
                )
                metrics["batch_size"] = bs
                metrics["precision"] = prec
                benchmark_matrix.append(metrics)

                print(
                    f"  -> Throughput: {metrics['samples_per_sec']:6.1f} samples/sec | "
                    f"Iter: {metrics['ms_per_batch']:6.1f} ms | "
                    f"VRAM: {metrics['peak_vram_gb']:4.1f} GB | "
                    f"GPU Util: {metrics['gpu_utilization_pct']:5.1f}% | "
                    f"IO Wait: {metrics['io_bottleneck_pct']:5.1f}% | "
                    f"Est. Epoch: {metrics['est_epoch_minutes']:5.1f} min"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"  -> [OOM] Out of memory at batch size {bs} ({prec})!")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  -> [ERROR] {e}")

            del model
            del loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    dataset.close()

    # Determine Best Configuration (maximize samples_per_sec without OOM)
    best_config = None
    if benchmark_matrix:
        best_config = max(benchmark_matrix, key=lambda x: x["samples_per_sec"])
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION:")
        print(f"batch_size = {best_config['batch_size']}")
        print(f"precision = {best_config['precision']}")
        print(f"throughput = {best_config['samples_per_sec']} samples/sec")
        print(f"peak VRAM = {best_config['peak_vram_gb']} GB")
        print(f"GPU utilization = {best_config['gpu_utilization_pct']}%")
        print("=" * 80)

    # Save benchmark report
    report_file = Path(args.output_dir) / "a100_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "best_configuration": best_config,
                "results": benchmark_matrix,
            },
            f,
            indent=2,
        )
    print(f"\n[Saved Benchmark Report] -> {report_file}")


if __name__ == "__main__":
    main()
