"""Standalone evaluation engine with absolute test-set lock protection and stratified diagnostics."""
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.environmental import EnvironmentalFeatureManager
from src.data.sequence_dataset import TCIRSequenceDataset
from src.evaluation.classification_metrics import compute_ri_metrics
from src.evaluation.sanity_checks import PhysicalSanityChecker
from src.evaluation.stratified import evaluate_regime_stratified
from src.models.probabilistic import compute_probabilistic_metrics
from train import build_model_from_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate CycML models with strict test-set lock safeguards.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (if not embedded in checkpoint)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--eval-test", action="store_true", help="Explicit confirmation flag required to evaluate test split")
    parser.add_argument("--confirm-locked-test-eval", action="store_true", help="Secondary explicit lock unlock flag for test set")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # MANDATORY TEST-SET PROTECTION GUARD
    # -----------------------------------------------------------------------
    if args.split == "test":
        if not (args.eval_test and args.confirm_locked_test_eval):
            print("\033[91m" + "=" * 80)
            print("ACCESS DENIED: TEST SET IS LOCKED")
            print("=" * 80)
            print("Per experimental protocol, the test set must remain locked during development.")
            print("To run final post-hoc evaluation on the locked test set, you must pass BOTH:")
            print("    --eval-test --confirm-locked-test-eval")
            print("=" * 80 + "\033[0m")
            sys.exit(1)

    dev = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=dev)
    cfg = ckpt.get("config")
    if cfg is None:
        if args.config is None:
            raise ValueError("No embedded config in checkpoint. Please provide --config <path>.")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    exp_id = cfg.get("experiment_id", ckpt_path.parent.name)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"EVALUATION PROTOCOL: {exp_id.upper()}")
    print(f"  • Split:      {args.split.upper()} {'(UNLOCKED BY CONFIRMATION)' if args.split == 'test' else ''}")
    print(f"  • Checkpoint: {ckpt_path}")
    print(f"  • Device:     {dev}")
    print("=" * 80)

    # Load Data
    k_frames = cfg.get("dataset", {}).get("k_history", 5)
    channels = cfg.get("dataset", {}).get("channels", [0, 1, 2])
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    meta_dir = Path("data/metadata")

    manifest_file = meta_dir / f"forecast_{args.split}_sequences_k{k_frames}.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    split_df = pd.read_csv(manifest_file)
    with open(meta_dir / "normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)

    mean = [norm_stats["mean"][c] for c in channels]
    std = [norm_stats["std"][c] for c in channels]

    dataset = TCIRSequenceDataset(split_df, mean=mean, std=std, channels=channels, is_training=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=(dev.type == "cuda"), drop_last=False
    )

    feature_group = cfg.get("features", {}).get("group", "full_feature_set")
    env_manager = EnvironmentalFeatureManager(metadata_dir=meta_dir, feature_group=feature_group)

    # Load Model
    model = build_model_from_config(cfg, in_channels=len(channels)).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    m_type = cfg.get("model", {}).get("type", "cnn_transformer").lower()

    preds_list = []
    targets_list = []
    probs_list = []
    v_curr_list = []

    with torch.no_grad():
        for images, vis_masks, targets, meta in loader:
            images = images.to(dev, non_blocking=True)
            vis_masks = vis_masks.to(dev, non_blocking=True)
            v_curr = meta["vmax_curr"].to(dev).float()

            env_vectors = [
                env_manager.get_features(meta["cyclone_id"][i], int(meta["target_t_timestamp"][i]))
                for i in range(len(images))
            ]
            x_env = torch.stack(env_vectors).to(dev)

            if m_type == "residual":
                v_hat, _ = model(images, v_curr=v_curr, vis_masks=vis_masks)
                preds_list.append(v_hat.float().cpu().numpy())
            elif m_type == "ri_dedicated":
                logits = model(images, vis_masks=vis_masks, x_env=x_env)
                probs = torch.sigmoid(logits).float().cpu().numpy().flatten()
                probs_list.append(probs)
            elif m_type in ["ri_multitask", "multitask"]:
                v_hat, ri_logits, _ = model(images, vis_masks=vis_masks, x_env=x_env)
                preds_list.append(v_hat.float().cpu().numpy())
                probs_list.append(torch.sigmoid(ri_logits).float().cpu().numpy().flatten())
            elif m_type == "probabilistic":
                q_out = model(images, vis_masks=vis_masks)
                preds_list.append(q_out.float().cpu().numpy())
            else:
                preds = model(images, vis_masks=vis_masks)
                preds_list.append(preds.float().cpu().numpy())

            targets_list.append(targets.numpy())
            v_curr_list.extend(meta["vmax_curr"].numpy())

    dataset.close()

    eval_results = {
        "experiment_id": exp_id,
        "split": args.split,
        "n_samples": len(split_df),
        "checkpoint": str(ckpt_path),
    }

    # Physical Sanity Check Inspector & Trajectory Evaluator (Never modifies data!)
    sanity_checker = PhysicalSanityChecker(
        min_intensity_kt=cfg.get("sanity", {}).get("min_intensity_kt", 0.0),
        max_plausible_kt=cfg.get("sanity", {}).get("max_plausible_kt", 200.0),
        large_step_change_kt=cfg.get("sanity", {}).get("large_step_change_kt", 45.0),
        max_24h_delta_kt=cfg.get("sanity", {}).get("max_24h_delta_kt", 80.0),
    )
    trajectory_evaluator = TrajectoryEvaluator(
        step_thresholds_kt=(20.0, 30.0, 45.0, 60.0),
        dip_dip_tolerance_kt=5.0,
        steady_deadband_kt=2.5,
    )

    if m_type == "ri_dedicated":
        all_probs = np.concatenate(probs_list)
        all_targets = np.concatenate(targets_list, axis=0)
        actual_ri = (all_targets[:, 2] - np.array(v_curr_list) >= 30.0).astype(int)
        ri_metrics = compute_ri_metrics(actual_ri, all_probs)
        eval_results["ri_metrics"] = ri_metrics
        print("\n" + "=" * 80)
        print(f"RI EVALUATION RESULTS ({args.split.upper()}):")
        print(f"  • PR-AUC:          {ri_metrics['pr_auc']:.4f} (Prevalence: {ri_metrics['prevalence']:.4f})")
        print(f"  • ROC-AUC:         {ri_metrics['roc_auc']:.4f}")
        print(f"  • Brier Score:     {ri_metrics['brier_score']:.4f}")
        print(f"  • ECE:             {ri_metrics['ece']:.4f}")
        print(f"  • Best F1:         {ri_metrics['optimal_f1']:.4f} (at threshold {ri_metrics['optimal_threshold']:.3f})")
        print(f"  • Confusion Matrix: [[TN={ri_metrics['confusion_matrix'][0][0]}, FP={ri_metrics['confusion_matrix'][0][1]}], [FN={ri_metrics['confusion_matrix'][1][0]}, TP={ri_metrics['confusion_matrix'][1][1]}]]")
        print("=" * 80)

    elif m_type == "probabilistic":
        all_q = np.concatenate(preds_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        prob_metrics = compute_probabilistic_metrics(all_q, all_targets)
        eval_results["probabilistic_metrics"] = prob_metrics

        # Sanity check on median predictions
        sanity_report = sanity_checker.inspect(all_q[:, :, 1], v_curr=v_curr_list)
        eval_results["physical_sanity_checks"] = sanity_report

        print("\n" + "=" * 80)
        print(f"PROBABILISTIC EVALUATION RESULTS ({args.split.upper()}):")
        for h in ["+6h", "+12h", "+24h"]:
            print(f"  [{h}] Median MAE: {prob_metrics[f'mae_q50_{h}']:.2f} kt | "
                  f"Coverage: {prob_metrics[f'coverage_{h}']:.3f} | "
                  f"Width: {prob_metrics[f'width_{h}']:.1f} kt | "
                  f"Winkler: {prob_metrics[f'winkler_{h}']:.1f} | "
                  f"Crossings: {prob_metrics[f'crossing_rate_{h}']:.4f}")
        print("=" * 80)

    else:
        all_preds = np.concatenate(preds_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        v_curr_arr = np.array(v_curr_list)

        # Sanity check
        sanity_report = sanity_checker.inspect(all_preds, v_curr=v_curr_arr)
        eval_results["physical_sanity_checks"] = sanity_report

        # Trajectory Coherence & Roughness Evaluation
        traj_report = trajectory_evaluator.evaluate_trajectories(
            predictions=all_preds, targets=all_targets, v_curr=v_curr_arr
        )
        eval_results["trajectory_coherence"] = traj_report

        # Stratified evaluation
        stratified_metrics = evaluate_regime_stratified(
            df=split_df, preds=all_preds, targets=all_targets,
            ri_probs=np.concatenate(probs_list) if probs_list else None
        )
        eval_results["stratified_metrics"] = stratified_metrics

        print("\n" + "=" * 80)
        print(f"INTENSITY FORECASTING EVALUATION ({args.split.upper()}):")
        for h in ["+6h", "+12h", "+24h"]:
            m = stratified_metrics["by_horizon"][h]
            print(f"  [{h:4s}] MAE: {m['mae']:5.2f} kt (95% CI: [{m['mae_95ci'][0]:.2f}, {m['mae_95ci'][1]:.2f}]) | "
                  f"RMSE: {m['rmse']:5.2f} kt | Bias: {m['bias']:+5.2f} kt")
        print("-" * 80)
        print("Trajectory Coherence & False-Dip Diagnostics:")
        print(f"  • Trajectory Roughness Ratio (Pred/True): {traj_report['roughness_ratio_pred_vs_true']:.3f}")
        print(f"  • Second-Diff Error:                     {traj_report['second_diff_error_kt']:.2f} kt")
        print(f"  • False Dips Detected:                   {traj_report['false_dip_count']} ({traj_report['false_dip_rate_pct']}%)")
        print(f"  • False Peaks Detected:                  {traj_report['false_peak_count']} ({traj_report['false_peak_rate_pct']}%)")
        print(f"  • Directional Accuracy:                  {traj_report['directional_accuracy_pct']:.1f}%")
        print(f"  • Max 6h Forecast Change (Mean / P95):   {traj_report['max_6h_change_mean_kt']:.1f} kt / {traj_report['max_6h_change_p95_kt']:.1f} kt")
        print("-" * 80)
        print("Physical Realism Diagnostic Report:")
        print(f"  • Negative Intensities:   {sanity_report['negative_intensity_count']} ({sanity_report['negative_intensity_pct']}%)")
        print(f"  • Exceeds Physical Limit: {sanity_report['exceeds_ceiling_count']} ({sanity_report['exceeds_ceiling_pct']}%)")
        print(f"  • Large Step (>45kt/6h):  {sanity_report['large_single_step_count']} ({sanity_report['large_single_step_pct']}%)")
        print(f"  • Status:                 {sanity_report['status']}")
        print("=" * 80)

    # Save metrics JSON
    metrics_path = out_dir / f"evaluation_{args.split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[Saved Evaluation Report] -> {metrics_path}")


if __name__ == "__main__":
    main()
