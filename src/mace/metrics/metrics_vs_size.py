#!/usr/bin/env python3
# =====================================
# Script Name: metrics_vs_size.py
# Purpose: Analyze MACE model performance (RMSE, MAE) as a function of training dataset size
# Usage: Run with --base_dir argument to process multiple training runs and generate performance plots
# =====================================

"""Model performance metrics analysis module for MACE.

This script analyzes MACE training results across different training set sizes, generating
log-log plots that show how model metrics (RMSE, MAE for energies and forces) vary with
training data quantity. It uses specific linear tick intervals on log-log axes to clearly
visualize scaling behavior and convergence patterns.

Functions:
    parse_args: Parse command-line arguments for the analysis script
"""

import argparse
import json
import os
import glob
import re
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments for metrics vs size analysis.

    Returns:
        Namespace: Parsed arguments including base_dir and other settings.
    """
    parser = argparse.ArgumentParser(
        description="Analyze MACE training results across training set sizes."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Root directory containing one sub-folder per training size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size used during training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mace_analysis",
        help="Directory where JSON summary and PNG plots are saved.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of training sizes to process.",
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def discover_sizes(base_dir: Path) -> list[int]:
    sizes = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            sizes.append(int(entry.name))
    return sorted(sizes)


def find_train_txt(size_dir: Path) -> Path | None:
    pattern = str(size_dir / "results" / "*_train.txt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    primary = [m for m in matches if "stage" not in os.path.basename(m).lower()]
    return Path(primary[0] if primary else matches[0])


def parse_eval_records(txt_path: Path) -> list[dict]:
    records = []
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("mode") == "eval" and obj.get("epoch") is not None:
                records.append(obj)
    return records


def best_epoch(records: list[dict], metric: str) -> dict | None:
    valid = [r for r in records if metric in r and r[metric] is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r[metric])


def steps_for_epoch(epoch: int, train_size: int, batch_size: int) -> int:
    steps_per_epoch = math.ceil(train_size / batch_size)
    return (epoch + 1) * steps_per_epoch


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = args.sizes if args.sizes else discover_sizes(base_dir)
    if not sizes:
        raise RuntimeError(f"No training-size directories found under {base_dir}")
    print(f"Training sizes found: {sizes}")

    # Metrics: (json_key, human_label, file_suffix, factor, tick_interval)
    metrics = [
        ("rmse_e_per_atom", "RMSE Energy (meV/atom)", "energy_per_atom", 1000, 5.0),
        ("rmse_f", "RMSE Forces (meV/Å)", "forces", 1000, 50.0),
        ("rmse_stress", r"RMSE Stress ($meV/Å^3$)", "stress", 1000, 1.0),
    ]

    all_results: dict[int, dict] = {}

    for size in sizes:
        size_dir = base_dir / str(size)
        txt_path = find_train_txt(size_dir)

        if txt_path is None:
            print(f"  [WARN] No *_train.txt found in {size_dir / 'results'} — skipping.")
            continue

        print(f"\nProcessing size={size}  →  {txt_path.name}")
        records = parse_eval_records(txt_path)

        if not records:
            print(f"  [WARN] No eval records found — skipping.")
            continue

        size_summary: dict = {}

        for metric_key, metric_label, _, factor, _ in metrics:
            best = best_epoch(records, metric_key)
            
            if best is None:
                print(f"  [WARN] Metric '{metric_key}' not found in records.")
                continue
            
            # Create a copy to avoid modifying the original record mid-loop
            best_val = best[metric_key] * factor
            epoch = int(best["epoch"])
            n_steps = steps_for_epoch(epoch, size, args.batch_size)

            print(f"  {metric_key}: best epoch={epoch} value={best_val:.6f}")

            size_summary[metric_key] = {
                "best_epoch": epoch,
                "total_steps": n_steps,
                "value": best_val, # We store the scaled value here
                "rmse_e": best.get("rmse_e"),
                "rmse_e_per_atom": best.get("rmse_e_per_atom"),
                "rmse_f": best.get("rmse_f"),
                "rmse_stress": best.get("rmse_stress"),
                "loss": best.get("loss"),
            }

        all_results[size] = size_summary

    # Save JSON
    json_path = output_dir / "mace_results_summary.json"
    with open(json_path, "w") as fh:
        json.dump({str(k): v for k, v in sorted(all_results.items())}, fh, indent=2)

    # Plots
    plot_style = {"marker": "o", "markersize": 8, "linewidth": 1.8}
    colors = {"rmse_e_per_atom": "#e05c5c", "rmse_f": "#4a90d9", "rmse_stress": "#5cb85c"}

    for metric_key, metric_label, file_suffix, _, interval in metrics:
        pairs = []
        for size in sorted(all_results.keys()):
            entry = all_results[size].get(metric_key)
            if entry and entry.get("value") is not None:
                pairs.append((size, entry["value"]))

        if len(pairs) < 2:
            continue

        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)

        log_x = np.log10(xs)
        log_y = np.log10(ys)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        fit_y = 10 ** (slope * log_x + intercept)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(xs, ys, color=colors[metric_key], label="data", **plot_style)
        ax.loglog(xs, fit_y, color=colors[metric_key], linestyle="--", linewidth=1.4, alpha=0.7, label=f"power-law fit (slope={slope:.2f})")

        # TICK LOGIC
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=interval))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        ax.set_xlabel("Training set size", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f"{metric_label} vs Training Size", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(4, 6), textcoords="offset points", fontsize=8, color=colors[metric_key])

        plt.tight_layout()
        fig.savefig(output_dir / f"mace_{file_suffix}_vs_training_size.png", dpi=150)
        plt.close(fig)

    # Combined plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MACE Model: RMSE vs Training Set Size (log-log)", fontsize=14)

    for ax, (metric_key, metric_label, _, _, interval) in zip(axes, metrics):
        pairs = []
        for size in sorted(all_results.keys()):
            entry = all_results[size].get(metric_key)
            if entry and entry.get("value") is not None:
                pairs.append((size, entry["value"]))

        if len(pairs) < 2:
            ax.set_title(metric_label)
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue

        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)
        slope, intercept = np.polyfit(np.log10(xs), np.log10(ys), 1)
        fit_y = 10 ** (slope * np.log10(xs) + intercept)

        ax.loglog(xs, ys, color=colors[metric_key], label="data", **plot_style)
        ax.loglog(xs, fit_y, color=colors[metric_key], linestyle="--", linewidth=1.4, alpha=0.7, label=f"slope={slope:.2f}")

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=interval))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        ax.set_xlabel("Training set size", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "mace_all_metrics_combined.png", dpi=150)
    plt.close(fig)
    print("\nDone.")

if __name__ == "__main__":
    main()