# this works for nequip output as set from the tutorial used
# should implement something similar for mace output as well

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams.update({'font.size': 14})
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath("results.py")))
import results


colors_dict = {
    "forces_mae": "#d51700",
    "forces_mse": "#d51700",
    "per_atom_energy_mae": "#0096ff",
    "per_atom_energy_mse": "#0096ff",
    "total_energy_mae": "#0433ff",
    "total_energy_mse": "#0433ff",
    "weighted_sum": "#9437ff",
}

def plot_metric_in_ax(ax, df, group, type="metric"):
    print(f"[INFO]: Plotting {group} {type}")
    df, metric_names = results.get_metrics(df, group, type)
    if df is None:
        return
    
    print("[INFO]: dataframe columns:", df.columns)
    print("[INFO]: number of points:", len(df))

    group_label = group.capitalize()
    lines, labels = [], []

    # if type == "metric":
    if True:
        ax2 = ax.twinx()
        for metric in metric_names:
            label = metric.split("/")[0]
            if label not in colors_dict:
                print(f"[WARNING] unrecognised label {label}, skipping. Add it to colors_dict at the top of this script to plot it")
                continue
            if "force" in metric:
                line = ax2.plot(df["epoch"], df[metric], color=colors_dict[label])[0]
            else:
                line = ax.plot(df["epoch"], df[metric], color=colors_dict[label])[0]
            lines.append(line)
            labels.append(label)
        ax.set_ylabel("[eV]")
        ax2.set_ylabel("[eV/Å]")
        ax2.set_yscale("log")
    else:
        for metric in metric_names:
            label = metric.split("/")[0]
            if label not in colors_dict:
                continue
            line = ax.plot(df["epoch"], df[metric], color=colors_dict[label])[0]
            lines.append(line)
            labels.append(label)

    ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_title(f"{group_label} {type}")

def plot_metrics(df, dirname):
    groups = results.detect_groups(df)
    print("[INFO]: Detected groups:", groups)
    train_val_groups = [g for g in groups if g.startswith("train") or g.startswith("val")]
    test_groups = [g for g in groups if g.startswith("test")]

    n_groups = len(train_val_groups)
    fig, ax = plt.subplots(n_groups, 2, figsize=(14, 4 * n_groups))
    if n_groups == 1:
        ax = [ax]  # handle single group

    for i, group in enumerate(train_val_groups):
        plot_metric_in_ax(ax[i][0], df, group, type="loss")
        plot_metric_in_ax(ax[i][1], df, group, type="metric")

    plt.tight_layout()
    plt.savefig(os.path.join(dirname, "training_metrics.pdf"))

    # Print test results
    results.print_test_results(df, test_groups, dirname)

def plot_lr(df, dirname):
    if "lr-Adam" not in df.columns:
        print("[WARNING]: No learning rate data found.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot([lr for lr in filter(lambda v: v==v, df["lr-Adam"])])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, "learning_rate.pdf"))

if __name__ == "__main__":
    if len(sys.argv) < 1:
        metrics_path = input("Path to metrics file: ")
    else:
        metrics_path = sys.argv[1]
    metrics_dir = os.path.dirname(metrics_path)
    metrics_df = pd.read_csv(metrics_path)

    plot_metrics(metrics_df, metrics_dir)
    plot_lr(metrics_df, metrics_dir)