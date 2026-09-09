# standard way of reading/writing output files for model evaluation on test set
"""Results processing and aggregation module for NequIP model evaluations.

This module provides utilities to extract, process, and format metrics from
NequIP model training outputs and test set evaluations. It handles detection of
training/validation/test groups, extraction of specific metric types (loss or
performance metrics), and formatting of results into JSON files.
"""

import os, pandas, numpy as np
from typing import Literal

import json


def detect_groups(df):
    """Detect group prefixes like train, val0, val1, test0, test1, etc.

    Args:
        df (pandas.DataFrame): DataFrame containing metric columns with epoch/step suffixes

    Returns:
        list: Sorted list of unique group prefix names (e.g., ['train', 'val0', 'test1'])
    """
    groups = set()
    for col in df.columns:
        if "_epoch" in col or "_step" in col:
            prefix = col.split("_")[0]  # e.g., train, val0, test1
            groups.add(prefix)
    return sorted(groups)

def get_metrics(df, group, type: Literal["loss", "metric"] = "metric"):
    """Extract metrics for a given group and type.
       For train: only use *_epoch columns (ignore step)."""
    if group == "train":
        if type == "loss":
            column_names = [c for c in df.columns if c.startswith("train_loss_epoch/")]
        else:  # metric
            column_names = [c for c in df.columns if c.startswith("train_metric_epoch/")]
    elif group.startswith("val"):
        column_names = [c for c in df.columns if c.startswith(group)]
    elif group.startswith("test"):
        column_names = [c for c in df.columns if c.startswith(group)]
    else:
        column_names = []

    if not column_names:
        return None, None

    # Keep only relevant columns + epoch
    df_out = df[["epoch"] + column_names].dropna(axis=0, subset=column_names)

    # Rename so only "forces_mae", "forces_rmse", etc. remain
    renamed = {c: c.split("/")[-1] for c in column_names}
    df_out = df_out.rename(columns=renamed)
    return df_out, list(renamed.values())

def print_test_results(df, groups, dirname):
    """Print metrics for test sets (computed once)."""
    with open(os.path.join(dirname, "test_results.json"), "w") as f:
        results_dict = {}
        for group in groups:
            if not group.startswith("test"):
                continue

            df_group, metric_names = get_metrics(df, group, type="metric")
            if df_group is None or df_group.empty:
                continue
            results_dict['comment'] = f"# Results for {group}"

            print(f"\n===== Results for {group.upper()} =====")
            # just take last available row (only one point usually)
            row = df_group.iloc[-1]
            for metric in metric_names:
                val = row[metric]
                print(f"{metric:25s}: {val:.6f}")
                results_dict[f'{metric}'] = val

            json.dump(
                results_dict, f, indent=4
            )