#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def find_validation_files(root_dir, max_depth=4):
    """Finds all validation_statistics.json files up to a specific depth."""
    root_path = Path(root_dir)
    target_name = "validation_statistics.json"

    # Walk directories and enforce depth constraints manually for cross-platform robustness
    for dirpath, dirnames, filenames in os.walk(root_path):
        depth = len(Path(dirpath).relative_to(root_path).parts)
        if depth >= max_depth:
            # Prevent deeper walking
            dirnames.clear()

        if target_name in filenames:
            yield Path(dirpath) / target_name


def extract_metrics(json_path):
    """Extracts required metrics and quantiles from a validation JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    total = data.get("total", {})

    # Mapping internal keys to the target metrics
    metric_mappings = {
        "energy_per_atom_RMSE": total.get("E_per_atom", {}).get("RMSE", {}),
        "aggregated_force_norm_RMSE": total.get("aggregated_F_norm", {})
        .get("RMSE", {}),
        "stress_RMSE": total.get("S_norm", {}).get("RMSE", {}),
    }

    model_data = {}
    for key, source in metric_mappings.items():
        if source:
            model_data[key] = {
                "value": source.get("value"),
                "quantiles": source.get("quantiles"),
            }
        else:
            model_data[key] = {"value": None, "quantiles": None}

    return model_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract and aggregate validation metrics from model subdirectories."
    )
    parser.core_argument = parser.add_argument(
        "root", type=str, help="Root directory containing the model folders"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="aggregated_validation_results.json",
        help="Path to save the aggregated JSON output",
    )
    args = parser.parse_args()

    aggregated_results = {}

    print(f"Scanning for validation files in: {args.root}")
    for file_path in find_validation_files(args.root):
        # Extract model identifier name from parent path (e.g., 'mace-GAP2020-FROM_SCRATCH_v3')
        # Adjust index if your path hierarchy dictates a different structural key
        model_name = file_path.parts[-4]

        print(f"Extracting metrics from: {model_name}")
        try:
            metrics = extract_metrics(file_path)
            aggregated_results[model_name] = metrics
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    with open(args.output, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\nSuccessfully aggregated data saved to: {args.output}")


if __name__ == "__main__":
    main()