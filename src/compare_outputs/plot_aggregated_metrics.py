#!/usr/bin/env python3
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def create_plot(models, data_key, title, ylabel, output_filename):
    """Generates and saves a clean matplotlib bar plot with error bars."""
    values = []
    yerr_lower = []
    yerr_upper = []
    valid_models = []

    for model in models:
        metric = models[model].get(data_key)
        if not metric or metric.get("value") is None:
            continue

        val = metric["value"]
        quantiles = metric.get("quantiles")

        values.append(val)
        valid_models.append(model)

        # Calculate asymmetric error bars relative to the central value if quantiles exist
        if quantiles and "0.025" in quantiles and "0.975" in quantiles:
            q_low = quantiles["0.025"]
            q_high = quantiles["0.975"]
            # Error arrays must be positive sizes absolute differences relative to value
            yerr_lower.append(max(0, val - q_low))
            yerr_upper.append(max(0, q_high - val))
        else:
            yerr_lower.append(0)
            yerr_upper.append(0)

    if not values:
        print(f"No valid data found for metric: {data_key}. Skipping plot.")
        return

    yerr = [yerr_lower, yerr_upper]

    fig, ax = plt.subplots()
    x_pos = np.arange(len(valid_models))

    # Plotting using error bars over categorical bar charts
    ax.bar(
        x_pos,
        values,
        yerr=yerr,
        align="center",
        alpha=0.8,
        capsize=5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_models, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    # ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf")
    plt.close()
    print(f"Saved plot: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from aggregated validation metrics json."
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to the aggregated JSON file containing model metrics",
    )
    args = parser.parse_args()

    data = load_data(args.input_json)

    # Dictionary mapping internal data keys to layout metadata
    plots_config = {
        "energy_per_atom_RMSE": {
            "title": "Energy per Atom RMSE",
            "ylabel": "E RMSE [eV/atom]",
            "filename": "MACECOMPARISONenergy_rmse.pdf",
        },
        "aggregated_force_norm_RMSE": {
            "title": "Aggregated Force Norm RMSE",
            "ylabel": r"$|\Delta F|$ RMSnE [eV/$\AA$]",
            "filename": "MACECOMPARISONforce_rmse.pdf",
        },
        "stress_RMSE": {
            "title": "Stress Component RMSE",
            "ylabel": r"$|\Delta \sigma|$ RMSnE [eV/$\AA^3$]",
            "filename": "MACECOMPARISONstress_rmse.pdf",
        },
    }

    for metric_key, config in plots_config.items():
        create_plot(
            models=data,
            data_key=metric_key,
            title=config["title"],
            ylabel=config["ylabel"],
            output_filename=config["filename"],
        )


if __name__ == "__main__":
    main()