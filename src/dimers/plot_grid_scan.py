# =====================================
# Script Name: plot_grid_scan.py
# Purpose: Plot grid scan results for 2D or 3D potential energy surface scans
# Usage: Run with --root_dir to visualize how energy varies with different bond lengths or angles
#        in multi-dimensional configuration spaces using MACE calculator evaluations.
# =====================================

"""Grid scan visualization module for potential energy surfaces.

This script plots grid scan results for 2D or 3D potential energy surface scans, visualizing
how energy varies with different bond lengths or angles in multi-dimensional configuration spaces.
It evaluates trained MACE models on validation points, uses CSV caching for efficiency, and
generates comprehensive grid visualization plots.
"""

#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator


def parse_args():
    parser = ArgumentParser(
        description="Evaluate trained MACE models with validation points, CSV caching, and grid visualization."
    )
    parser.add_argument(
        "--root_dir",
        "-r",
        type=str,
        default=".",
        help="Root directory containing the grid scan folders",
    )
    parser.add_argument(
        "--train_file",
        "-t",
        type=str,
        default="../dimers/train.extxyz",
        help="Path to training file to extract species keys",
    )
    parser.add_argument(
        "--valid_file",
        "-v",
        type=str,
        default="../dimers/val.extxyz",
        help="Path to validation file containing reference points",
    )

    return parser.parse_args()


def get_chemical_species(train_file):
    """Parses training configs to find unique atomic species pairs."""
    configs = read(train_file, index=":")
    all_symbols = set()
    for c in configs:
        all_symbols.update(c.get_chemical_symbols())
    species = sorted(list(all_symbols))
    return list(itertools.combinations_with_replacement(species, 2))


def extract_validation_data(valid_file, pair):
    """Extracts bond lengths and target potential energies for a given pair from the validation file."""
    configs = read(valid_file, index=":")
    pair_symbols = sorted(list(pair))
    distances = []
    energies = []

    for c in configs:
        if len(c) != 2:
            continue
        if sorted(c.get_chemical_symbols()) == pair_symbols:
            distances.append(c.get_distance(0, 1))
            try:
                energies.append(c.get_potential_energy())
            except RuntimeError:
                continue

    return np.array(distances), np.array(energies)


def generate_dimer_curve(symbols, r_min=1.0, r_max=6.0, steps=150):
    """Generates equidistant configurations representing a clear dimer stretch."""
    distances = np.linspace(r_min, r_max, steps)
    dimers = [
        Atoms(symbols=symbols, positions=[[0, 0, 0], [0, 0, d]])
        for d in distances
    ]
    return distances, dimers


def setup_plot_style(ax, x_label=True, y_label=True):
    """Applies uniform, publication-ready styling to an axis."""
    if x_label:
        ax.set_xlabel(r"Distance, $d$ [$\mathrm{\AA}$]", fontsize=12, labelpad=5)
    if y_label:
        ax.set_ylabel(r"Potential Energy, $E$ [eV]", fontsize=12, labelpad=5)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    args = parse_args()
    root_path = Path(args.root_dir)

    pairs = get_chemical_species(args.train_file)
    pair_names = ["".join(p) for p in pairs]
    print(f"[INFO] Target chemical pairs detected: {pair_names}")

    # Track data across runs: merged_data["AB"]["model_label"] = (distances, energies)
    merged_data = {p_name: {} for p_name in pair_names}

    # Find models
    model_paths = sorted(list(root_path.glob("**/MACE_v0.model")))
    if not model_paths:
        print("[ERROR] No 'MACE_v0.model' files found.")
        return 1
    print(f"[INFO] Found {len(model_paths)} trained checkpoints to process.")

    # 1. Compute/Load and save individual folder potentials
    for model_path in model_paths:
        model_dir = model_path.parent
        model_label = model_dir.name
        
        # Initialize calculator variable to lazy-load only when required
        calc = None

        for pair in pairs:
            pair_name = "".join(pair)
            
            # Local Directory: modeldir/2_body_potential/AB/
            out_dir = model_dir / "2_body_potential" / pair_name
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "potential_curve.csv"

            # Check if cached CSV exists
            if csv_path.exists():
                print(f"[CACHE] Loading pre-computed data for {model_label} ({pair_name})")
                df = pd.read_csv(csv_path)
                distances = df["distance"].to_numpy()
                energies = df["pe"].to_numpy()
            else:
                print(f"[COMPUTE] Simulating potential curve via: {model_label} ({pair_name})")
                distances, dimers = generate_dimer_curve(pair)

                # Instantiating MACE Calculator only on demand
                if calc is None:
                    try:
                        calc = MACECalculator(model_paths=str(model_path), device="cuda")
                    except Exception:
                        calc = MACECalculator(model_paths=str(model_path), device="cpu")

                energies = []
                for atoms in dimers:
                    atoms.calc = calc
                    try:
                        energies.append(atoms.get_potential_energy())
                    except Exception:
                        energies.append(np.nan)
                energies = np.array(energies)

                # Save predictions to CSV
                df = pd.DataFrame({"distance": distances, "pe": energies})
                df.to_csv(csv_path, index=False)

            # Keep for aggregated plotting steps
            merged_data[pair_name][model_label] = (distances, energies)

            # Extract validation coordinates matching this chemistry
            v_dist, v_energies = extract_validation_data(args.valid_file, pair)

            fig, ax = plt.subplots(figsize=(7, 5))
            setup_plot_style(ax)

            # Add validation reference scatter underneath curve
            if len(v_dist) > 0:
                ax.scatter(
                    v_dist,
                    v_energies,
                    color="#e377c2",
                    edgecolor="k",
                    s=35,
                    alpha=0.8,
                    zorder=2,
                    label="Validation Reference",
                )

            ax.plot(
                distances,
                energies,
                color="#1f77b4",
                linewidth=2.5,
                zorder=3,
                label="Model Prediction",
            )
            ax.set_title(f"{pair_name} - {model_label}", fontsize=11)
            ax.legend(fontsize=9, frameon=True)
            fig.tight_layout()

            fig.savefig(
                out_dir / "dimer_potential.svg",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close(fig)

    # 2. Master Aggregations (Executed inside root/2_body_potential/AB/)
    print("\n[INFO] Gathering plots for master reports...")
    for pair in pairs:
        pair_name = "".join(pair)
        master_dir = root_path / "2_body_potential" / pair_name
        master_dir.mkdir(parents=True, exist_ok=True)

        v_dist, v_energies = extract_validation_data(args.valid_file, pair)
        models_dict = merged_data[pair_name]
        num_models = len(models_dict)

        if num_models == 0:
            continue

        # --- Calculate Global Extremes for the Grid scaling ---
        minima = []
        for label, (distances, energies) in models_dict.items():
            valid_energies = energies[~np.isnan(energies)]
            if len(valid_energies) > 0:
                minima.append(np.min(valid_energies))
        
        if minima:
            lomin = np.min(minima)
            himin = np.max(minima)

            ymin_bound = lomin - 1
            ymax_bound = himin + 30
            grid_ylim = (ymin_bound, ymax_bound)
        else:
            grid_ylim = None

        # --- PLOT Type A: Overlapped Master Plot ---
        fig_ol, ax_ol = plt.subplots(figsize=(11, 8))
        setup_plot_style(ax_ol)

        if len(v_dist) > 0:
            ax_ol.scatter(
                v_dist,
                v_energies,
                color="#ff0000",
                edgecolor="k",
                s=45,
                alpha=0.9,
                zorder=2,
                label="Validation Reference",
            )

        for label, (distances, energies) in models_dict.items():
            ax_ol.plot(distances, energies, linewidth=1.5, alpha=0.7, label=label)

        ax_ol.set_title(f"Overlapped Potentials: {pair_name}", fontsize=14)
        ax_ol.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True, ncol=2
        )
        fig_ol.tight_layout()
        fig_ol.savefig(
            master_dir / f"merged_{pair_name}_overlapped.svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig_ol)

       # --- PLOT Type B: Grid Scan View (Unified scale applied) ---
        cols = 10
        rows = math.ceil(num_models / cols)

        fig_grid, axes = plt.subplots(
            rows, cols, figsize=(cols * 3.5, rows * 3.0), sharex=True, sharey=True
        )
        axes = axes.flatten()

        for idx, (label, (distances, energies)) in enumerate(models_dict.items()):
            ax = axes[idx]
            
            is_left = (idx % cols == 0)
            # A subplot needs x-labels if it's in the last row OR if the plot below it doesn't exist
            is_bottom = (idx >= (rows - 1) * cols) or (idx + cols >= num_models)
            
            setup_plot_style(ax, x_label=is_bottom, y_label=is_left)

            # CRITICAL FIX: Explicitly tell matplotlib to show the labels 
            # for the bottom-most active plots in incomplete rows
            if is_bottom:
                ax.xaxis.set_tick_params(labelbottom=True)

            if len(v_dist) > 0:
                ax.scatter(
                    v_dist,
                    v_energies,
                    color="#ff0000",
                    edgecolor="none",
                    s=15,
                    alpha=0.6,
                    zorder=2,
                )

            ax.plot(distances, energies, color="#1f77b4", linewidth=2, zorder=3)
            
            # Apply the unified calculated y-limits here
            if grid_ylim is not None:
                ax.set_ylim(grid_ylim)
            
            clean_title = label.replace("run_mlp_", "")
            ax.set_title(clean_title, fontsize=9, pad=2)

        for idx in range(num_models, len(axes)):
            fig_grid.delaxes(axes[idx])

        fig_grid.suptitle(f"Grid Scan Overview: {pair_name}", fontsize=18, y=0.99)
        fig_grid.tight_layout()
        fig_grid.savefig(
            master_dir / f"merged_{pair_name}_grid.svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig_grid)
        
        print(f"[SUCCESS] Master plots complete for chemistry: {pair_name}")

    return 0


if __name__ == "__main__":
    main()