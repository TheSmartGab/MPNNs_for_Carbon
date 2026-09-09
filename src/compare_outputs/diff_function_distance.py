#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

from pprint import pprint
import json


def mae_rmse(x):
    mae = np.mean(np.abs(x))
    rmse = np.sqrt(np.mean(x**2))
    return mae, rmse


def compute_distance(atoms):
    """Return the distance between the two atoms."""
    if len(atoms) != 2:
        raise ValueError("Frame does not have exactly 2 atoms.")
    pos = atoms.get_positions()
    return np.linalg.norm(pos[0] - pos[1])


def main(args):
    """Used to analyze errors of many models on the dimer as a function of distance."""
    frames_ref = read(args.ref_file, ":")
    frames_model = read(args.model_file, ":")

    if len(frames_ref) != len(frames_model):
        raise ValueError("Files have different number of frames.")

    distances = []
    energy_errors = []
    force_errors = []

    for a_ref, a_model in zip(frames_ref, frames_model):
        if len(a_ref) != 2 or len(a_model) != 2:
            continue  # skip frames that are not 2 atoms

        # --- distance ---
        d = compute_distance(a_ref)
        distances.append(d)

        # --- energy per atom error ---
        e_ref = a_ref.get_potential_energy() / 2
        e_model = a_model.get_potential_energy() / 2
        energy_errors.append(e_model - e_ref)

        # --- force error (component-wise) ---
        f_ref = a_ref.get_forces()
        f_model = a_model.get_forces()
        df = f_model[0][0] - f_ref[0][0]
        force_errors.append(df)  # assume forces are directed along x axis

    distances = np.array(distances)
    energy_errors = np.array(energy_errors)
    force_errors = np.array(force_errors)

    # --- sort by distance for plotting ---
    sort_idx = np.argsort(distances)
    distances = distances[sort_idx]
    energy_errors = energy_errors[sort_idx]
    force_errors = force_errors[sort_idx]

    e_mae, e_rmse = mae_rmse(energy_errors)
    e_mean = np.mean(energy_errors)
    f_mae, f_rmse = mae_rmse(force_errors)
    f_mean = np.mean(force_errors)

    # --- plot energy error vs distance ---
    plt.figure(figsize=(6, 4))
    plt.scatter(distances, energy_errors, c="blue", label="Energy per atom error")
    plt.xlabel("C-C distance (Å)")
    plt.ylabel("Energy error (eV/atom)")
    plt.title(f"Energy error vs distance (model - reference)\nMAE: {e_mae:.3f}, RMSE: {e_rmse:.3f}, MEAN: {e_mean:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "energy_error_vs_distance.pdf"))
    plt.close()

    # --- plot force error vs distance ---
    plt.figure(figsize=(6, 4))
    plt.scatter(distances, force_errors, c="red", label="Mean force error per atom")
    plt.xlabel("C-C distance (Å)")
    plt.ylabel("Force error (eV/Å)")
    plt.title(f"Force error vs distance (model -  reference)\nMAE:{f_mae:.3f}, RMSE: {f_rmse:.3f}, MEAN: {f_mean:.3f}")
    plt.grid(True)
    plt.tight_layout()
    print(f"[INFO] Saving plots to {args.output_dir}/force_error_vs_distance.pdf")

    plt.savefig(os.path.join(args.output_dir, "force_error_vs_distance.pdf"))
    plt.close()

    results = np.array(
        [[d, de, df] for d, de, df in zip(distances, energy_errors, force_errors)]
    )
    results_path = os.path.join(args.output_dir, "energy_force_errors.txt")
    print("[INFO] Saving results to", results_path)
    np.savetxt(results_path, results)

    metrics_path = os.path.join(args.output_dir, "mae_rmse.json")
    metrics = {
        "energy_per_atom" : {
            "mae" : e_mae,
            "rmse" : e_rmse,
            "mean" : e_mean
        },
        "force" : {
            "mae" : f_mae,
            "rmse" : f_rmse,
            "mean" : f_mean
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot error vs distance for C2 molecule"
    )
    parser.add_argument("ref_file", help="Reference XYZ/extxyz file")
    parser.add_argument("model_file", help="Model XYZ/extxyz file")
    parser.add_argument("output_dir", help="Output directory for plots and results")
    args = parser.parse_args()

    print("[INFO] running", __file__, "with args:")
    pprint(args)
    main(args)
