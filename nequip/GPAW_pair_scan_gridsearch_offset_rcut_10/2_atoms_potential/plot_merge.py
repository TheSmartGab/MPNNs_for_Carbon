import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
import re
from ase.io import read
from numpy.linalg import norm

# adjust as needed
REL_PATH = "2_atoms_potential/model_penergy_forces.out"
HOOK = REL_PATH.split('/')[0]


def get_datafiles(root_dir):
    """
    Walk through root_dir and collect all valid energy_vs_distance files.
    """
    all_data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if HOOK in dirnames:
            data_file_path = os.path.join(dirpath, REL_PATH)
            if os.path.isfile(data_file_path):
                try:
                    data = np.loadtxt(data_file_path)
                    all_data.append((dirpath, data))
                except Exception as e:
                    print(f"Could not load {data_file_path}: {e}")
    return all_data


def auto_grid(n):
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)
    return nrows, ncols


def extract_version_number(path):
    digits = re.findall(r"\d+", path)
    if not digits:
        raise ValueError(f"No digits found in directory name '{path}'")
    return int(digits[-1])


def load_extxyz_dist_energy(path):
    """
    Load extxyz with two-atom systems and extract (distance, energy).
    """
    frames = read(path, index=":")
    distances = []
    energies = []
    for atoms in frames:
        pos = atoms.get_positions()
        if len(pos) != 2:
            continue
        d = norm(pos[1] - pos[0])
        e = atoms.get_potential_energy()
        distances.append(d)
        energies.append(e)
    return np.array(distances), np.array(energies)


def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple energy vs distance datasets into a single PDF."
    )
    parser.add_argument("root_dir", type=str,
                        help="Root directory containing model subdirectories.")
    parser.add_argument("--output_file", type=str, default="energy_plots.pdf",
                        help="Output PDF file.")

    parser.add_argument("--training_data", type=str, required=False,
                        help="extxyz training dataset; will be plotted on all plots.")
    parser.add_argument("--validation_data", type=str, required=False,
                        help="extxyz validation dataset; will be plotted on all plots.")
    parser.add_argument("--test_data", type=str, required=False,
                        help="extxyz test dataset; will be plotted on all plots.")

    args = parser.parse_args()

    # Load main data
    all_data = get_datafiles(args.root_dir)

    if not all_data:
        print("No data files found.")
        return

    # Extract version numbers & sort
    versions = [extract_version_number(path) for path, _ in all_data]
    sorted_idx = np.argsort(versions)
    sorted_data = [all_data[i] for i in sorted_idx]
    versions = [versions[i] for i in sorted_idx]

    n = len(all_data)
    nrows, ncols = auto_grid(n)

    print(f"Found {n} data files. Using grid {nrows} x {ncols}")

    # Load optional datasets
    train = val = test = None

    if args.training_data:
        train = load_extxyz_dist_energy(args.training_data)

    if args.validation_data:
        val = load_extxyz_dist_energy(args.validation_data)

    if args.test_data:
        test = load_extxyz_dist_energy(args.test_data)

    # Plotting
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3*ncols, 3*nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, (path, data), version in zip(axes, sorted_data, versions):
        # Model curve
        ax.plot(data[:, 0], data[:, 1], label="Model curve")

        # Training data
        if train is not None:
            ax.scatter(train[0], train[1], s=2, marker="o", label="Train", alpha=0.2)
            xmin = np.min(train[0])
            xmax = np.max(train[0])
            ax.set_xlim((xmin, xmax))
            ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax)))

        # Validation data
        if val is not None:
            ax.scatter(val[0], val[1], s=4, marker="s", label="Validation", alpha=0.3)

        # Test data
        if test is not None:
            ax.scatter(test[0], test[1], s=4, marker="^", label="Test", alpha=0.3)

        ax.set_title(f"v{version}")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Energy")
        ax.grid(True)

        min_energy = np.min(data[:, 1])
        ax.set_ylim((min_energy - 5, min_energy + 30))
        if train is None:
            ax.set_xticks(np.arange(0, 11))

        ax.legend(fontsize=6)

    # Remove unused axes
    for ax in axes[len(all_data):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(args.output_file)
    print(f"Saved PDF to {args.output_file}")


if __name__ == "__main__":
    main()

