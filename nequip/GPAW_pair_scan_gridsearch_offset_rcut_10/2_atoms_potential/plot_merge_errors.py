import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt
import re

SPLIT_KEYS = ["training", "validation", "test"]
MODEL_FILE = "energy_force_errors.txt"


# -------------------------
# utilities
# -------------------------

def auto_grid(n):
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)
    return nrows, ncols


def extract_version_number(path):
    digits = re.findall(r"v(\d+)", path)
    return int(digits[-1]) if digits else -1


def collect_models(root_dir):
    """Find all energy_force_errors.txt files."""
    models = []
    for dirpath, _, filenames in os.walk(root_dir):
        if MODEL_FILE in filenames:
            data = np.loadtxt(os.path.join(dirpath, MODEL_FILE))
            models.append((dirpath, data))
    return models


def compute_quantiles(values, qs=(5, 50, 95)):
    """
    values: (n_models, n_points)
    returns: (len(qs), n_points)
    """
    return np.percentile(values, qs, axis=0)


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge NequIP 2-atom scan errors and compute quantiles."
    )
    parser.add_argument("root_dir", type=str)
    parser.add_argument("--output_prefix", type=str, default="merged")

    args = parser.parse_args()
    out_dir = args.output_prefix
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    models = collect_models(args.root_dir)
    if not models:
        print("No model data found.")
        return

    versions = [extract_version_number(p) for p, _ in models]
    order = np.argsort(versions)

    models = [models[i] for i in order]
    versions = [versions[i] for i in order]

    n_models = len(models)
    nrows, ncols = auto_grid(n_models)

    # --------------------------------------------------
    # Stack data across models
    # --------------------------------------------------

    all_d = []
    all_E = []
    all_F = []

    for _, data in models:
        d, E, F = data.T
        all_d.append(d)
        all_E.append(E)
        all_F.append(F)

    all_d = np.array(all_d)
    all_E = np.array(all_E)
    all_F = np.array(all_F)

    # assume common distance grid
    dist = all_d[0]

    # --------------------------------------------------
    # Compute quantiles
    # --------------------------------------------------

    E_q = compute_quantiles(all_E)   # shape (3, N)
    F_q = compute_quantiles(all_F)

    # --------------------------------------------------
    # ENERGY vs DISTANCE (per version)
    # --------------------------------------------------

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    fig_all, (ax_e_all, ax_f_all) = plt.subplots(1, 2, figsize=(10, 5))

    ax_e_all.set_xlabel("Distance [Å]")
    ax_e_all.set_ylabel("Energy error [eV]")
    ax_e_all.grid(True)

    ax_f_all.set_xlabel("Distance [Å]")
    ax_f_all.set_ylabel("Force error [eV/Å]")
    ax_f_all.grid(True)

    for ax, (path, data), v in zip(axes, models, versions):
        d, E, F = data.T

        ax.scatter(d, E, s=6)
        ax.set_title(f"v{v}")
        ax.set_xlabel("Distance [Å]")
        ax.set_ylabel("Energy error [eV]")
        ax.grid(True)

        ax_e_all.scatter(d, E, s=2, alpha=0.15, color="blue")

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_errors.pdf"))
    plt.close()

    # --------------------------------------------------
    # FORCE vs DISTANCE (per version)
    # --------------------------------------------------

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for ax, (path, data), v in zip(axes, models, versions):
        d, E, F = data.T

        ax.scatter(d, F, s=6)
        ax.set_title(f"v{v}")
        ax.set_xlabel("Distance [Å]")
        ax.set_ylabel("Force error [eV/Å]")
        ax.grid(True)

        ax_f_all.scatter(d, F, s=2, alpha=0.15, color="red")

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "force_errors.pdf"))
    plt.close()

    # --------------------------------------------------
    # Draw quantile bars on overlapped plots
    # --------------------------------------------------

    bar_width = 0.015 * (dist.max() - dist.min())

    # Energy quantiles
    for q, lw in zip(E_q, [1, 2, 1]):
        for x, y in zip(dist, q):
            ax_e_all.hlines(
                y,
                x - bar_width,
                x + bar_width,
                color="black",
                linewidth=lw,
                alpha=0.85,
            )

    # Force quantiles
    for q, lw in zip(F_q, [1, 2, 1]):
        for x, y in zip(dist, q):
            ax_f_all.hlines(
                y,
                x - bar_width,
                x + bar_width,
                color="black",
                linewidth=lw,
                alpha=0.85,
            )

    fig_all.tight_layout()
    fig_all.savefig(os.path.join(out_dir, "overlapped_errors.pdf"))
    plt.close(fig_all)

    # --------------------------------------------------
    # Save quantiles to file
    # --------------------------------------------------

    out_file = os.path.join(out_dir, "errors_quantiles_energy_force.txt")

    with open(out_file, "w") as f:
        f.write("# distance  E_q05  E_q50  E_q95  F_q05  F_q50  F_q95\n")
        for i in range(len(dist)):
            f.write(
                f"{dist[i]:.8f}  "
                f"{E_q[0,i]:.8e}  {E_q[1,i]:.8e}  {E_q[2,i]:.8e}  "
                f"{F_q[0,i]:.8e}  {F_q[1,i]:.8e}  {F_q[2,i]:.8e}\n"
            )

    print(f"Saved quantiles to {out_file}")


if __name__ == "__main__":
    main()
