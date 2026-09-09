from argparse import ArgumentParser
from ase.io import read
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np  # Imported numpy for easy indexing


def parse_args():
    parser = ArgumentParser(description="Plot dimer energy vs distance from file of configurations")
    parser.add_argument("--inputfile", "-i", type=str, required=True, help="Input ASE readable file")
    parser.add_argument("--E0", type=float)
    return parser.parse_args()


def plot(ds, es, min_d=None, min_e=None, binding_energy=0):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter plot of data points
    ax.scatter(ds, es, color="#1f77b4", edgecolor="k", zorder=3, label="Data")
    
    # If we have a lot of points, a smooth line helps guide the eye
    # sorted_indices = np.argsort(ds)
    # ax.plot(np.array(ds)[sorted_indices], np.array(es)[sorted_indices], color="#1f77b4", alpha=0.5, zorder=2)

    # Highlight the minimum energy point if provided
    if min_d is not None and min_e is not None:
        ax.scatter(min_d, min_e, color="#d62728", edgecolor="k", s=50, zorder=4, label=f"Min: {min_d:.2f} Å,\nbinding energy: {binding_energy:.3f}")        
        ax.axvline(min_d, color="#d62728", linestyle=":", alpha=0.7, zorder=2)
        ax.axhline(min_e, color="#d62728", linestyle=":", alpha=0.7, zorder=2)

    # Academic/Presentation styling
    ax.set_xlabel(r"$d$ [$\mathrm{\AA}$]")
    ax.set_ylabel(r"$V$ [eV]")
    
    ax.tick_params(axis='both', which='major')
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc="best")

    return fig, ax


def main():
    args = parse_args()

    configs = read(args.inputfile, index=":")

    ds = []
    es = []

    for c in configs:
        if len(c) != 2:
            print(f"[WARNING] Skipping frame: number of atoms is {len(c)} (expected 2 for dimer)")
            continue

        ds.append(c.get_distance(0, 1))
        
        try:
            es.append(c.get_potential_energy())
        except RuntimeError:
            print("[WARNING] Frame missing calculated potential energy. Skipping.")
            continue

    if not ds:
        print("[ERROR] No valid dimer configurations with energies found.")
        return 1

    # --- Data Analysis Section ---
    # Convert to numpy arrays for easier logical indexing
    ds = np.array(ds)
    es = np.array(es)

    # 1. Find minimum energy and its corresponding distance
    min_idx = np.argmin(es)
    min_energy = es[min_idx]
    min_distance = ds[min_idx]

    # 2. Find energy at the largest distance
    max_dist_idx = np.argmax(ds)
    energy_at_max_dist = es[max_dist_idx]
    max_distance = ds[max_dist_idx]

    # 3. Calculate binding energy
    binding_energy = -min_energy + args.E0 * 2

    # Print results to console
    print("\n" + "="*40)
    print(f"Dimer Analysis Results:")
    print(f"  Minimum Energy:          {min_energy:.6f} eV")
    print(f"  Equilibrium Distance:    {min_distance:.4f} Å")
    print(f"  Energy at Max Distance:  {energy_at_max_dist:.6f} eV (at {max_distance:.4f} Å)")
    print(f"  Binding Energy:          {binding_energy:.6f} eV")
    print("="*40 + "\n")
    # -----------------------------

    # Pass the minimum points to the plot function to visualize it
    fig, ax = plot(ds, es, min_d=min_distance, min_e=min_energy, binding_energy=binding_energy)

    fig.tight_layout()
    outpath = Path(args.inputfile).with_suffix(".pdf")
    fig.savefig(outpath, bbox_inches="tight", transparent=True)
    print(f"[INFO] Plot successfully saved to {outpath}")

    return 0


if __name__ == "__main__":
    main()