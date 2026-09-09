import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator

def get_zbl_potential(r, z1, z2):
    """Calculates the ZBL repulsion potential in eV."""
    # Constants
    e_squared = 14.3996  # eV * Å
    a_0 = 0.529177       # Bohr radius in Å
    
    # Universal screening length
    a = 0.8854 * a_0 / (z1**0.23 + z2**0.23)
    x = r / a
    
    # Screening function phi(x)
    phi = (0.18175 * np.exp(-3.1998 * x) + 
           0.50986 * np.exp(-0.94229 * x) + 
           0.28022 * np.exp(-0.4029 * x) + 
           0.02817 * np.exp(-0.20162 * x))
    
    return (z1 * z2 * e_squared / r) * phi

def plot_ref(args, ax):
    """Plots the reference training, validation, and test data after strict species filtering."""
    print("[INFO] Gathering and filtering reference data...")

    SPLIT_KEYS = ["training", "validation", "test"]
    REF_SPLIT_DATA = {k: [] for k in SPLIT_KEYS}
    
    FILES = {
        k: os.path.join(args.ref_split, f"{k}", f"{k}.{args.ref_extension}")
        for k in SPLIT_KEYS
    }

    target_symbols = sorted(args.elements)
    colors = {"training": "#2ca02c", "validation": "#ff7f0e", "test": "#d62728"}

    for k in SPLIT_KEYS:
        species_suffix = f"_{''.join(target_symbols)}"
        penergy_file = os.path.join(args.ref_split, f"{k}", f"penergy{species_suffix}.dat")

        if not os.path.isfile(penergy_file):
            if os.path.isfile(FILES[k]):
                images = read(FILES[k], index=":", format=args.ref_format)
                if not images:
                    continue
                
                for idx, atoms in enumerate(images):
                    if len(atoms) != 2:
                        print(f"[WARNING] Split {k} frame {idx} skipped: Contains {len(atoms)} atoms (not a dimer).")
                        continue
                    
                    current_symbols = sorted(atoms.get_chemical_symbols())
                    if current_symbols != target_symbols:
                        continue

                    d = atoms.get_distance(0, 1)
                    try:
                        e = atoms.get_potential_energy()
                    except RuntimeError:
                        print(f"[WARNING] Split {k} frame {idx} has missing energy properties. Skipping.")
                        continue

                    REF_SPLIT_DATA[k].append([d, e])

                if len(REF_SPLIT_DATA[k]) == 0:
                    continue

                REF_SPLIT_DATA[k] = np.array(REF_SPLIT_DATA[k])
                k_sorted_idx = np.argsort(REF_SPLIT_DATA[k][:, 0])
                REF_SPLIT_DATA[k] = REF_SPLIT_DATA[k][k_sorted_idx]

                np.savetxt(penergy_file, REF_SPLIT_DATA[k])
        else:
            REF_SPLIT_DATA[k] = np.loadtxt(penergy_file)

        REF_SPLIT_DATA[k] = np.array(REF_SPLIT_DATA[k])
        if REF_SPLIT_DATA[k].size == 0 or len(REF_SPLIT_DATA[k].shape) < 2:
            continue
            
        ax.scatter(
            REF_SPLIT_DATA[k][:, 0], 
            REF_SPLIT_DATA[k][:, 1], 
            label=f"Ref {k.capitalize()}", 
            color=colors[k], 
            s=10, 
            zorder=3
        )


def main():
    parser = ArgumentParser(
        description="Compute and plot the diatomic potential energy curves using a MACE machine learning model combined with ZBL."
    )

    parser.add_argument("--model_path", type=str, required=True, help="path/to/model.model")
    parser.add_argument("--head", type=str, required=False, default="Default", help="head of the model to use.")
    parser.add_argument(
        "--elements",
        type=str,
        nargs=2,
        required=True,
        help="Atomic symbols of the diatomic molecule (e.g., Au C).",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="device to use (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output curves and info will be saved.",
    )
    parser.add_argument("--min_d", required=False, default=0.5, type=float, help="minimum distance")
    parser.add_argument("--max_d", required=False, default=6.0, type=float, help="maximum distance")
    parser.add_argument("--num", required=False, default=200, type=int, help="number of points to evaluate")

    parser.add_argument(
        "--ref_split",
        required=False,
        type=str,
        default=None,
        help="Reference split directory with training, validation, and test subfolders.",
    )
    parser.add_argument("--ref_extension", required=False, type=str, default="extxyz")
    parser.add_argument("--ref_format", required=False, type=str, default="extxyz")

    args = parser.parse_args()

    if not args.model_path.endswith(".model"):
        print("[ERROR] Model file must end in .model extension.")
        exit(-1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calculator = MACECalculator(model_path=args.model_path, device=args.device, head=args.head)
    atoms = Atoms(args.elements, calculator=calculator)
    z1, z2 = atoms.get_atomic_numbers()

    ds = np.linspace(args.min_d, args.max_d, args.num)
    energies = []

    for d in ds:
        atoms.positions = [[0, 0, 0], [d, 0, 0]]
        energies.append(atoms.get_potential_energy())

    energies = np.array(energies)
    min_idx = np.argmin(energies)
    min_e = energies[min_idx]
    min_d = ds[min_idx]
    last_e = energies[-1]

    zbl_energies = get_zbl_potential(ds, z1, z2)

    plot_points_path = output_path / "model_penergy.out"
    with open(plot_points_path, "w") as f:
        f.write("# Distance[A] MACE_Energy[eV] ZBL_Energy[eV]\n")
        for d, e, z in zip(ds, energies, zbl_energies):
            f.write(f"{d:.6f} {e:.6f} {z:.6f}\n")

    # Figure Configuration
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))

    ax.plot(ds, energies, label="MACE Potential Energy", color="#1f77b4", linewidth=1, zorder=4)
    ax.plot(ds, zbl_energies, label="ZBL Repulsion", color="#7f7f7f", linestyle="--", linewidth=2.0, zorder=2)
    ax.scatter(min_d, min_e, marker="x", color="#f90000", s=30, label=f"MACE Min ({min_d:.2f} Å)", zorder=5)

    if args.ref_split:
        plot_ref(args, ax)

    # Global Font Size Unified to 16
    ax.set_xlabel(r"Interatomic Distance, $d$ [$\mathrm{\AA}$]", fontsize=16, labelpad=10)
    ax.set_ylabel(r"Potential Energy, $E$ [eV]", fontsize=16, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    # Structural details
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim((min_e - 1.5, min_e + 35.0))
    ax.set_xlim((args.min_d, args.max_d))

    fig.tight_layout()

    svg_out = output_path / "potential_energy.svg"
    fig.savefig(svg_out, bbox_inches="tight", transparent=True)
    print(f"[INFO] Plot beautifully saved to {svg_out}")

    binding_energy = last_e - min_e
    info_out = output_path / "info.out"
    with open(info_out, "w") as f:
        f.write(f"Binding Energy (eV):\t{binding_energy:.6f}\n")
        f.write(f"Equilibrium Distance (A):\t{min_d:.6f}\n")
    print(f"[INFO] Properties metrics exported to {info_out}")

    return 0

if __name__ == "__main__":
    main()