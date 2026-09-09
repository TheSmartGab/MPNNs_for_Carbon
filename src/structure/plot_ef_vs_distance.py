import os
from argparse import ArgumentParser
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser(
        "Scan input_dir for all files with extension --ext, read them with ASE "
        "assuming format --format, and produce an energy vs. distance plot. "
        "Input structures must contain exactly two atoms."
    )

    parser.add_argument("input_dir", type=str, help="Directory to search")
    parser.add_argument("--ext", type=str, default="xyz",
                        help="File extension to search for (default: xyz)")
    parser.add_argument("--format", type=str, default=None,
                        help="ASE format override (default: autodetect)")
    parser.add_argument("--output_path", type=str,
                        help="Output path for plots")

    args = parser.parse_args()

    distances = []
    energies = []
    forces = []

    # Collect all matching files
    for fname in sorted(os.listdir(args.input_dir)):
        if fname.lower().endswith("." + args.ext.lower()):
            path = os.path.join(args.input_dir, fname)

            images = read(path, format=args.format, index=":")
            for atoms in images:
                if len(atoms) != 2:
                    raise ValueError(f"File {fname} does not contain exactly 2 atoms.")

                # distance between atom 0 and atom 1
                d = atoms.get_distance(0, 1)

                # energy per atom (ASE total energy is usually total; divide by 2)
                e = atoms.get_potential_energy() / len(atoms)
                f = np.linalg.norm(atoms.get_forces()[0])

                distances.append(d)
                energies.append(e)
                forces.append(f)

    distances = np.array(distances)
    energies = np.array(energies)
    forces = np.array(forces)

    # Sort by distance
    order = np.argsort(distances)
    distances = distances[order]
    energies = energies[order]
    forces = forces[order]

    # Determine y-limits: from minimum up to +25 eV above
    emin = energies.min()
    ymax = emin + 25.0

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(distances, energies, "-o")
    plt.xlabel(r"distance [$\AA$]")
    plt.ylabel("potential energy [eV/atom]")
    plt.ylim(emin-5, ymax)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, "e.pdf"))
    print(f"Saved plot to {os.path.join(args.output_path, 'e.pdf')}")

    # plot forces
    plt.figure(figsize=(6, 4))
    plt.plot(distances, forces, "-o")
    plt.xlabel(r"distance [$\AA$]")
    plt.ylabel(r"Force [$eV/\AA$]")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, "f.pdf"))
    print(f"Saved plot to {os.path.join(args.output_path, 'f.pdf')}")
    
    return 0


if __name__ == "__main__":
    main()

