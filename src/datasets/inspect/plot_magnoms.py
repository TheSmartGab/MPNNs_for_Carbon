# plot magnoms from extxyz files
# note that I had to change line 264 in ase/io/espresso.py to correctely read magnoms from qe-7.2 output files

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read


def main():
    
    parser = argparse.ArgumentParser(
        description="Plot magnetic moments from extxyz file into a single PDF."
    )

    parser.add_argument("input_file", type=str,
                        help="Input extxyz file containing magnetic moments.")
    parser.add_argument("--output_file", type=str, default="magnoms_plot.pdf",
                        help="Output PDF file.")
    parser.add_argument("--format", type=str, required=False, default='extxyz', help="input file format. may be different than extension.")

    args = parser.parse_args()


    images = read(args.input_file, index=":", format=args.format)
    magnoms = [img.get_magnetic_moments() for img in images]
    magnoms = np.array(magnoms)  # shape (n_images, n_atoms)
    distances = [img.get_all_distances()[0][1] for img in images]
    sorted_indices = np.argsort(distances)


    sorted_distances = np.array([distances[i] for i in sorted_indices])
    sorted_magnoms = np.array([magnoms[i] for i in sorted_indices])
    n_images, n_atoms = magnoms.shape

    fig, ax = plt.subplots(figsize=(6, 4))
    for atom_idx in range(n_atoms):
        ax.plot(sorted_distances, sorted_magnoms[:, atom_idx], "-o", markersize=2, label=f'Atom {atom_idx+1}')
        ax.set_title('Magnetic Moments vs Distance')
        ax.set_xlabel(f'Distance [$\AA$]')
        ax.set_ylabel('Magnom')

    ax.plot(sorted_distances, np.sum(sorted_magnoms, axis=1), 'k--x', markersize=2, label='Total Magnom')

    # Major ticks every 0.5 (with labels)
    ax.set_xticks(np.arange(0, 5.1, 0.5))

    # Minor ticks every 0.1 (no labels)
    ax.set_xticks(np.arange(0, 5.1, 0.1), minor=True)
    ax.grid(True, which='major', linewidth=1)
    ax.grid(True, which='minor', linewidth=0.5, alpha=0.4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.output_file)
    print(f"Saved PDF to {args.output_file}")
    
    return 0


if __name__ == "__main__":
    main()
