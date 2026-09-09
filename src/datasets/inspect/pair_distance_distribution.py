import numpy as np
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
from os import path

from ase.io import read, write

from tqdm import tqdm

def main():
    """Plot pair distance distribution from extxyz file"""

    parser = ArgumentParser(description="Plot pair distance distribution from extxyz file")
    parser.add_argument("input_file", type=str, help="Input extxyz file")

    parser.add_argument("--n_bins", type=int, default=100, help="Number of bins for histogram")
    parser.add_argument("--max_distance", type=float, default=10.0, help="Maximum distance for histogram")

    args = parser.parse_args()
    dirname = args.input_file.rsplit("/", 1)[0] if "/" in args.input_file else "."

    # Read the atomic configurations from the extxyz file
    atoms_list = read(args.input_file, index=":")
    all_distances = []
    for atoms in tqdm(atoms_list):
        positions = atoms.get_positions()
        num_atoms = len(positions)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = np.linalg.norm(positions[i] - positions[j])
                all_distances.append(distance)
    all_distances = np.array(all_distances)
    # Compute histogram
    hist, bin_edges = np.histogram(all_distances, bins=args.n_bins, range=(0, args.max_distance))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Plot histogram
    plt.figure(figsize=(8, 6))
    bar = plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], align='center', alpha=0.7)
    data_values = bar.datavalues
    plt.xlabel("Pair Distance")
    plt.ylabel("Frequency")
    plt.title("Pair Distance Distribution")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path.join(dirname, "pair_distance_distribution.pdf"))
    with open(path.join(dirname, "pair_distance_distribution.txt"), "w") as f:
        f.write("# Pair Distance Distribution\n")
        f.write("# Bin_Center\tFrequency\n")
        for center, value in zip(bin_centers, data_values):
            f.write(f"{center:.6f}\t{value}\n")

    plt.show()



    return

if __name__ == "__main__":
    main()
