import os
from ase.io import read
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

def plot(ls, pots, output_file):
    plt.plot(ls, pots)
    plt.title("")
    plt.xlabel(r"cell vector length $\AA$")
    plt.ylabel("potential energy eV/atom")
    plt.grid()
    plt.tight_layout()

    print(f"[INFO] saving to {output_file}")
    plt.savefig(output_file)


def convergence_index(arr, threshold=1e-8):
    val = arr[-1]
    diffs = np.abs(arr - val) <= threshold

    return np.argmax([np.all(diffs[i:]) for i in range(len(arr))])


def main():
    parser=ArgumentParser(
            """This script plots the energy vs the cell side length (for orthorombic cells) for output files somewhere in the root directory passed. os.walk is used, so make sure you have only relevant files with the given extension in the children of the root. Usefull to visualize convergence. For diatomic molecules one unit vector is actually longer. Pass --increases parameters as well to indicate eventual increases in lengths for some unit vectors. Then their length minus the corresponding increase is the value actually taken as lenth."""
            )
    parser.add_argument("root", type=str, help="root directory with data.")
    parser.add_argument("output_file", type=str, help="output file with plot")
    parser.add_argument("--format", type=str, required=False, default = "extxyz", help="data file format. must be a vali ase format")
    parser.add_argument("--ext", type=str, required=False, default="extxyz", help="data file extension")
    parser.add_argument("--increases", type=float, required = False, nargs=3, default=[0,0,0], help="use this argument if, in the converence, you have one or two primitive vectors that have a constante increase in their length wrt the other/s. Used, for example, for diatomic molecules, since it is reasonable to keep a longer length in the bond axis.")
    parser.add_argument("--convthr", type=float, required=False, default=1e-4, help="Threshold for energy convergence. The programm will find the first length for which the calculations differs from the last value by less than this valus per atom, it will also check that all following values are within this threshold. Note that this is an energy per atom.")


    args = parser.parse_args()
    walk=os.walk(args.root)

    ls = np.empty(shape=(0, 1))
    pots = np.empty(shape=(0, 1))

    for (root,dirs,files) in walk:
        for file in files:
            if file.endswith("."+args.ext):
                path=os.path.join(root, file)
                print(f"[INFO] data file found at {path}")
                images=read(path, index=":", format=args.format)

                for i, atoms in enumerate(images):
                    cell = atoms.get_cell()
                    lengths=cell.lengths()
                    # check all cell vectors have the same length
                    # ase provides cell.get_bravais_lattice, but a cubic lattice is CUB, not robohedral
                    # and both cases must be included. 
                    # use numpy instead
                    if not np.allclose(lengths - np.array(args.increases), lengths[0] - args.increases[0], atol=1e-8): # cell vectors are expected to be of order unity
                        print(f"[WARNING] file {path} have atoms with non rombohedral cell at index {i}. skip index")
                        continue
                    l = lengths[0] - args.increases[0]
                    pot = atoms.get_potential_energy() / len(atoms)

                    ls = np.vstack((ls, l))
                    pots = np.vstack((pots, pot))

    sorted_indexes = np.argsort(ls[:,0])
    ls = ls[sorted_indexes]
    pots = pots[sorted_indexes]

    plot(ls, pots, args.output_file)
    
    c_index = convergence_index(pots, args.convthr)
    if c_index >= len(pots) - 1: # only the last element is equal to itself
        print('[INFO] calculations did not converge. consider increasing cell sides.')
    else:
        print(f'convergence threshold: {args.convthr} eV/atom')
        print(f'calculations converged at distance {ls[c_index,0]}')
        print(f'converged energy: {pots[-1,0]}')

    
    return 0


if __name__ == "__main__":
    main()
