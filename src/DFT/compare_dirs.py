"""Compare energies and forces across DFT calculation directories.

This module provides utilities to compare reference and calculated atomic structures,
validating that calculations are reproducible or comparing different computational setups.
It generates histograms of energy per atom differences and force component differences (Fx, Fy, Fz).

Functions:
    compare: Compare energies and forces between reference and calculation directories
    hist_array: Create histogram plots from numpy arrays
    main: CLI entry point for the comparison script
"""

from argparse import ArgumentParser
from ase.io import read
from ase import Atoms
import os
import numpy as np
import matplotlib.pyplot as plt


def compare(
    ref_dir, references: list[str], calc_dir, calculations: list[str], format: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compare energies and forces between reference and calculation directories.

    This function reads pairs of atomic structures from reference and calculation directories,
    computing the per-atom energy difference and force differences for each structure pair.

    Args:
        ref_dir: String path to the directory containing reference data files.
        references: List of reference filename strings.
        calc_dir: String path to the directory containing calculated data files.
        calculations: List of calculation filename strings (should match references).
        format: String specifying the ASE file format to use for reading (e.g., 'extxyz').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - de: 2D array of shape (N, 1) with per-atom energy differences (calc - ref).
            - df: 2D array of shape (N, 3) with force component differences (Fx, Fy, Fz).
    """
    # note that files are being stored in lists. This is fine as long as files are stored in the same directory. consider the possibility for duplicate names if files are stored in several dirs
    de = np.empty((0,1))
    df = np.empty((0,3))
    for ref in references:
        try:
            calc_index = calculations.index(ref)
        except ValueError as e:
            print(f"[WARNING] reference {ref} had no matching calculation. skiping.")
            continue

        calculations.pop(
            calc_index
        )  # this will progressively shorten the list making successive .index calls faster

        ref_data = read(os.path.join(ref_dir, ref), ":", format=format)
        calc_data = read(os.path.join(calc_dir, ref), ":", format=format)

        if len(ref_data) != len(calc_data):
            print(f"[WARNING] Images length mismatch between reference and calculation for {ref}. ref has {len(ref_data)}, calc has {len(calc_data)} elements, skip.")
            continue

        for atoms_ref, atoms_calc in zip(ref_data, calc_data):
            de = np.vstack((de, (atoms_calc.get_potential_energy() - atoms_ref.get_potential_energy())  / len(atoms_calc)))
            df = np.vstack((df, atoms_calc.get_forces() - atoms_ref.get_forces()))

        # print("="*50)
        # print(de)
        # print("="*50)
        # print(df)

    return de, df

def hist_array(arr, output_file, xlabel, title = "", bins =30):
    """Create a histogram plot from a numpy array and save to file.

    Args:
        arr: 1D numpy array of data values to plot.
        output_file: String path for the output file (e.g., 'energy.pdf').
        xlabel: String label for the x-axis.
        title: Optional string title for the plot.
        bins: Integer number of histogram bins (default 30).
    """
    plt.hist(arr, bins)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("counts")

    plt.savefig(output_file)
    plt.close()


def main():
    """CLI entry point for comparing energies and forces across directories.

    This function parses command-line arguments, reads files from reference and calculation
    directories, computes the differences, and generates histogram plots of energy per atom
    and force component (Fx, Fy, Fz) differences saved as PDF files in the output directory.
    """

    parser = ArgumentParser(
        description="Compare forces and energies in 2 directories. Useful to control you are able to reproduce calculations found in datasets. The filenames in the 2 directories should match exactly, although this may be generalised at need. ase is used to read files, so the format should be ase compatible. This produces histograms with differences of computed energies and forces."
    )

    parser.add_argument("references", type=str, help="directory with reference data.")
    parser.add_argument(
        "calculations", type=str, help="directory with new calculations to compare."
    )
    parser.add_argument(
        "--extension",
        required=False,
        type=str,
        default="extxyz",
        help="extension of datafiles",
    )
    parser.add_argument(
        "--format",
        required=False,
        type=str,
        default="extxyz",
        help="ase format to read data",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory where to store output data and plots",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reference_files = [
        f for f in os.listdir(args.references) if f.endswith("." + args.extension)
    ]
    calculation_files = [
        f for f in os.listdir(args.calculations) if f.endswith("." + args.extension)
    ]

    print(f"[INFO] Found {len(reference_files)} reference files")
    print(f"[INFO] Found {len(calculation_files)} calculation files")

    de, df = compare(
        args.references,
        reference_files,
        args.calculations,
        calculation_files,
        args.format,
    )

    hist_array(de[:,0], os.path.join(args.output_dir, "energy_per_atom.pdf"), "eV/atom")
    for i, coord in enumerate(["Fx", "Fy", "Fz"]):
        hist_array(df[:,i], os.path.join(args.output_dir, coord+".pdf"), xlabel=coord+r" $eV/\AA$")

    return 0


if __name__ == "__main__":
    main()
