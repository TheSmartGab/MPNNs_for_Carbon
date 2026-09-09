# =====================================
# Script Name: compute_strains.py
# Purpose: Compute energy of bulk graphene with various applied strains
# Usage: Run with --configs, --relaxed_config, --output_dir, and --model arguments
#        to evaluate MACE model predictions on strained graphene configurations.
# =====================================

"""ASE-based workflow script for computing energies of strained graphene configurations.

This script reads a JSON file containing a list of dictionaries. Dictionaries with 'strain'
key are used to compute the energy of bulk graphene with that specific strain applied.
The script uses MACE calculators via ASE to evaluate energies and forces for each strained
configuration, supporting research on graphene elastic properties and MLIP accuracy under deformation.
"""

from argparse import ArgumentParser
import os
import numpy as np
import ase
from ase.build import surface
import json
import mace
from mace.calculators import MACECalculator
from ase.io import read, write


def parse_args():
    """Parse command line arguments for strain computation script.

    Returns:
        argparse.Namespace: Parsed arguments including configs, relaxed_config, output_dir, model, device
    """

    parser = ArgumentParser("Compute energy for given .json file strain")

    parser.add_argument("--configs", help="json file with configurations to compute")
    parser.add_argument("--relaxed_config", help="input file with relaxed configuration (ase readable)")
    parser.add_argument("--output_dir")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu", required=False)


    args = parser.parse_args()
    return args


def strain(atoms, strain):
    """Apply uniform strain to an ASE atoms object.

    Args:
        atoms (ase.Atoms): Original atomic structure
        strain (float): Strain factor to apply (e.g., 0.05 for +5% strain)

    Returns:
        ase.Atoms: Strained atomic structure with scaled positions and cell
    """
    factor = 1.+strain

    atoms_copy = atoms.copy()
    atoms_copy.positions*=factor
    atoms_copy.cell*=factor

    return atoms_copy

def main():

    args = parse_args()

    with open(args.configs, "r") as f:
        configs = json.load(f)

    relaxed_conf = read(args.relaxed_config)
    outdir = args.output_dir

    os.makedirs(outdir, exist_ok = True)

    calc = MACECalculator(args.model, device=args.device)
    summary_file = os.path.join(args.output_dir, 'strain_e.out')
    f = open(summary_file, 'w')
    f.write('#strain\tenergy')

    strains = [c['strain'] for c in configs]

    for s in strains:
        strained = strain(relaxed_conf, s)
        strained.calc = calc

        energy=strained.get_potential_energy()
        forces = strained.get_forces()
        stress = strained.get_stress()

        outfile = os.path.join(args.output_dir, f"{s}.extxyz")

        write(outfile, strained)

        string = str(s) + ' ' + str(energy) + '\n'
        f.write(string)

    f.close()

    return

if __name__ == "__main__":
    print(__file__, "main called")
    main()