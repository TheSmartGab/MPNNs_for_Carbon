#!/usr/bin/env python
"""Compute phonon dispersion using a trained MACE model.

This script relaxes an input atomic structure using the MACE (Moment Map Atomic Cluster Expansion)
interatomic potential, then computes phonon dispersions and density of states (DOS) using ASE's
Phonons module. It generates band structures and DOS plots saved as PDF files.

Functions:
    parse_args: Parse command-line arguments for the computation script
    main: Main entry point to run relaxation, phonon calculation, and plotting
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.optimize import BFGS
from ase.phonons import Phonons
from ase.filters import ExpCellFilter

from ase import Atoms

import json

from mace.calculators import MACECalculator

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for phonon dispersion computation.

    Returns:
        Namespace: Parsed arguments with input, model, output_dir, and other settings.
    """
    p = argparse.ArgumentParser(
        description="Relax structure with MACE and compute phonons + DOS"
    )

    p.add_argument("--input", required=True, help="Input structure file (extxyz, cif, vasp, ...)")
    p.add_argument("--format", required=True, help="Format of input data (extxyz, cif, vasp, lammps-data)")
    p.add_argument("--model", required=True, help="Path to MACE .model file")
    p.add_argument("--head", required=False, type=str, default="Default",
                   help="Head to use. Models can be trained with multi head to include multiple levels of theory. After multi-head fine-tuning, you have available 'pt_head' (the pretrained head which is also trained during fine tuning using the replay set), and 'Default' (or whatever other heads you used) after training.")
    p.add_argument("--output_dir", required=True, help="Directory to store output files")

    p.add_argument("--device", default="cpu", help="Device to use: cpu | cuda | mps")

    # relaxation
    p.add_argument("--fmax", type=float, default=0.00001, help="Force convergence criterion (eV/Å)")
    p.add_argument("--mask", type=bool, required=False, nargs=6, default=True,
                   help="Mask to pass to the ExpCellFilter. True indicates a component of the stress tensor to relax; False to ignore. Voigt notation is used: [xx yy zz yz xz xy]")

    # phonons
    p.add_argument("--supercell", nargs=3, type=int, default=[8, 8, 8], help="Supercell dimensions for phonon calculation")
    p.add_argument("--delta", type=float, default=0.01, help="Displacement amplitude in Å")
    p.add_argument("--kpath", type=str, default="GMKG",
                   help="Path in k-space (e.g., GMKG for hexagonal, GXWKGLUWLK for diamond, GXUGL for diamond minimal)")

    args = p.parse_args()

    return args


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """Main entry point for MACE phonon dispersion computation.

    This function:
      1. Loads the input atomic structure and MACE model
      2. Relaxes the structure using BFGS with ExpCellFilter
      3. Computes phonon dispersions using ASE Phonons module
      4. Generates band structure and density of states plots
      5. Saves relaxed structure, lattice parameters, and plots to output directory

    Returns:
        None (saves files to output_dir)
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "phonon_calculation_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # -----------------------
    # Load structure & model
    # -----------------------
    atoms: Atoms = read(args.input, format=args.format)

    calc = MACECalculator(
        model_path=args.model,
        device=args.device,
        head = args.head
    )
    atoms.calc = calc

    # -----------------------
    # Relaxation
    # -----------------------
    print("[INFO] Starting relaxation")
    ecf = ExpCellFilter(atoms, mask=args.mask)
    dyn = BFGS(ecf, trajectory=os.path.join(args.output_dir, "relax.traj"))

    dyn.run(fmax=args.fmax)

    # -----------------------
    # Save relaxed structure
    # -----------------------
    relaxed_path = os.path.join(args.output_dir, "relaxed.extxyz")
    write(relaxed_path, atoms)

    # lattice info
    a, b, c = atoms.cell.lengths()
    with open(os.path.join(args.output_dir, "lattice.out"), "w") as f:
        f.write(f"a = {a:.6f} Å\n")
        f.write(f"b = {b:.6f} Å\n")
        f.write(f"c = {c:.6f} Å\n")

    # -----------------------
    # Phonons
    # -----------------------
    print("[INFO] Starting phonons calculations")
    ph = Phonons(
        atoms,
        calc,
        supercell=tuple(args.supercell),
        delta=args.delta
    )
    print("[INFO] ph.run")
    ph.run()
    print("[INFO] ph.cache:")
    print(ph.cache)
    print("[INFO] Checking phonon cache entries")

    print("[INFO] ph.read")
    ph.read(acoustic=True)
    ph.clean()

    # band path (graphene-friendly default)
    path = atoms.cell.bandpath(args.kpath, npoints=200)
    bs = ph.get_band_structure(path)
    with open(os.path.join(args.output_dir, "band.json"), "w") as f:
        bs.write(f)

    dos = ph.get_dos(kpts=(30, 30, 1)).sample_grid(npts=200, width=1e-3)

    # -----------------------
    # Plot
    # -----------------------
    print("[INFO] Plotting")
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_axes([0.12, 0.07, 0.67, 0.85])

    emax = bs.energies.max()
    bs.plot(ax=ax, emin=0.0, emax=emax)

    dosax = fig.add_axes([0.8, 0.07, 0.17, 0.85])
    dosax.fill_between(
        dos.get_weights(),
        dos.get_energies(),
        y2=0,
        color="grey",
        edgecolor="k",
        lw=1,
    )
    dosax.set_ylim(0, emax)
    dosax.set_yticks([])
    dosax.set_xticks([])
    dosax.set_xlabel("DOS")

    print("saving fig to", os.path.join(args.output_dir, "phonons.pdf"))
    fig.savefig(os.path.join(args.output_dir, "phonons.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
