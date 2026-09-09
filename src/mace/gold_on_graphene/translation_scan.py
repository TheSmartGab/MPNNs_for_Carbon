#!/usr/bin/env python3
"""
grid_scan.py — Grid sampling of molecular adsorption configurations using MACE.
Updated: Handles input files where Gold/Carbon are represented by indices 1 and 2.
"""

import argparse
import os
import sys
import logging

import numpy as np
from tqdm import tqdm

from ase.io import read, write
from ase.optimize import BFGS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------

class ConstrainGroupCM:
    def __init__(self, masks, groups_mask):
        self.masks = {k: np.array(v, dtype=float) for k, v in masks.items()}
        self.groups_mask = np.array(groups_mask)

    def adjust_positions(self, atoms, newpositions):
        masses = atoms.get_masses()
        positions = atoms.get_positions()

        for cls, mask_vec in self.masks.items():
            atom_mask = self.groups_mask == cls
            if not np.any(atom_mask):
                continue

            grp_masses = masses[atom_mask]
            total_mass = grp_masses.sum()
            if total_mass == 0:
                continue

            step = newpositions[atom_mask] - positions[atom_mask]
            com_step = np.einsum("i,ij->j", grp_masses, step) / total_mass
            newpositions[atom_mask] -= com_step[np.newaxis, :] * mask_vec[np.newaxis, :]

    def adjust_forces(self, atoms, forces):
        masses = atoms.get_masses()

        for cls, mask_vec in self.masks.items():
            atom_mask = self.groups_mask == cls
            if not np.any(atom_mask):
                continue

            grp_masses = masses[atom_mask]
            total_mass = grp_masses.sum()
            if total_mass == 0:
                continue

            total_force = forces[atom_mask].sum(axis=0)
            com_accel = total_force / total_mass
            forces[atom_mask] -= (
                (com_accel * mask_vec)[np.newaxis, :] * grp_masses[:, np.newaxis]
            )


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def run_grid_sampling(
    atoms_input,
    model,
    outdir,
    n_grid=5,
    fmax_relax=0.01,
    divide=5,
    max_steps=1000,
    au_symbol="Au",
    c_symbol="C",
):
    # Index setup - Now searching for strings like "Au" and "C"
    AuMask = [a.index for a in atoms_input if a.symbol == au_symbol]
    CMask  = [a.index for a in atoms_input if a.symbol == c_symbol]

    if not AuMask:
        raise ValueError(f"No atoms with symbol '{au_symbol}' found. Check input mapping.")
    if not CMask:
        raise ValueError(f"No atoms with symbol '{c_symbol}' found. Check input mapping.")

    log.info("Substrate (%s): %d atoms | Adsorbate (%s): %d atoms",
             au_symbol, len(AuMask), c_symbol, len(CMask))

    groups_mask = np.full(len(atoms_input), -1)
    groups_mask[AuMask] = 0
    groups_mask[CMask]  = 1

    COM_masks = {
        0: [1, 1, 0],   # substrate COM: constrain XY, free Z
        1: [1, 1, 1],   # adsorbate COM: constrain XYZ
    }

    cell = atoms_input.get_cell()
    v1 = cell[0] / divide
    v2 = cell[1] / divide
    grid_points = np.linspace(0, 1, n_grid, endpoint=False)

    dir_traj  = os.path.join(outdir, "TRAJECTORIES")
    dir_relax = os.path.join(outdir, "RELAXATIONS")
    os.makedirs(dir_traj,  exist_ok=True)
    os.makedirs(dir_relax, exist_ok=True)

    convergences = []
    total = n_grid * n_grid

    for i in tqdm(grid_points, desc="grid i"):
        for j in tqdm(grid_points, desc="grid j", leave=False):
            work_atoms = atoms_input.copy()
            work_atoms.calc = model

            target_com = i * v1 + j * v2
            target_com[2] = 0.0

            au_masses = work_atoms.get_masses()[AuMask]
            current_com = (
                np.einsum("i,ij->j", au_masses, work_atoms.positions[AuMask])
                / au_masses.sum()
            )

            shift = target_com - current_com
            shift[2] = 0.0
            work_atoms.positions[AuMask] += shift

            con = ConstrainGroupCM(COM_masks, groups_mask)
            work_atoms.set_constraint(con)

            label = f"i{i:.4f}_j{j:.4f}"
            outtraj    = os.path.join(dir_traj,  label + ".traj")
            outrelaxed = os.path.join(dir_relax, label + ".extxyz")

            dyn = BFGS(work_atoms, trajectory=outtraj, logfile=None)
            converged = dyn.run(fmax=fmax_relax, steps=max_steps)
            convergences.append(converged)

            write(outrelaxed, work_atoms, format="extxyz")

    convergences = np.array(convergences)
    np.save(os.path.join(outdir, "converged.npy"), convergences)
    log.info("Scan complete: %d/%d grid points converged.", convergences.sum(), total)


def parse_args():
    parser = argparse.ArgumentParser(description="2-D lateral grid scan with MACE.")
    
    # --- Required ---
    req = parser.add_argument_group("required arguments")
    req.add_argument("--model", "-m", required=True, help="Path to MACE model.")
    req.add_argument("--input", "-i", required=True, help="Path to initial structure.")
    req.add_argument("--outdir", "-o", required=True, help="Output directory.")

    # --- Parameters ---
    parser.add_argument("--n-grid", "-n", type=int, default=5)
    parser.add_argument("--divide", "-d", type=int, default=5)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--au-symbol", default="Au", help="Symbol to map from index 1.")
    parser.add_argument("--c-symbol", default="C", help="Symbol to map from index 2.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--index", default="-1")

    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.isfile(args.model) or not os.path.isfile(args.input):
        log.error("Files not found.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    log.info("Reading structure...")
    atoms = read(args.input, index=args.index)
    if isinstance(atoms, list): atoms = atoms[0]

    # ------------------------------------------------------------------
    # SPECIES MAPPING LOGIC
    # ------------------------------------------------------------------
    # Force atoms read as type 1 to Au and type 2 to C
    mapping = {1: args.au_symbol, 2: args.c_symbol}
    new_symbols = []
    
    for atom in atoms:
        if atom.number in mapping:
            new_symbols.append(mapping[atom.number])
        else:
            new_symbols.append(atom.symbol)
            
    # set_chemical_symbols updates symbols, atomic numbers, and masses
    atoms.set_chemical_symbols(new_symbols)
    log.info("Mapped species: 1->%s, 2->%s", args.au_symbol, args.c_symbol)
    log.info("Formula after mapping: %s", atoms.get_chemical_formula())

    # Load MACE
    try:
        from mace.calculators import MACECalculator
        model = MACECalculator(args.model, device=args.device)
    except Exception as exc:
        log.error("MACE failed: %s", exc); sys.exit(1)

    run_grid_sampling(
        atoms_input=atoms,
        model=model,
        outdir=args.outdir,
        n_grid=args.n_grid,
        fmax_relax=args.fmax,
        divide=args.divide,
        max_steps=args.max_steps,
        au_symbol=args.au_symbol,
        c_symbol=args.c_symbol,
    )

if __name__ == "__main__":
    main()