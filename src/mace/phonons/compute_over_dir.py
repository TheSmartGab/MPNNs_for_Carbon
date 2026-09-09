#!/usr/bin/env python
import os
import argparse
import subprocess
from pathlib import Path
from os import walk
from os.path import join

# -----------------------------------------------------------------------------
# System definitions
# -----------------------------------------------------------------------------
SYSTEMS = {
    "graphene": {
        "input": "src/mace/phonons/INPUTS/single_graphene.data",
        "format": "lammps-data",
        "kpath": "GMKG",
        "mask": "1 1 0 0 0 0",
        "output_dir": "phonons_graphene",
    },
    "diamond": {
        "input": "src/mace/phonons/INPUTS/diamond.data",
        "format": "lammps-data",
        "kpath": "GXWKGLUWLK",
        "mask": "1 1 1 0 0 0",
        "output_dir": "phonons_diamond",
    },
}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Driver for MACE phonon calculations"
    )

    p.add_argument("--root", required=True, help="Root directory containing MACE .model files")
    p.add_argument("--script", required=True, help="Path to phonon calculation script")
    p.add_argument("--model_name", required=False, help="name of models to look for", default="MACE.model")

    p.add_argument(
        "--systems",
        nargs="+",
        required=True,
        help="Systems to compute (e.g. graphene diamond)",
    )

    p.add_argument(
        "--supercells",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Supercells flattened per system. "
            "Example: --systems graphene diamond "
            "--supercells 25 25 1 8 8 8"
        ),
    )

    p.add_argument(
        "--fmax",
        nargs="+",
        type=float,
        default=None,
        help="Force convergence per system (default: 1e-5)",
    )

    p.add_argument(
        "--heads",
        nargs="+",
        default=None,
        help="MACE head per system (default: Default)",
    )

    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--delta", type=float, default=0.01)

    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def chunk_supercells(supercells, nsystems):
    if len(supercells) != 3 * nsystems:
        raise ValueError(
            f"Expected {3 * nsystems} integers for supercells, got {len(supercells)}"
        )
    return [
        tuple(supercells[i * 3 : (i + 1) * 3]) for i in range(nsystems)
    ]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    root = Path(args.root).resolve()
    script = Path(args.script).resolve()

    systems = args.systems
    nsys = len(systems)

    supercells = chunk_supercells(args.supercells, nsys)

    fmax_list = args.fmax if args.fmax is not None else [1e-5] * nsys
    if len(fmax_list) == 1:
        fmax_list *= nsys

    heads = args.heads if args.heads is not None else ["Default"] * nsys
    if len(heads) == 1:
        heads *= nsys

    if not (len(fmax_list) == len(heads) == nsys):
        raise ValueError("Mismatch in number of systems, fmax, or heads")

    # -------------------------------------------------------------------------
    # Loop over systems
    # -------------------------------------------------------------------------
    for name, sc, fmax, head in zip(systems, supercells, fmax_list, heads):
        if name not in SYSTEMS:
            raise KeyError(f"System '{name}' not defined in SYSTEMS dictionary")

        cfg = SYSTEMS[name]

        for root, dirs, files in walk(args.root):

            if args.model_name in files:
                model_path = join(root, args.model_name)
            
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"Missing model: {model_path}")

                output_dir = join(root,cfg["output_dir"])
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n[INFO] Running system: {name} with model: {model_path}")
                print(f"       supercell = {sc}")
                print(f"       fmax      = {fmax}")
                print(f"       head      = {head}")

                cmd = [
                    "python3",
                    str(script),
                    "--input", cfg["input"],
                    "--format", cfg["format"],
                    "--model", str(model_path),
                    "--head", head,
                    "--output_dir", str(output_dir),
                    "--device", args.device,
                    "--fmax", str(fmax),
                    "--delta", str(args.delta),
                    "--kpath", cfg["kpath"],
                    "--supercell", *map(str, sc),
                    "--mask", *cfg["mask"].split(),
                ]

                try:
                    subprocess.run(cmd, check=True)
                    print(f"[SUCCESS] {name} finished successfully")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] {name} failed")
                    print(f"        Command: {' '.join(cmd)}")
                    print(f"        Return code: {e.returncode}")
                    continue



if __name__ == "__main__":
    main()
