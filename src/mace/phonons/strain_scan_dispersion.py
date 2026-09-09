import os
import subprocess
import argparse
import numpy as np
import torch
from ase.io import read

def voigt_to_strain(v):
    """6-element Voigt vector to 3x3 symmetric strain tensor."""
    e1, e2, e3, e4, e5, e6 = v
    return np.array([
        [e1,       0.5 * e6, 0.5 * e5],
        [0.5 * e6, e2,       0.5 * e4],
        [0.5 * e5, 0.5 * e4, e3      ]
    ])

def main():
    parser = argparse.ArgumentParser(description="Strain scan via Coordinate System Rotation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--final_strain", nargs=6, type=float, required=True)
    parser.add_argument("--rotation", type=float, default=0.0, help="Rotation of the coordinate system in degrees")
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--python_script", required=True)
    parser.add_argument("--supercell", nargs=3, type=int, default=[10, 10, 1])
    parser.add_argument("--kpath", type=str, default="GMKG")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--mask", nargs=6, type=int, default=[0, 0, 0, 0, 0, 0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 1. Load the original, unrotated crystal
    base_atoms = read(args.input, format="lammps-data")
    original_cell = base_atoms.get_cell()

    # 2. Define the Coordinate Rotation Matrix R
    # This defines the orientation of the new axes x', y'
    theta = np.radians(args.rotation)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # 3. Transform the nominal strain into the new coordinate system
    # If we want to pull along x' and y', we take the 'nominal' strain 
    # and transform it back to the crystal's frame.
    eps_nominal = voigt_to_strain(np.array(args.final_strain))
    
    # Transformation: eps_crystal = R @ eps_nominal @ R.T
    # This is effectively "what does a strain in x'/y' look like to the crystal?"
    eps_crystal_frame = R @ eps_nominal @ R.T

    root_dir = f"coord_rot_{args.rotation}_strain_{'_'.join(map(str, args.final_strain))}"
    os.makedirs(root_dir, exist_ok=True)

    for i in range(args.steps):
        scale = i / (args.steps - 1) if args.steps > 1 else 1.0
        current_eps = eps_crystal_frame * scale
        
        # Deformation Gradient F = I + epsilon
        F = np.eye(3) + current_eps
        
        # Apply to the ORIGINAL lattice vectors
        # new_cell = original_cell @ F.T
        new_cell = original_cell @ F.T
        
        strained_atoms = base_atoms.copy()
        strained_atoms.set_cell(new_cell, scale_atoms=True)

        # Labeling for directories
        step_name = f"step_{i:02d}"
        step_dir = os.path.join(root_dir, step_name)
        os.makedirs(step_dir, exist_ok=True)

        print(f"[DRIVER] Step {i+1}/{args.steps} | Frame: {args.rotation} deg")

        tmp_input = os.path.join(step_dir, "strained_input.extxyz")
        strained_atoms.write(tmp_input)

        cmd = [
            "python", args.python_script,
            "--input", tmp_input,
            "--format", "extxyz",
            "--model", args.model,
            "--output_dir", step_dir,
            "--supercell", str(args.supercell[0]), str(args.supercell[1]), str(args.supercell[2]),
            "--kpath", args.kpath,
            "--device", args.device
        ]
        if args.optimize:
            cmd.append("--optimize")
            cmd.append("--mask")
            cmd.extend([str(m) for m in args.mask])

        subprocess.run(cmd)

if __name__ == "__main__":
    main()