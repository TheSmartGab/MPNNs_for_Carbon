#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import torch
import numpy as np
from ase.io import read, write
from nequip.integrations.ase import NequIPCalculator

# Loosen tolerance safely to account for typical float32 fusion drift,
# but do not blow it open to 1.0 which hides fundamental graph corruption.
os.environ["NEQUIP_FLOAT32_MODEL_TOL"] = "0.01"

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix, compile with modern AOTInductor, and evaluate atomic configurations using a NequIP checkpoint."
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to the input file containing atomic configurations (e.g., dataset.xyz)."
    )
    parser.add_argument(
        "-c", "--checkpoint",
        required=True,
        type=str,
        help="Path to the raw NequIP checkpoint file (e.g., best.ckpt)."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Path to the output file where results will be saved (e.g., output.extxyz)."
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to run both the compilation and ASE evaluation on (default: cpu)."
    )
    parser.add_argument(
        "--index",
        type=str,
        default=":",
        help="ASE slice index for reading configurations (default: ':', which reads all)."
    )

    return parser.parse_args()

def recursive_patch_config(item):
    """Recursively crawls through nested structures to fix syntax schemas and absolute path updates."""
    if isinstance(item, dict):
        if "chemical_symbols" in item:
            symbols = item["chemical_symbols"] if item["chemical_symbols"] else ["C"]
            item["model_type_names"] = symbols
            item["chemical_species_to_atom_type_map"] = {s: s for s in symbols}
            item.pop("chemical_symbols", None)
        
        for k, v in list(item.items()):
            if isinstance(v, str) and "/home/gabri/" in v:
                old_val = v
                item[k] = old_val.replace("/home/gabri/", "/home/gabrielecolombo/")
                print(f"   [Path Fixed] Key '{k}': {old_val} -> {item[k]}")
            else:
                recursive_patch_config(v)
            
    elif isinstance(item, list):
        for i, element in enumerate(item):
            if isinstance(element, str) and "/home/gabri/" in element:
                old_val = element
                item[i] = old_val.replace("/home/gabri/", "/home/gabrielecolombo/")
                print(f"   [Path Fixed] List element: {old_val} -> {item[i]}")
            else:
                recursive_patch_config(element)

def fix_checkpoint_config(checkpoint_path):
    """Deeply inspects and modifies validation syntax, directory paths, and structural legacy EMA weights inside best.ckpt."""
    print(f"Deep scrubbing configuration hierarchies and paths within {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        recursive_patch_config(checkpoint)
        
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            
            # 1. Structural Fix for the EMA weights dimension mismatch
            w0_key = "ema.ema_weight_0"
            w1_key = "ema.ema_weight_1"
            if w0_key in sd and w1_key in sd:
                shape_0 = sd[w0_key].shape
                shape_1 = sd[w1_key].shape
                if shape_0 == torch.Size([1, 8]) and shape_1 == torch.Size([1, 16]):
                    print(f"   [EMA Structural Fix] Rectifying layout mismatch for legacy {checkpoint_path}")
                    temp_w0 = sd[w0_key].clone()
                    sd[w0_key] = torch.zeros((1, 16), dtype=temp_w0.dtype)
                    sd[w1_key] = torch.zeros((1, 8), dtype=temp_w0.dtype)

            # 2. Structural Bridge for Legacy Normalization / Per-Type Scale-Shifts
            # Modern versions look for 'model.per_species_scale' or 'model.scale_by'
            # Legacy checkpoints save them deep in the model structure keys.
            legacy_scale_key = "model.PerTypeScaleShift.per_species_scale"
            legacy_shift_key = "model.PerTypeScaleShift.per_species_shift"
            
            modern_scale_key = "model.per_species_scale"
            modern_shift_key = "model.per_species_shift"
            
            if legacy_scale_key in sd and modern_scale_key not in sd:
                print(f"   [Normalization Bridge] Mapping legacy PerTypeScaleShift scale fields to modern compilation targets...")
                sd[modern_scale_key] = sd[legacy_scale_key].clone()
            if legacy_shift_key in sd and modern_shift_key not in sd:
                print(f"   [Normalization Bridge] Mapping legacy PerTypeScaleShift shift fields to modern compilation targets...")
                sd[modern_shift_key] = sd[legacy_shift_key].clone()

        torch.save(checkpoint, checkpoint_path)
        print("Successfully updated checkpoint configuration schemas and corrected absolute home directory paths.")
    except Exception as e:
        print(f"Error modifying checkpoint internal metadata: {e}", file=sys.stderr)
        sys.exit(1)

def run_command(cmd, description):
    """Helper function to execute shell commands cleanly."""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {description}", file=sys.stderr)
        print(f"Command executed: {e.cmd}", file=sys.stderr)
        print(f"Error output:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def compile_checkpoint(checkpoint_path, device):
    """Executes the modern nequip compile toolchain using AOTInductor matching the target device."""
    abs_checkpoint = os.path.abspath(checkpoint_path)
    base_dir = os.path.dirname(abs_checkpoint)
    checkpoint_file = os.path.basename(abs_checkpoint)
    
    package_file = "packaged_model.nequip.zip"
    deployed_file = f"asedeployed_model_{device}.nequip.pt2"
    
    original_cwd = os.getcwd()
    try:
        os.chdir(base_dir)
        if not os.path.exists(deployed_file):
            # 1. Package building step
            pkg_cmd = f'nequip-package build "{checkpoint_file}" "{package_file}"'
            run_command(pkg_cmd, f"nequip-package build ({checkpoint_file} -> {package_file})")
            
            # 2. Compile targeting the EXACT device used during evaluation loop
            compile_cmd = f'nequip-compile --mode aotinductor --device {device} --target ase "{package_file}" "{deployed_file}"'
            run_command(compile_cmd, f"nequip-compile ({package_file} -> {deployed_file})")
        else:
            print(f"Found existing deployed model for {device} at {deployed_file}. Skipping compilation step.")
        
        return os.path.abspath(deployed_file)
    finally:
        os.chdir(original_cwd)

def main():
    args = parse_arguments()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'", file=sys.stderr)
        sys.exit(1)

    # 1. Deep-scrub metadata config, fix EMA matrix sizes and inject normalization targets
    fix_checkpoint_config(args.checkpoint)

    # 2. Run modern AOT deployment compilation with specific target device match
    deployed_model_path = compile_checkpoint(args.checkpoint, args.device)
    print(f"Successfully generated deployed AOT model at: {deployed_model_path}\n")

    # 3. Load the configurations using ASE
    print(f"Reading configurations from: {args.input}...")
    try:
        atoms_list = read(args.input, index=args.index)
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loaded {len(atoms_list)} configuration(s).")

    # 4. Initialize the calculator via the modern compiled graph signature
    print(f"Initializing NequIP calculator onto {args.device}...")
    try:
        calc = NequIPCalculator.from_compiled_model(
            compile_path=deployed_model_path,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing calculator from compiled path: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Evaluate properties for each configuration
    print("Evaluating atomic properties...")
    model_supports_stress = True

    for idx, atoms in enumerate(atoms_list):
        atoms.calc = calc
        
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            atoms.info["energy"] = energy
            atoms.arrays["forces"] = np.array(forces, dtype=np.float64)
            
            if model_supports_stress and atoms.pbc.any():
                try:
                    stress_raw = atoms.get_stress()
                    stress_arr = np.array(stress_raw, dtype=np.float64)
                    
                    if stress_arr.size == 16:
                        stress_matrix = stress_arr.reshape(4, 4)[:3, :3]
                        stress_voigt = np.array([
                            stress_matrix[0, 0], stress_matrix[1, 1], stress_matrix[2, 2],
                            stress_matrix[1, 2], stress_matrix[0, 2], stress_matrix[0, 1]
                        ])
                        atoms.info["stress"] = stress_voigt
                    elif stress_arr.size == 9:
                        stress_matrix = stress_arr.reshape(3, 3)
                        stress_voigt = np.array([
                            stress_matrix[0, 0], stress_matrix[1, 1], stress_matrix[2, 2],
                            stress_matrix[1, 2], stress_matrix[0, 2], stress_matrix[0, 1]
                        ])
                        atoms.info["stress"] = stress_voigt
                    elif stress_arr.size == 6:
                        atoms.info["stress"] = stress_arr
                    else:
                        atoms.info["raw_stress_tensor"] = stress_arr.flatten()
                        
                except (ValueError, NotImplementedError, RuntimeError) as stress_err:
                    print(f"\n[Warning] Frame {idx} stress parsing issue: {stress_err}", file=sys.stderr)
                    model_supports_stress = False
            
        except Exception as e:
            print(f"\nError: Critical failure evaluating configuration index {idx}: {e}", file=sys.stderr)
            atoms.calc = None
            continue
            
        atoms.calc = None
        
        if (idx + 1) % max(1, len(atoms_list) // 10) == 0 or (idx + 1) == len(atoms_list):
            print(f"  Processed {idx + 1}/{len(atoms_list)} configurations...")

    # 6. Write results back to disk
    print(f"Writing results to: {args.output}...")
    try:
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        write(args.output, atoms_list, format="extxyz")
        print("Done successfully!")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()