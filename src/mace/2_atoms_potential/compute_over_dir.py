#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import subprocess
import sys


def find_models(root_dir, model_name):
    """Recursively find model_name under root_dir"""
    matches = []
    for root, _, files in os.walk(root_dir):
        if model_name in files:
            matches.append(os.path.join(root, model_name))
    return matches


def main():
    parser = ArgumentParser(
        description="Driver script to locate MACE models and run diatomic PES"
    )

    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to search recursively for models"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MACE.model",
        help="Model filename to look for (default: MACE.model)"
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to inner diatomic PES script"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="2_atoms_potential",
        help="Output directory name to create inside each model directory"
    )

    # Everything else goes to the inner script
    args, inner_args = parser.parse_known_args()

    models = find_models(args.root_dir, args.model_name)

    if not models:
        print(f"[ERROR] No '{args.model_name}' found under {args.root_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(models)} model(s)")

    for model_path in models:
        model_dir = os.path.dirname(model_path)

        # Create per-model output directory
        model_output_dir = os.path.join(model_dir, args.output_dir)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"[INFO] Running model in {model_dir}")
        print(f"[INFO] Output dir: {model_output_dir}")

        cmd = [
            sys.executable,
            args.script,
            "--model_path",
            model_path,
            "--output_dir",
            model_output_dir,
        ] + inner_args

        print("[INFO] Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
