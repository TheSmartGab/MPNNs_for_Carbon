#!/usr/bin/env python3

"""Driver script for dimer frequency extraction across directory structures.

This module searches a root directory recursively for target configuration files
(model_penergy.out within 2_atoms_potential/ directories) and processes them using
a specified dimer frequency computation script.

Functions:
    find_targets: Find all target files matching the expected relative path pattern
    main: CLI entry point to process found targets
"""

import argparse
import os
import subprocess
import sys


TARGET_RELATIVE_PATH = os.path.join(
    "2_atoms_potential",
    "model_penergy.out"
)


def find_targets(root_dir):
    """Find target files matching the exact relative path pattern.

    Searches recursively through root_dir for files at the expected relative path:
    <root>/2_atoms_potential/model_penergy.out

    Args:
        root_dir: String path to the root directory to search.

    Returns:
        list[str]: List of absolute file paths matching the target pattern.
    """
    matches = []
    for root, _, files in os.walk(root_dir):
        candidate = os.path.join(root, TARGET_RELATIVE_PATH)
        if os.path.isfile(candidate):
            matches.append(candidate)
    return matches


def main():
    """CLI entry point for processing dimer configurations across directories.

    This function searches a root directory recursively for target files matching
    2_atoms_potential/model_penergy.out, then processes each found file using the
    specified dimer frequency script with appropriate output directories.

    Returns:
        None (exits with status 1 on error, or 0 after processing all targets)
    """
    parser = argparse.ArgumentParser(
        description="Driver for dimer frequency extraction across directory structures."
    )

    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to search recursively for target model_penergy.out files"
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to the dimer frequency computation script"
    )

    # Everything else is forwarded as arguments to the processing script
    args, inner_args = parser.parse_known_args()

    targets = find_targets(args.root_dir)

    if not targets:
        print("[ERROR] No matching model_penergy.out files found")
        sys.exit(1)

    print(f"[INFO] Found {len(targets)} target file(s)")

    for fpath in targets:
        outdir = os.path.dirname(fpath)

        print(f"[INFO] Processing {fpath}")
        print(f"[INFO] Output directory: {outdir}")

        cmd = [
            sys.executable,
            args.script,
            fpath,
            "--outdir",
            outdir,
        ] + inner_args

        print("[INFO] Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
