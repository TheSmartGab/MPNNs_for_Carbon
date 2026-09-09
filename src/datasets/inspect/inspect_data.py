# =====================================
# Script Name: inspect_data.py
# Purpose: Inspect extxyz dataset contents for distributions, correlations, and structural properties
# Usage: Run from command line with dataset directories to generate histograms and statistical summaries
# =====================================

"""Dataset inspection utilities module.
This script was originally written to parse data from a particular NOMAD data structure.
"""

import os
from argparse import ArgumentParser
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def parse_filename(filename):
    """Extract entry_id, run_number, Force_type, dim, chemical_hill from filename, preserving leading underscores."""
    try:
        if "." not in filename:
            raise ValueError("No file extension found")

        name, _ = filename.split(".", 1)
        prefix_len = len(name) - len(name.lstrip("_"))
        underscores = "_" * prefix_len
        fields = name.split("_")

        if len(fields) < 5:
            print(f"WARNING: could not parse filename '{filename}' (too few fields)")
            entry_id = underscores + fields[0] if fields else underscores + "UNKNOWN"
            return entry_id, "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"

        entry_id = underscores + fields[-5]
        run_number = fields[-4]
        Force_type = fields[-3]
        dim = fields[-2]
        chemical_hill = fields[-1]

        return entry_id, run_number, Force_type, dim, chemical_hill

    except Exception as e:
        print(f"ERROR parsing filename '{filename}': {e}")
        return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"


def parse_info(info_path):
    """Read filenames from info file after 'ENTRIES' line"""
    if not os.path.exists(info_path):
        print(f"WARNING: info file not found: {info_path}")
        return []

    filenames = []
    try:
        with open(info_path, "r") as f:
            lines = f.readlines()

        read = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if read:
                filenames.append(line)
            if "ENTRIES" in line:
                read = True
    except Exception as e:
        print(f"ERROR reading info file '{info_path}': {e}")
        return []

    return filenames


def extract_features(filenames):
    """Vectorized extraction of dim and chemical_hill from list of filenames"""
    dims, chem_hills, ids = [], [], []
    for fn in filenames:
        entry_id, _, _, dim, chem = parse_filename(fn)
        ids.append(entry_id)
        dims.append(dim or "UNKNOWN")
        chem_hills.append(chem or "UNKNOWN")

    if not dims:  # empty
        return np.array([]), np.array([]), np.array([])

    return np.array(ids), np.array(dims), np.array(chem_hills)


def parse_extxyz(xyz_path):
    """
    Parse an .xyz file in extended xyz format.
    Returns:
        forces: list of [fx, fy, fz] for all atoms
        energies: list of total energies (one per structure)
        natoms: list of atom counts (one per structure)
    """
    if not os.path.exists(xyz_path):
        print(f"WARNING: xyz file not found: {xyz_path}")
        return np.array([]), np.array([]), np.array([])

    forces, energies, natoms = [], [], []

    try:
        with open(xyz_path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            try:
                n = int(lines[i].strip())
            except ValueError:
                print(f"Invalid atom count line at {i}: {lines[i].strip()}")
                i += 1
                continue

            if i + 1 >= len(lines):
                print("Incomplete header line near end of file.")
                break

            natoms.append(n)
            header = lines[i + 1].strip()
            energy = None
            for token in header.split():
                if token.startswith("energy="):
                    try:
                        energy = float(token.split("=")[1])
                    except ValueError:
                        energy = np.nan
                    break
            energies.append(energy if energy is not None else np.nan)

            for j in range(n):
                if i + 2 + j >= len(lines):
                    print("Incomplete atom data near end of file.")
                    break
                parts = lines[i + 2 + j].split()
                if len(parts) < 3:
                    continue
                try:
                    fx, fy, fz = map(float, parts[-3:])
                except ValueError:
                    fx = fy = fz = np.nan
                forces.append([fx, fy, fz])

            i += n + 2

    except Exception as e:
        print(f"ERROR reading xyz file '{xyz_path}': {e}")

    return np.array(forces), np.array(energies), np.array(natoms)


def plot_histogram(data, title, xlabel, ylabel, save_path, bins=50):
    """Plot a histogram for continuous data"""
    if len(data) == 0:
        print(f"Skipping empty histogram: {title}")
        return
    plt.figure(figsize=(10,6))
    plt.hist(data[~np.isnan(data)], bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bar_chart(counter_dict, title, xlabel, ylabel, save_path):
    """Plot a bar chart from a Counter dictionary"""
    if not counter_dict:
        print(f"Skipping empty bar chart: {title}")
        return
    labels, counts = zip(*counter_dict.items())
    plt.figure(figsize=(8,5))
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_split(split_name, split_dir):
    """Parse filenames, count properties, plot histograms, and analyze xyz file"""
    print(f"🔹 Processing split: {split_name}")
    info_path = os.path.join(split_dir, f"{split_name}_info.txt")
    filenames = parse_info(info_path)
    ids, dims, chem_hills = extract_features(filenames)

    if len(filenames) == 0:
        print(f"No filenames found for split: {split_name}")
        return {}, {}

    chem_counts = dict(zip(*np.unique(chem_hills, return_counts=True))) if len(chem_hills) > 0 else {}
    dim_counts = dict(zip(*np.unique(dims, return_counts=True))) if len(dims) > 0 else {}

    plot_bar_chart(chem_counts, f"{split_name.capitalize()} - Chemical Hill Distribution",
                   "Chemical Hill", "Count", os.path.join(split_dir, f"{split_name}_chemical_hill.png"))
    plot_bar_chart(dim_counts, f"{split_name.capitalize()} - Dimensionality Distribution",
                   "Dimensionality", "Count", os.path.join(split_dir, f"{split_name}_dimensionality.png"))

    xyz_path = os.path.join(split_dir, f"{split_name}.xyz")
    forces, energies, natoms = parse_extxyz(xyz_path)

    if len(forces) > 0 and len(energies) > 0:
        try:
            force_modulus = np.linalg.norm(forces, axis=1)
            plot_histogram(force_modulus, f"{split_name.capitalize()} - Force Modulus",
                           "Force magnitude (|F|)", "Count", os.path.join(split_dir, f"{split_name}_force_modulus.png"))

            for idx, comp in enumerate(["Fx", "Fy", "Fz"]):
                plot_histogram(forces[:, idx], f"{split_name.capitalize()} - {comp} Component",
                               f"{comp} (force)", "Count", os.path.join(split_dir, f"{split_name}_{comp.lower()}.png"))

            with np.errstate(divide='ignore', invalid='ignore'):
                energy_per_atom = np.divide(energies, natoms, out=np.full_like(energies, np.nan), where=natoms!=0)
            plot_histogram(energy_per_atom, f"{split_name.capitalize()} - Energy per Atom",
                           "Energy per atom", "Count", os.path.join(split_dir, f"{split_name}_energy_per_atom.png"))
        except Exception as e:
            print(f"Error during xyz plotting for split '{split_name}': {e}")

    return chem_counts, dim_counts


def main():
    parser = ArgumentParser(description="Inspect dataset splits and plot histograms")
    parser.add_argument("root_dir", help="Root dir with training, validation, test subdirectories")
    args = parser.parse_args()
    root_dir = args.root_dir

    splits = ["training", "validation", "test"]
    all_dataset_properties = {}

    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            continue
        chem_counts, dim_counts = process_split(split, split_dir)
        all_dataset_properties[split] = {
            "chemical_hill": chem_counts,
            "dimensionality": dim_counts
        }

    from pprint import pprint
    print("\nDataset summary:")
    pprint(all_dataset_properties)


if __name__ == "__main__":
    main()
