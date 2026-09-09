# =====================================
# Script Name: compare_freqs.py
# Purpose: Compare dimer vibrational frequencies across multiple MACE models and DFT reference data
# Usage: Run to generate scatter plot comparing model predictions against DFT reference frequencies
# =====================================

import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


# File paths for dimer frequency JSON outputs from various MACE models and DFT reference
FILES = [
    "./mace-GAP2020-FROM_SCRATCH_v3/2_atoms_potential/dimer_frequency.json",
    "./from_C2_cut/2_atoms_potential/dimer_frequency.json",
    "./mace-GAP2020-FROM_SCRATCH_v5/2_atoms_potential/dimer_frequency.json",
    "./mace-GAP2020-FROM_SCRATCH_v4/2_atoms_potential/dimer_frequency.json",
    "./mace-GAP2020_FROM_SCRATCH_smaller2_l2/2_atoms_potential/dimer_frequency.json",
    "./from_C2_v1/2_atoms_potential/dimer_frequency.json",
    "./mace-GAP2020_FROM_SCRATCH_smaller/2_atoms_potential/dimer_frequency.json",
    "/home/gabri/Thesis/PROJECT/DATASETS/C/GAP2020/config_types/Dimer/dimer_frequency.json"
]

# Corresponding labels for the models above (in order)
LABELS = [
    "v3", "from-C2-cut", "v5", "v4", "smaller-l2",
    "from-C2", "smaller", "reference"
]


def read_exact_freq(path):
    """Read exact vibrational frequency from a dimer_frequency.json file.

    Args:
        path (str): File path to the dimer_frequency.json output

    Returns:
        float: The exact vibrational frequency value
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data["exact_freq"]


def main():
    freqs = [read_exact_freq(f) for f in FILES]

    x = np.arange(len(freqs))

    # reference = last entry
    ref_freq = freqs[-1]

    plt.figure(figsize=(8, 4))

    plt.scatter(x[:-1], freqs[:-1])
    plt.axhline(ref_freq, linestyle="--", label="reference")

    plt.xticks(x[:-1], LABELS[:-1], rotation=45, ha="right")
    plt.ylabel("Frequency [THz]")
    plt.legend()
    plt.tight_layout()

    plt.savefig("frequency_comparison.pdf")


    plt.show()

    return 0


if __name__ == "__main__":
    main()
