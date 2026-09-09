#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram
from pathlib import Path


def parse_lammps_thermo(filename, thermo_idx):
    """
    Reads a LAMMPS thermo-style table.
    Returns:
        headers (list of str)
        data (2D numpy array)
    """
    headers = None
    data = []
    thermo_index = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect header line (starts with Step ...)
            if line.startswith("Step"):
                if thermo_index == thermo_idx:
                    print("[INFO] Found thermo_index", thermo_idx)
                    headers = line.split()
                    thermo_index+=1
                    continue
                else:
                    thermo_index+=1

            # Read numeric data
            if headers is not None:
                parts = line.split()
                if len(parts) == len(headers):
                    try:
                        data.append([float(x) for x in parts])
                    except ValueError:
                        pass

    if headers is None or not data:
        raise RuntimeError("Could not parse thermo data from file")

    # from pprint import pprint
    # pprint(data)
    return headers, np.array(data)


def autocorrelation(x):
    """
    Normalized autocorrelation function.
    """
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    corr /= corr[0]
    return corr


def analyze_quantity(time, values, dt, name, outdir):
    """
    Computes PSD and autocorrelation and saves a PDF.
    """
    # PSD
    # freqs, psd = welch(values, fs=1.0 / dt, nperseg=min(1024, len(values)))
    freqs, psd = periodogram(values, fs=1.0/dt)

    # Autocorrelation
    acf = autocorrelation(values)
    lag_time = np.arange(len(acf)) * dt

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    axes[0].loglog(freqs, psd)
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("PSD")
    axes[0].set_title(f"{name} - Power Spectral Density")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].set_ylim(1e-23, np.max(psd))


    axes[1].plot(lag_time, acf)
    axes[1].set_xlabel("Time lag")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].set_title(f"{name} - Autocorrelation Function")
    axes[1].grid(True)

    fig.tight_layout()
    outfile = outdir / f"{name}_analysis.pdf"
    fig.savefig(outfile)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="PSD and autocorrelation analysis of LAMMPS thermo output"
    )
    parser.add_argument(
        "--input", required=True, help="LAMMPS output file"
    )
    parser.add_argument(
        "--dt", type=float, required=True, help="Simulation timestep"
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Maximum step to include (optional)",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_results",
        help="Output directory for PDFs",
    )
    parser.add_argument("--thermo_idx", required=False, default=0, type=int, help="This script looks for the first instance of a line which starts with Step. If multiple thermo are written to the same file, select the desired one with --thermo_idx (zero-indexed)")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    headers, data = parse_lammps_thermo(args.input, args.thermo_idx)

    # Separate Step column
    step_index = headers.index("Step")
    steps = data[:, step_index].astype(int)

    if args.max_step is not None:
        mask = steps <= args.max_step
        data = data[mask]
        steps = steps[mask]

    time = steps * args.dt

    for i, name in enumerate(headers):
        if name == "Step":
            continue

        values = data[:, i]
        analyze_quantity(time, values, args.dt, name, outdir)


if __name__ == "__main__":
    main()
