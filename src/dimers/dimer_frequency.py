# =====================================
# Script Name: dimer_frequency.py
# Purpose: Compute vibrational frequencies from potential energy data for two-atom configurations
# Usage: Run with JSON input files containing pair distance and energy data to generate frequency analyses
# =====================================

"""Dimer frequency computation module.

This script computes harmonic vibrational frequencies from potential energy data for
two-atom (dimer) configurations. It performs quadratic fitting of the potential energy
curve near the equilibrium bond length and converts the curvature to physical frequency
units (THz) using reduced mass calculations and SI unit conversions.
"""

#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Physical constants (SI)
EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1e-10
AMU_TO_KG = 1.66053906660e-27
TWOPI = 2.0 * np.pi


def quadratic_fit(x, y):
    """
    Fit y = ax^2 + bx + c
    Returns coefficients (a, b, c)
    """
    return np.polyfit(x, y, 2)


def frequency_from_parabola(a, mu_amu):
    """
    Compute harmonic frequency (THz) from parabola curvature.

    E(x) = a x^2 + ...
    k = d2E/dx2 = 2a

    Units assumed:
    - x in Å
    - E in eV
    - mu in amu
    """
    # Convert curvature to SI
    k_si = 2.0 * a * EV_TO_J / (ANGSTROM_TO_M ** 2)
    mu_si = mu_amu * AMU_TO_KG

    omega = np.sqrt(k_si / mu_si)      # rad/s
    freq_thz = omega / TWOPI / 1e12     # THz
    return freq_thz


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate minimum to extract dimer frequency"
    )
    parser.add_argument("input", help="2-column file: distance energy")
    parser.add_argument(
        "-n", "--nfit",
        type=int,
        default=5,
        help="Number of lowest-energy points for fit (default: 5)"
    )
    parser.add_argument(
        "-o", "--outdir",
        default="output",
        help="Output directory (created if missing)"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help="Reduced mass in amu (default: 1.0)"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = np.loadtxt(args.input)
    x = data[:, 0]
    y = data[:, 1]

    # Sort by energy
    idx = np.argsort(y)
    x_sorted = x[idx]
    y_sorted = y[idx]

    # Lowest points
    x3, y3 = x_sorted[:3], y_sorted[:3]
    xn, yn = x_sorted[:args.nfit], y_sorted[:args.nfit]

    # Fits
    a3, b3, c3 = quadratic_fit(x3, y3)
    an, bn, cn = quadratic_fit(xn, yn)

    # Frequencies
    freq_exact = frequency_from_parabola(a3, args.mu)
    freq_fit = frequency_from_parabola(an, args.mu)

    # Plot grid
    xgrid = np.linspace(xn.min(), xn.max(), 500)
    y_exact = a3 * xgrid**2 + b3 * xgrid + c3
    y_fit = an * xgrid**2 + bn * xgrid + cn

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o", label="Data")
    plt.plot(
        xgrid, y_exact, "--",
        label=f"Exact (3 pts): {freq_exact:.3f} THz"
    )
    plt.plot(
        xgrid, y_fit, "-",
        label=f"Fit ({args.nfit} pts): {freq_fit:.3f} THz"
    )

    plt.xlim(xn.min(), xn.max())
    plt.ylim(yn.min()-1, yn.max()+1)

    plt.xlabel("Distance")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(args.outdir, "dimer_fit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # JSON output
    result = {
        "exact_freq": freq_exact,
        "fitted_freq": freq_fit,
        "nfit": args.nfit
    }

    json_path = os.path.join(args.outdir, "dimer_frequency.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results written to {args.outdir}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
