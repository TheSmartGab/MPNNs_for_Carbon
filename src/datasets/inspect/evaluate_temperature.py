from argparse import ArgumentParser
import numpy as np
import os
from ase.io import read
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.constants import Boltzmann, e
from scipy.optimize import curve_fit

KB = Boltzmann / e  # Boltzmann constant in eV/K units

# Maxwell-Boltzmann 3D kinetic energy probability density
def maxwell_boltzmann(E, T):
    prefactor = 2 / np.sqrt(np.pi) * (1 / (KB * T))**(1.5)
    return prefactor * np.sqrt(E) * np.exp(-E / (KB * T))

def main():
    parser = ArgumentParser(
        prog="evaluate_kinetic_energy.py",
        description="Estimate MD temperature and plot KE histogram with MB fit"
    )
    parser.add_argument("input_file", type=str, help="Input file with momenta (ASE-readable)")
    parser.add_argument("output_file", type=str, help="Output plot file")
    parser.add_argument("--format", default="extxyz", type=str, help="Format of input file")
    parser.add_argument("--nbins", default=50, type=int, help="Histogram bins")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"[ERROR] input file {args.input_file} does not exist. Exit.")
        exit(-1)

    # Read all frames
    images = read(args.input_file, format=args.format, index=":")
    print("[INFO] found", len(images), "images")

    # Collect kinetic energies
    kins = []
    for atoms in images:
        momenta = atoms.get_momenta()
        masses = atoms.get_masses()
        energies = [np.sum(mom*mom)/(2*mass) for mom,mass in zip(momenta, masses)]
        kins.extend(energies)
    kins = np.array(kins)
    N = len(kins)
    print(f"[INFO] Total particles: {N}")

    # Direct estimate of temperature
    T_direct = 2.0 * np.sum(kins) / (3.0 * N * KB)
    print(f"[INFO] Direct temperature estimate: {T_direct:.2f} K")

    # Fit Maxwell-Boltzmann to histogram
    counts, bin_edges = np.histogram(kins, bins=args.nbins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Avoid zero energies for sqrt
    mask = bin_centers > 0
    bin_centers_fit = bin_centers[mask]
    counts_fit = counts[mask]

    popt, pcov = curve_fit(maxwell_boltzmann, bin_centers_fit, counts_fit, p0=[T_direct])
    T_fit = popt[0]
    print(f"[INFO] Fitted temperature from Maxwell-Boltzmann: {T_fit:.2f} K")

    # Plot histogram and fit
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.hist(kins, bins=args.nbins, density=True, alpha=0.6, color='b', label='MD kinetic energies')

    E_main = np.linspace(0, np.max(kins), 500)
    ax.plot(E_main, maxwell_boltzmann(E_main, T_direct), 'r--', lw=2, label=f'MB direct T={T_direct:.1f} K')
    ax.plot(E_main, maxwell_boltzmann(E_main, T_fit), 'g-', lw=2, label=f'MB fit T={T_fit:.1f} K')

    ax.set_xlabel("Kinetic energy (eV)")
    ax.set_ylabel("Probability density")
    ax.set_title("Kinetic energy distribution with MB fit")
    ax.legend()

    plt.savefig(args.output_file)
    plt.close()
    print(f"[INFO] Histogram with MB fit saved to {args.output_file}")

if __name__ == "__main__":
    main()
