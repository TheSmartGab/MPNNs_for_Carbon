import numpy as np
import matplotlib.pyplot as plt
import argparse

energy_frame = 15  # eV


def load_distribution(distribution_file):
    """Load the pair distance distribution data if provided."""
    try:
        data = np.loadtxt(
            distribution_file, comments="#", skiprows=0
        )  # Skip comment lines
        bin_centers = data[:, 0]
        frequencies = data[:, 1]
        return bin_centers, frequencies
    except Exception as e:
        print(f"Error reading distribution file: {e}")
        return None, None

def ZBL_potential(distance, Zi, Zj):
    """Calculate ZBL potential for two atoms with atomic numbers Zi and Zj at a given distance."""
    # Constants
    a0 = 0.529177  # Bohr radius in Angstroms
    e2 = 14.3996  # e^2 in eV*Angstrom
    # Screening length
    a = 0.8854 * a0 / (Zi**0.23 + Zj**0.23)
    # ZBL potential formula
    phi = (
        0.1818 * np.exp(-3.2 * distance / a)
        + 0.5099 * np.exp(-0.9423 * distance / a)
        + 0.2802 * np.exp(-0.4029 * distance / a)
        + 0.02817 * np.exp(-0.2016 * distance / a)
    )
    V = (Zi * Zj * e2 / distance) * phi
    return V

def derivative(y, x):
    """Compute the numerical derivative dy/dx."""
    dy = np.diff(y)
    dx = np.diff(x)
    derivative = dy / dx
    # To match the original array size, append the last value
    derivative = np.append(derivative, derivative[-1])
    return derivative

def main():
    # --- Argument parser setup ---
    parser = argparse.ArgumentParser(
        description="Plot potential energy vs distance and optionally overlay pair distance distribution."
    )
    parser.add_argument("input_file", help="Input data file (distance energy).")
    parser.add_argument("output_file", help="Output plot file (e.g., output.png).")
    parser.add_argument("info_file", help="Output info file (summary).")
    parser.add_argument(
        "--distribution_file",
        help="Optional pair distance distribution file (Bin_Center, Frequency).",
        default=None,
    )
    parser.add_argument("--ZBL_Zi", type=int, help="Atomic number of atom i for ZBL potential.", default=None)

    args = parser.parse_args()

    # --- Load potential energy data ---
    data = np.loadtxt(args.input_file)
    distances = data[:, 0]
    energies = data[:, 1]

    # --- Compute key properties ---
    last_energy = energies[-1]  # Asymptotic energy
    min_index = np.argmin(energies)
    min_energy = energies[min_index]
    min_distance = distances[min_index]

    # Detect saturation (energy approaches last_energy)
    tolerance = 1e-12
    diffs = np.abs(energies[min_index:] - last_energy)
    first_saturated_index = next(
        (min_index + i for i, diff in enumerate(diffs) if diff <= tolerance),
        len(energies) - 1,
    )

    forces = -derivative(energies, distances)

    # --- Print results ---
    print(f"Minimum Potential Energy: {min_energy:.6f} eV")
    print(f"Distance at Minimum Energy: {min_distance:.6f} Å")
    print(f"Asymptotic Energy: {last_energy:.6f} eV")
    print(f"Distance at which Energy Saturates: {distances[first_saturated_index]:.6f} Å")
    print(f"Binding Energy: {last_energy - min_energy:.6f} eV")

    # --- Write results to info file ---
    with open(args.info_file, "w") as f:
        f.write(f"Minimum Potential Energy: {min_energy:.6f} eV\n")
        f.write(f"Distance at Minimum Energy: {min_distance:.6f} Å\n")
        f.write(f"Asymptotic Energy: {last_energy:.6f} eV\n")
        f.write(
            f"Distance at which Energy Saturates: {distances[first_saturated_index]:.6f} Å\n"
        )
        f.write(f"Binding Energy: {last_energy - min_energy:.6f} eV\n")

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Primary axis: Potential Energy
    ax1.plot(distances, energies, marker="o", linestyle="-", markersize=2, label="Potential Energy")
    ax1.set_title(r"Potential Energy [eV] vs Distance [$\AA$] Between Two Atoms")
    ax1.set_xlabel(r"Distance [$\AA$]")
    ax1.set_ylabel("Potential Energy [eV]")
    ax1.grid(True)
    ax1.set_ylim(min_energy - 5, min_energy + energy_frame)

    # Annotate important lines
    ax1.hlines(
        last_energy,
        xmin=distances[0],
        xmax=distances[-1],
        colors="r",
        linestyles="dashed",
        label=f"Asymptotic Energy {last_energy:.3f} eV",
    )
    ax1.vlines(
        distances[first_saturated_index],
        ymin=min_energy - energy_frame,
        ymax=min_energy + energy_frame,
        colors="g",
        linestyles="dashed",
        label=rf"Saturation Distance {distances[first_saturated_index]:.2f}$\AA$",
    )
    if args.ZBL_Zi is not None:
        Zj = args.ZBL_Zi
        ZBL_energies = ZBL_potential(distances, args.ZBL_Zi, Zj) + last_energy
        ax1.plot(
            distances,
            ZBL_energies,
            marker="",
            linestyle="--",
            color="orange",
            label=f"ZBL Potential (Z={args.ZBL_Zi})",
        )
    ax1.plot(
        min_distance,
        min_energy,
        marker="x",
        color="red",
        markersize=5,
        label=f"Minimum Energy {min_energy:.3f} eV at {min_distance:.2f} Å",
    )

    xticks = np.arange(0, 11)
    ax1.set_xticks(xticks)

    # --- Optional: Plot pair distance distribution ---
    if args.distribution_file:
        bin_centers, frequencies = load_distribution(args.distribution_file)
        if bin_centers is not None:
            ax2 = ax1.twinx()
            ax2.bar(
                bin_centers,
                frequencies,
                width=bin_centers[1] - bin_centers[0],
                color="gray",
                alpha=0.4,
                label="Pair Distance Distribution",
            )
            ax2.set_ylabel("Frequency (a.u.)")

            # Add legend for secondary axis
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        else:
            ax1.legend()
    else:
        ax1.legend()

    plt.tight_layout()
    plt.savefig(args.output_file)
    plt.close()

    # plot the force
    plt.figure(figsize=(8, 6))
    plt.plot(distances, forces, marker="o", linestyle="-", markersize=2, label="Force")
    plt.title(r"Force vs Distance Between Two Atoms")
    plt.xlabel(r"Distance [$\AA$]")
    plt.ylabel(r"Force [eV/$\AA$]")
    plt.grid(True)
    plt.xticks(xticks)
    plt.ylim(-25, 100)

    plt.tight_layout()
    force_output_file = args.output_file.replace(".pdf", "_force.pdf")
    plt.savefig(force_output_file)
    plt.close()


if __name__ == "__main__":
    main()
