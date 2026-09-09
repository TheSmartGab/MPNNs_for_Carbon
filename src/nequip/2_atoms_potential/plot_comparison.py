import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

energy_frame = 15  # eV


def load_distribution(distribution_file):
    """Load the pair distance distribution data if provided."""
    try:
        data = np.loadtxt(distribution_file, comments="#", skiprows=0)
        bin_centers = data[:, 0]
        frequencies = data[:, 1]
        return bin_centers, frequencies
    except Exception as e:
        print(f"Error reading distribution file: {e}")
        return None, None


def load_reference_data(file_path):
    """Load reference data organized as (distance, energy, force)."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        data = np.loadtxt(file_path)
        distances = data[:, 0]
        energies = data[:, 1]
        forces = data[:, 2]
        return {
            "distances": distances,
            "energies": energies,
            "forces": forces,
        }
    except Exception as e:
        print(f"Error reading reference data file {file_path}: {e}")
        return None


def ZBL_potential(distance, Zi, Zj):
    """Calculate ZBL potential."""
    a0 = 0.529177  # Bohr radius in Angstrom
    e2 = 14.3996  # e^2 in eV*Angstrom

    a = 0.8854 * a0 / (Zi**0.23 + Zj**0.23)

    phi = (
        0.1818 * np.exp(-3.2 * distance / a)
        + 0.5099 * np.exp(-0.9423 * distance / a)
        + 0.2802 * np.exp(-0.4029 * distance / a)
        + 0.02817 * np.exp(-0.2016 * distance / a)
    )

    return (Zi * Zj * e2 / distance) * phi


def derivative(y, x):
    """Compute numerical derivative."""
    dy = np.diff(y)
    dx = np.diff(x)
    deriv = dy / dx
    deriv = np.append(deriv, deriv[-1])
    return deriv


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot one or more potential energy curves and optionally overlay "
            "reference datasets and pair distributions."
        )
    )

    # One or more model files
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input model files (distance energy).",
    )

    parser.add_argument(
        "output_file",
        help="Output figure filename.",
    )

    parser.add_argument(
        "info_file",
        help="Summary info filename.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each input file. Required if multiple input files are supplied.",
    )
    parser.add_argument(
        "--distribution_file",
        default=None,
        help="Optional pair distribution file.",
    )

    parser.add_argument(
        "--ZBL_Zi",
        type=int,
        default=None,
        help="Atomic number for ZBL potential.",
    )

    parser.add_argument(
        "--train",
        default=None,
        help="Training reference data (dist energy force).",
    )

    parser.add_argument(
        "--val",
        default=None,
        help="Validation reference data (dist energy force).",
    )

    args = parser.parse_args()

    # -------------------------
    # Validate labels
    # -------------------------
    if len(args.input_files) > 1:
        if args.labels is None:
            parser.error(
                "When multiple input files are provided you must also provide "
                "--labels with the same number of entries."
            )

        if len(args.labels) != len(args.input_files):
            parser.error(
                f"Received {len(args.input_files)} input files but "
                f"{len(args.labels)} labels."
            )
    else:
        if args.labels is None:
            args.labels = ["Model"]
        elif len(args.labels) != 1:
            parser.error(
                "For a single input file provide at most one label."
            )

    # -------------------------
    # Load all models
    # -------------------------
    models = []

    for infile, label in zip(args.input_files, args.labels):
        data = np.loadtxt(infile)

        models.append(
            {
                "label": label,
                "distances": data[:, 0],
                "energies": data[:, 1],
                "forces": -derivative(data[:, 1], data[:, 0]),
            }
        )

    # Use FIRST model for reported quantities
    distances = models[0]["distances"]
    energies = models[0]["energies"]
    forces = models[0]["forces"]

    train_ref = load_reference_data(args.train)
    val_ref = load_reference_data(args.val)

    last_energy = energies[-1]

    min_index = np.argmin(energies)
    min_energy = energies[min_index]
    min_distance = distances[min_index]

    tolerance = 1e-12

    diffs = np.abs(energies[min_index:] - last_energy)

    first_saturated_index = next(
        (
            min_index + i
            for i, diff in enumerate(diffs)
            if diff <= tolerance
        ),
        len(energies) - 1,
    )

    # -------------------------
    # Print summary
    # -------------------------

    print(f"Minimum Potential Energy: {min_energy:.6f} eV")
    print(f"Distance at Minimum Energy: {min_distance:.6f} Å")
    print(f"Asymptotic Energy: {last_energy:.6f} eV")
    print(
        f"Distance at which Energy Saturates: "
        f"{distances[first_saturated_index]:.6f} Å"
    )
    print(f"Binding Energy: {last_energy - min_energy:.6f} eV")

    with open(args.info_file, "w") as f:
        f.write(f"Minimum Potential Energy: {min_energy:.6f} eV\n")
        f.write(f"Distance at Minimum Energy: {min_distance:.6f} Å\n")
        f.write(f"Asymptotic Energy: {last_energy:.6f} eV\n")
        f.write(
            "Distance at which Energy Saturates: "
            f"{distances[first_saturated_index]:.6f} Å\n"
        )
        f.write(
            f"Binding Energy: {last_energy - min_energy:.6f} eV\n"
        )

    # ============================================================
    # ENERGY PLOT
    # ============================================================

    fig, ax1 = plt.subplots(figsize=(12.5, 5))

    # Plot ALL supplied models on top of each other
    for model in models:
        ax1.plot(
            model["distances"],
            model["energies"],
            linewidth=2,
            label=model["label"],
        )

    if train_ref:
        ax1.scatter(
            train_ref["distances"],
            train_ref["energies"],
            color="blue",
            marker="s",
            s=8,
            label="Train",
            zorder=5,
        )

    if val_ref:
        ax1.scatter(
            val_ref["distances"],
            val_ref["energies"],
            color="purple",
            marker="^",
            s=8,
            label="Val",
            zorder=5,
        )

    ax1.set_xlabel(r"$\mathrm{Distance\ [\AA]}$")
    ax1.set_ylabel(r"$\mathrm{Potential\ Energy\ [eV]}$")
    ax1.grid(True)

    ax1.set_ylim(min_energy - 5.5, min_energy + energy_frame)
    ax1.set_xlim(0, 5)

    ax1.hlines(
        last_energy,
        xmin=distances[0],
        xmax=distances[-1],
        colors="red",
        linestyles="dashed",
    )

    if args.ZBL_Zi is not None:
        zbl = (
            ZBL_potential(
                distances,
                args.ZBL_Zi,
                args.ZBL_Zi,
            )
            + last_energy
        )

        ax1.plot(
            distances,
            zbl,
            "--",
            color="orange",
            label=f"ZBL (Z={args.ZBL_Zi})",
        )

    ax1.plot(
        min_distance,
        min_energy,
        "rx",
        markersize=8,
        markeredgewidth=2,
        label=(
            f"Binding Energy "
            f"({last_energy - min_energy:.3f} eV)"
        ),
    )

    xticks = np.linspace(0, 5, 6)
    ax1.set_xticks(xticks)

    if args.distribution_file:

        bin_centers, frequencies = load_distribution(
            args.distribution_file
        )

        if bin_centers is not None:

            ax2 = ax1.twinx()

            ax2.bar(
                bin_centers,
                frequencies,
                width=bin_centers[1] - bin_centers[0],
                color="gray",
                alpha=0.4,
                label="Pair Distribution",
            )

            ax2.set_ylabel("Counts")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()

            ax1.legend(
                h1 + h2,
                l1 + l2,
                loc=(1.35, 0),
            )

        else:
            ax1.legend(loc="best")

    else:
        ax1.legend(loc="best")

    plt.tight_layout()
    plt.savefig(args.output_file)
    plt.close()

    # ============================================================
    # FORCE PLOT
    # ============================================================

    plt.figure(figsize=(6, 4))

    # Plot ALL force curves together
    for model in models:
        plt.plot(
            model["distances"],
            model["forces"],
            linewidth=2,
            label=f"{model['label']} Force",
        )

    if train_ref:
        plt.scatter(
            train_ref["distances"],
            train_ref["forces"],
            color="blue",
            marker="s",
            s=25,
            label="Train Ref Force",
            zorder=5,
        )

    if val_ref:
        plt.scatter(
            val_ref["distances"],
            val_ref["forces"],
            color="purple",
            marker="^",
            s=30,
            label="Val Ref Force",
            zorder=5,
        )

    plt.xlabel(r"$\mathrm{Distance\ [\AA]}$")
    plt.ylabel(r"$\mathrm{Force\ [eV/\AA]}$")
    plt.grid(True)

    plt.xticks(xticks)
    plt.ylim(-25, 100)

    plt.legend(loc="best")

    plt.tight_layout()

    force_output_file = (
        args.output_file.replace(".pdf", "_force.pdf")
        .replace(".png", "_force.png")
    )

    if force_output_file == args.output_file:
        force_output_file += "_force.pdf"

    plt.savefig(force_output_file)
    plt.close()


if __name__ == "__main__":
    main()
