#!/usr/bin/env python3

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# Load band.yaml
# -----------------------------------------------------------
def load_band_yaml(path):
    """Load band.yaml and return distances, frequencies, group velocities, labels."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    phonons = data["phonon"]
    distances = np.array([p["distance"] for p in phonons])
    nbands = len(phonons[0]["band"])

    frequencies = np.zeros((nbands, len(distances)))
    gvectors = np.zeros((nbands, len(distances), 3))

    for iq, p in enumerate(phonons):
        for ib in range(nbands):
            frequencies[ib, iq] = p["band"][ib]["frequency"]
            if "group_velocity" in p["band"][ib]:
                gvectors[ib, iq] = p["band"][ib]["group_velocity"]
            else:
                gvectors = None

    # Extract high-symmetry labels and their distances
    # "labels" is a list of pairs:  [ ['Γ','M'], ['M','K'], ... ]
    labels = data.get("labels", None)

    hs_distances = []
    hs_labels = []

    if labels is not None:
        # The band path is segmented: each segment has Nq points.
        # segment_nqpoint tells how many points per segment.
        segments = data["segment_nqpoint"]  # e.g. [101,101,101]

        cum_index = 0
        # First high-symmetry point begins the first segment
        hs_distances.append(distances[0])
        hs_labels.append(labels[0][0])

        for seg_i, npts in enumerate(segments):
            end_idx = cum_index + npts - 1
            hs_distances.append(distances[end_idx])
            hs_labels.append(labels[seg_i][1])
            cum_index += npts

    return distances, frequencies, gvectors, hs_distances, hs_labels


# -----------------------------------------------------------
# Generate consistent colors for bands
# -----------------------------------------------------------
def get_band_colors(nbands):
    cmap = plt.get_cmap("tab20")  # good qualitative colormap
    colors = [cmap(i % 20) for i in range(nbands)]
    return colors


# -----------------------------------------------------------
# Plot Band Structure with high-symmetry vertical lines
# -----------------------------------------------------------
def plot_band_structure(distances, freqs, hs_distances, hs_labels, colors, outpath):
    plt.figure(figsize=(8, 6))

    nbands = freqs.shape[0]
    for ib in range(nbands):
        plt.plot(distances, freqs[ib], lw=1.2, color=colors[ib], label = f'Band {ib+1}')

    # Vertical lines & labels
    for x, lab in zip(hs_distances, hs_labels):
        plt.axvline(x, color="black", lw=0.7, ls="--")
        plt.text(x, 1.02, lab, transform=plt.gca().get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=10)

    plt.xlabel("k-path distance")
    plt.ylabel("Frequency (THz)")
    # plt.title("Phonon Band Structure")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -----------------------------------------------------------
# Plot group velocity magnitude
# -----------------------------------------------------------
def plot_group_velocity(distances, gvectors, hs_distances, hs_labels, colors, outpath):
    vmag = np.linalg.norm(gvectors, axis=2)

    plt.figure(figsize=(8, 6))
    nbands = vmag.shape[0]

    for ib in range(nbands):
        # avoid gamma point, where the group velocity is showed as 0
        plt.plot(distances[1:-1], vmag[ib][1:-1], lw=1.2, color=colors[ib], label=f'Band {ib+1}')

    # Vertical lines & labels
    for x, lab in zip(hs_distances, hs_labels):
        plt.axvline(x, color="black", lw=0.7, ls="--")
        plt.text(x, 1.02, lab, transform=plt.gca().get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=10)

    plt.xlabel("k-path distance")
    plt.ylabel("|group velocity| (THz·Å)")
    # plt.title("Phonon Group Velocities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot phonon band structure and group velocities from band.yaml"
    )
    parser.add_argument(
        "band_yaml",
        type=str,
        help="Path to band.yaml file (must contain group velocities)"
    )

    args = parser.parse_args()
    band_yaml_path = args.band_yaml

    if not os.path.isfile(band_yaml_path):
        raise FileNotFoundError(f"File not found: {band_yaml_path}")

    # Load data
    distances, freqs, gvectors, hs_distances, hs_labels = load_band_yaml(band_yaml_path)

    # Output paths
    directory = os.path.dirname(os.path.abspath(band_yaml_path))
    band_out = os.path.join(directory, "band_structure.pdf")
    gv_out = os.path.join(directory, "group_velocity.pdf")

    # Band colors (consistent)
    nbands = freqs.shape[0]
    colors = get_band_colors(nbands)

    # Plot Band Structure
    plot_band_structure(distances, freqs, hs_distances, hs_labels, colors, band_out)
    print(f"Band structure saved to: {band_out}")

    # Plot Group Velocity
    if gvectors is not None:
        plot_group_velocity(distances, gvectors, hs_distances, hs_labels, colors, gv_out)
        print(f"Group velocity plot saved to: {gv_out}")
    else:
        print("WARNING: band.yaml does not contain group_velocity data.")


if __name__ == "__main__":
    main()
