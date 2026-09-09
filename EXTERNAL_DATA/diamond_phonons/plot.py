#!/usr/bin/env python3
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoid GUI issues
import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fractional_to_cartesian(frac, lattice):
    """Convert fractional q-point to Cartesian using reciprocal lattice"""
    return np.dot(frac, lattice)


def compute_q_path(qpoints_frac, lattice):
    """Compute cumulative distance along q-path"""
    qpoints_cart = np.array([fractional_to_cartesian(q, lattice) for q in qpoints_frac])
    dist = np.zeros(len(qpoints_cart))
    for i in range(1, len(qpoints_cart)):
        dist[i] = dist[i-1] + np.linalg.norm(qpoints_cart[i] - qpoints_cart[i-1])
    return dist

# this does not really compute the position along the kpath of the high simmetry points
# rather it assigns to the high simmetry point the position of the closest point in the grid
# somewhat hacky, could just compute the position along the kpath
# DO NOT PUT A PLOT USING THIS IN ANYTHING SERIUS
def find_label_positions(labels_dict, qpoints_frac, qpath_dist):
    """Map high-symmetry labels to positions along cumulative q-path"""
    label_positions = {}
    for label, frac in labels_dict.items():
        frac = np.array(frac)
        distances = np.linalg.norm(np.array(qpoints_frac) - frac, axis=1)
        idx = np.argmin(distances)
        label_positions[label] = qpath_dist[idx]
    return label_positions


def plot_phonon_bs_dos(bs, dos, outfile):
    fig, (ax_bs, ax_dos) = plt.subplots(
        1, 2, figsize=(9, 4),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=True
    )

    # ----- q-path and distances -----
    qpoints_frac = bs["qpoints"]
    lattice = np.array(bs["reciprocal_lattice"])
    qpath_dist = compute_q_path(qpoints_frac, lattice)

    # ----- Frequencies -----
    freqs = np.array(bs["frequencies"])  # shape (n_modes, n_qpoints)
    if freqs.shape[1] != len(qpath_dist):
        raise ValueError(f"Number of q-points {len(qpath_dist)} does not match frequencies {freqs.shape[1]}")

    for band_idx in range(freqs.shape[0]):
        ax_bs.plot(qpath_dist, freqs[band_idx, :], color="black", lw=1)

    # ----- High-symmetry labels -----
    label_positions = find_label_positions(bs.get("labels_dict", {}), qpoints_frac, qpath_dist)
    for label, pos in label_positions.items():
        label = r"$\Gamma$" if label == "\Gamma" else label
        ax_bs.axvline(pos, color="gray", lw=0.5)
        ax_bs.text(pos, ax_bs.get_ylim()[0]-1, label, ha="center", va="top", fontsize=8)
    ax_bs.set_axis_off()
    ax_bs.axhline(0, color="gray", lw=0.5)
    ax_bs.set_xlabel("Wave vector")
    ax_bs.set_ylabel("Frequency (THz)")

    # ----- DOS -----
    dos_freqs = np.array(dos["frequencies"])
    dos_vals = np.array(dos["densities"])
    ax_dos.plot(dos_vals, dos_freqs, color="black")
    ax_dos.fill_betweenx(dos_freqs, dos_vals, alpha=0.3)
    ax_dos.set_xlabel("DOS")
    ax_dos.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Plot saved to {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Plot phonon band structure and DOS")
    parser.add_argument("--bs", required=True, help="Band structure JSON file")
    parser.add_argument("--dos", required=True, help="DOS JSON file")
    parser.add_argument("-o", "--out", default="phonons_bs_dos.png", help="Output image file")
    args = parser.parse_args()

    bs = load_json(args.bs)
    dos = load_json(args.dos)
    plot_phonon_bs_dos(bs, dos, args.out)


if __name__ == "__main__":
    main()

