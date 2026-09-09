# =====================================
# Script Name: plot_phonons.py
# Purpose: Main phonon plotting utility for visualizing phonon dispersions and density of states
# Usage: Run from command line with --sources, --formats, --labels, and --output_dir arguments
#        to generate comparison plots of phonon band structures across different models/DFT references.
# =====================================

"""Phonon dispersion plotting module.

This script provides the main functionality for visualizing phonon dispersions
and density of states from various sources including MACE/NequIP ML potentials,
DFT calculations (Quantum ESPRESSO), and Materials Project reference data. It supports
comparing multiple models on the same plot with customizable k-path selections.
"""

# add directory of this file to path
import os, sys
import numpy as np
import re  # Added for extracting strain numbers from paths

# import libraries
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from argparse import ArgumentParser
import yaml  # use yaml rather than json. phonopy output is in yaml format. yaml generalize json.

from lib import *
from lib import formats as lib_formats

import matplotlib.pyplot as plt
import matplotlib.cm as cm # colormap

# Enable LaTeX rendering in matplotlib for clean symbol presentation
plt.rcParams.update({
    "text.usetex": False,  # MathText is safer unless you have a full TeX distro installed
    "mathtext.fontset": "cm"
})

fmax = 0
DELTAF = 3  # how much to add on top of fmax when plotting


def add_plot(ax, data, format: str, kpath: str, label, color, lineplot = False):
    """Add a phonon dispersion plot to the given matplotlib axes.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on
        data: Phonon band structure data in specified format
        format (str): Data format type ('ase', 'phonopy', 'mp', etc.)
        kpath (str): High-symmetry k-path for slicing phonon bands
        label (str): Label for the legend
        color (str): Color for the plot line
        lineplot (bool): Whether to use line plots or scatter

    Returns:
        None - modifies the ax object in-place
    """
    global fmax

    get_kpoints_frequencies = lib_formats.FORMATS_STRUCTURE[format].get(
        "get_kpoints_frequencies"
    )

    kpoints, frequencies = get_kpoints_frequencies(data)
    kpoints, frequencies = kpoints_manager.slice_paths(kpoints, frequencies, kpath)
    coordinates = kpoints_manager.assign_positions(kpoints)

    k_shape = coordinates.shape
    f_shape = frequencies.shape

    assert f_shape[0] == 1
    assert f_shape[1] == k_shape[0]
    nbands = f_shape[2]
    print("[INFO] plotting", nbands, "bands for", label)

    for band in range(nbands):
        maxf = np.max(frequencies[0, :, band])
        if maxf > fmax:
            fmax = maxf

        if lineplot:
            ax.plot(
                coordinates, 
                frequencies[0, :, band], 
                "--",
                linewidth = 3,
                color=color, 
                label=label if band == 0 else None,
                alpha=0.7
            )
        else:
            ax.scatter(
                coordinates, 
                frequencies[0, :, band], 
                color=color, 
                label=label if band == 0 else None,
                s=15,
                alpha=0.7
            )

    return ax


def parse_args():
    parser = ArgumentParser(
        "This programs reads phonon kpoints and frequencies from multiple sources, and plots them against each other."
    )

    parser.add_argument(
        "--sources", nargs="+", type=str, required=True, help="list of data files."
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        type=str,
        required=True,
        choices=["mp", "ase", "phonopy"],
        help="format of data files (same order as sources)",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        required=True,
        help="labels for sources. eg: DFT, MACE-MP0, MACE_v1 etc ... will be shown in legends.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory to store output files",
    )

    parser.add_argument(
        "--kpath", type=str, required=True, help="kpath for phonons plot. eg GMKG"
    )

    args = parser.parse_args()

    if len(args.formats) == 1 and len(args.sources) > 1:
        args.formats = args.formats * len(args.sources)

    cleaned_labels = []
    for raw_label in args.labels:
        match = re.search(r"strain_([0-9.]+)", raw_label)
        if match:
            cleaned_labels.append(match.group(1))
        else:
            cleaned_labels.append(raw_label)
    args.labels = cleaned_labels

    os.makedirs(args.output_dir, exist_ok=True)

    assert len(args.sources) == len(args.formats) and len(args.sources) == len(
        args.labels
    ), f"Mismatch: Sources ({len(args.sources)}), Formats ({len(args.formats)}), Labels ({len(args.labels)})"

    return args


def initialize_plot(path: str, **kwargs):
    high_simmetry_points = kpoints_manager.string_to_vector(path)
    high_simmetry_positions = kpoints_manager.assign_positions(high_simmetry_points)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.axhline(0, xmin=0, xmax=1, color="gray", lw=1)
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.set_yticks(np.arange(0, 1000, 5))
    ax.set_yticklabels(np.arange(0, 1000, 5))

    ax.set_ylabel("Frequency [THz]")

    for raw_label, high_simmetry_position in zip(path, high_simmetry_positions):
        ax.vlines(
            high_simmetry_position,
            0, 
            ymax=1000,
            color="grey",
            linestyle="--",
            alpha=0.7,
        )

        clean_label = raw_label.upper()
        if clean_label == "G":
            clean_label = r"$\Gamma$"

        ax.text(
            high_simmetry_position, 
            -1.5, 
            clean_label, 
            ha="center", 
            va="top", 
        )

    return fig, ax


def analyze_gamma_point(sources, source_formats, labels, kpath, output_dir):
    kpath_upper = kpath.upper()
    
    # 1. Map out all distinct directions emanating from any Gamma point found
    g_indices = [i for i, char in enumerate(kpath_upper) if char == 'G']
    if not g_indices:
        print("[WARNING] Gamma point ('G') not found in kpath. Skipping analysis.")
        return

    high_simmetry_points = kpoints_manager.string_to_vector(kpath)
    high_simmetry_positions = kpoints_manager.assign_positions(high_simmetry_points)
    
    paths_from_gamma = []
    for g_idx in g_indices:
        gamma_pos = high_simmetry_positions[g_idx]
        
        if g_idx > 0:
            neighbor_label = kpath_upper[g_idx - 1]
            neighbor_pos = high_simmetry_positions[g_idx - 1]
            paths_from_gamma.append({
                "name": f"$\\Gamma \\rightarrow$ {neighbor_label if neighbor_label != 'G' else r'\\Gamma'}",
                "gamma_pos": gamma_pos,
                "direction": -1,
                "boundary_pos": neighbor_pos
            })
        if g_idx < len(kpath_upper) - 1:
            neighbor_label = kpath_upper[g_idx + 1]
            neighbor_pos = high_simmetry_positions[g_idx + 1]
            paths_from_gamma.append({
                "name": f"$\\Gamma \\rightarrow$ {neighbor_label if neighbor_label != 'G' else r'\\Gamma'}",
                "gamma_pos": gamma_pos,
                "direction": 1,
                "boundary_pos": neighbor_pos
            })

    all_source_data = []
    
    for source, fmt, label in zip(sources, source_formats, labels):
        with open(source, "r") as f:
            data = yaml.safe_load(f)
        
        get_kpoints_frequencies = lib_formats.FORMATS_STRUCTURE[fmt].get("get_kpoints_frequencies")
        kpoints, frequencies = get_kpoints_frequencies(data)
        kpoints, frequencies = kpoints_manager.slice_paths(kpoints, frequencies, kpath)
        coordinates = kpoints_manager.assign_positions(kpoints)
        
        freqs_at_path = frequencies[0]
        nbands = freqs_at_path.shape[1]
        n_acoustic = min(3, nbands)
        
        # We also need a reliable tracking of exact structural values explicitly AT Gamma
        first_g_pos = high_simmetry_positions[g_indices[0]]
        global_gamma_idx = np.argmin(np.abs(coordinates - first_g_pos))
        global_gamma_freqs = freqs_at_path[global_gamma_idx, :]
        sorted_band_indices = np.argsort(global_gamma_freqs)
        
        # Save optical frequencies exactly evaluated at the Gamma intercept
        exact_optical_freqs = global_gamma_freqs[sorted_band_indices]
        
        source_path_results = {}
        for p_info in paths_from_gamma:
            g_pos = p_info["gamma_pos"]
            b_pos = p_info["boundary_pos"]
            direction = p_info["direction"]
            
            if direction == -1:
                mask = (coordinates <= g_pos) & (coordinates >= b_pos)
                sorted_indices = np.argsort(coordinates[mask])[::-1]
            else:
                mask = (coordinates >= g_pos) & (coordinates <= b_pos)
                sorted_indices = np.argsort(coordinates[mask])
                
            seg_coords = coordinates[mask][sorted_indices]
            seg_freqs = freqs_at_path[mask][sorted_indices]
            
            path_acoustic_velocities = []
            
            # Perform origin-constrained 4th-order polynomial fits strictly for acoustic branches
            for rank, b in enumerate(sorted_band_indices[:n_acoustic]):
                fit_npoints = min(11, len(seg_coords))
                fit_coords = seg_coords[1:fit_npoints]
                dk_fit = np.abs(fit_coords - g_pos)
                band_freqs = seg_freqs[1:fit_npoints, b]
                
                # Fit omega/dk = c1 + c2*dk + c3*dk^2 + c4*dk^3 
                y_trans = band_freqs / dk_fit
                coeffs = np.polyfit(dk_fit, y_trans, deg=3)
                c1_coefficient = coeffs[3] # Represents accurate linear group velocity slope at dk->0
                path_acoustic_velocities.append(np.abs(c1_coefficient))
                
            source_path_results[p_info["name"]] = {
                "acoustic_velocities": np.array(path_acoustic_velocities)
            }
            
        all_source_data.append({
            "label": label,
            "paths": source_path_results,
            "optical_freqs": exact_optical_freqs, # 1D array ordered by sorted_band_indices
            "sorted_indices": sorted_band_indices,
            "nbands": nbands
        })

    ref_data = all_source_data[-1]
    ref_label = ref_data["label"]
    nbands = ref_data["nbands"]
    n_acoustic = min(3, nbands)
    
    # 1. Save standard text summary file
    txt_path = os.path.join(output_dir, "gamma_comparison_summary.txt")
    with open(txt_path, "w") as out_f:
        out_f.write("===================================================================\n")
        out_f.write("    GAMMA POINT PHONON BRANCHES SUMMARY (NON-REPEATED OPTICAL)\n")
        out_f.write("===================================================================\n")
        out_f.write(f"Reference Source (Last Element): {ref_label}\n\n")
        
        out_f.write("--- 1. ACOUSTIC BRANCH SOUND VELOCITIES (PER PATH SEGMENT) ---\n")
        for p_info in paths_from_gamma:
            p_name = p_info["name"]
            out_f.write(f"Direction Segment: {p_name}\n")
            out_f.write("-" * 75 + "\n")
            for i in range(len(all_source_data) - 1):
                src = all_source_data[i]
                out_f.write(f"{'Branch':<8}{src['label'] + ' Vel':<15}{ref_label + ' (Ref)':<15}{'Rel. Diff':<15}\n")
                for b in range(n_acoustic):
                    v_src = src["paths"][p_name]["acoustic_velocities"][b]
                    v_ref = ref_data["paths"][p_name]["acoustic_velocities"][b]
                    rel_str = f"{(v_src - v_ref) / v_ref:+.2%}" if v_ref != 0 else "0.00%"
                    out_f.write(f"{b+1:<8}{v_src:.3f}{'':<11}{v_ref:.3f}{'':<10}{rel_str:<15}\n")
            out_f.write("\n")
            
        out_f.write("\n--- 2. OPTICAL BRANCH FREQUENCIES (EXACTLY AT GAMMA, NO REPETITION) ---\n")
        out_f.write("-" * 75 + "\n")
        for i in range(len(all_source_data) - 1):
            src = all_source_data[i]
            out_f.write(f"Comparison: {src['label']} vs {ref_label}\n")
            out_f.write(f"{'Branch':<8}{src['label'] + ' Freq':<15}{ref_label + ' (Ref)':<15}{'Rel. Diff':<15}\n")
            for b in range(n_acoustic, nbands):
                v_src = src["optical_freqs"][b]
                v_ref = ref_data["optical_freqs"][b]
                rel_str = f"{(v_src - v_ref) / v_ref:+.2%}" if v_ref != 0 else "0.00%"
                out_f.write(f"{b+1:<8}{v_src:.3f} THz{'':<7}{v_ref:.3f} THz{'':<6}{rel_str:<15}\n")
            out_f.write("\n")
            
    print(f"[INFO] Analysis text summary saved to: {txt_path}")

    # 2. Save structured LaTeX code file containing ONLY relative discrepancies
    tex_path = os.path.join(output_dir, "gamma_comparison_table.tex")
    with open(tex_path, "w") as tex_f:
        tex_f.write("% ---- LaTeX Table Code for Gamma Point Comparison ----\n")
        tex_f.write("\\begin{table}[htbp]\n")
        tex_f.write("  \\centering\n")
        tex_f.write("  \\caption{Relative discrepancies (\\%) of calculated phonon profiles relative to the reference \\textbf{" + ref_label + "}. Acoustic rows present un-repeated group sound velocities evaluated along specific path projections using a constrained polynomial approach ($f(k) = c_1 k + c_2 k^2 + c_3 k^3 + c_4 k^4$). Optical branches are evaluated strictly at the isotropic $\\Gamma$ structural intercept, removing structural path repetition.}\n")
        tex_f.write("  \\label{tab:gamma_phonon_comparison}\n")
        
        align_str = "ccc" + "c" * (len(all_source_data) - 1)
        tex_f.write("  \\begin{tabular}{" + align_str + "}\n")
        tex_f.write("    \\hline\\hline\n")
        
        hdr = "    Path / Domain & Branch & Property Type "
        for src in all_source_data[:-1]:
            hdr += f" & {src['label']}"
        hdr += " \\\\\n"
        tex_f.write(hdr)
        tex_f.write("    \\hline\n")
        
        # --- Section A: Acoustic properties (broken down by path direction) ---
        for p_idx, p_info in enumerate(paths_from_gamma):
            p_name = p_info["name"]
            clean_path_tex = p_name.replace(r"\\Gamma", r"\Gamma").replace(r"\rightarrow", r"\rightarrow")
            
            for b in range(n_acoustic):
                path_col = f"\\midrule {clean_path_tex}" if b == 0 else ""
                row_str = f"    {path_col:<30} & {b+1} & Acoustic Velocity "
                
                v_ref = ref_data["paths"][p_name]["acoustic_velocities"][b]
                for src in all_source_data[:-1]:
                    v_src = src["paths"][p_name]["acoustic_velocities"][b]
                    diff_str = f"{(v_src - v_ref) / v_ref:+.2%}".replace("%", "\\%") if v_ref != 0 else "0.00\\%"
                    row_str += f" & {diff_str}"
                row_str += " \\\\\n"
                tex_f.write(row_str)
                
        # --- Section B: Optical properties (single domain block directly at Gamma) ---
        for b in range(n_acoustic, nbands):
            path_col = "\\midrule Exact $\\Gamma$-point" if b == n_acoustic else ""
            row_str = f"    {path_col:<30} & {b+1} & Optical Frequency  "
            
            v_ref = ref_data["optical_freqs"][b]
            for src in all_source_data[:-1]:
                v_src = src["optical_freqs"][b]
                diff_str = f"{(v_src - v_ref) / v_ref:+.2%}".replace("%", "\\%") if v_ref != 0 else "0.00\\%"
                row_str += f" & {diff_str}"
            row_str += " \\\\\n"
            tex_f.write(row_str)
            
        tex_f.write("    \\hline\\hline\n")
        tex_f.write("  \\end{tabular}\n")
        tex_f.write("\\end{table}\n")
        
    print(f"[INFO] LaTeX formatting code saved to: {tex_path}")


def main():
    args = parse_args()

    fig, ax = initialize_plot(path=args.kpath)

    colors = cm.inferno(np.linspace(0.2, 1, len(args.sources)))
    
    for idx, (source, fmt, label) in enumerate(zip(args.sources, args.formats, args.labels)):
        with open(source, "r") as f:
            data = yaml.safe_load(f)
            lineplot = (idx == len(args.sources)-1)
            color = "k" if lineplot else colors[idx]
            add_plot(ax, data, fmt, args.kpath, label=label, color=color, lineplot=lineplot)

    ax.set_ylim(-1, fmax + DELTAF)
    ax.legend(loc="upper center", frameon=True, shadow=False, bbox_to_anchor=(0.5, 1.5))
    #ax.set_title("Phonon Dispersion Comparison (All Sources)", fontsize=14, pad=10)

    img_output_path = os.path.join(args.output_dir, "phonons_all_sources.pdf")
    plt.tight_layout()
    plt.savefig(img_output_path)
    print(f"[INFO] Consolidated graphical output saved to: {img_output_path}")

    analyze_gamma_point(args.sources, args.formats, args.labels, args.kpath, args.output_dir)

    return 0


if __name__ == "__main__":
    main()