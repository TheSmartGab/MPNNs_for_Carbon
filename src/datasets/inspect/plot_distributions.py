import os
import json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm
from numba import njit
from ase.io import read
import argcomplete
from HistogramStats import HistogramStats

STRESS_INDEX_LITERAL_MAP = ["xx", "yy", "zz", "yz", "xz", "xy"]
FORCE_INDEX_LITERAL_MAP = ["x", "y", "z"]

@njit(fastmath=True)
def force_modulus(forces):
    n = forces.shape[0]
    out = np.empty(n)
    for i in range(n):
        fx, fy, fz = forces[i]
        out[i] = np.sqrt(fx*fx + fy*fy + fz*fz)
    return out

@njit(fastmath=True)
def pressure_from_stress(stress):
    return -(stress[0] + stress[1] + stress[2]) / 3.0

def collect_input_files(args):
    if args.input_file: return [args.input_file]
    files = []
    for root, _, fs in os.walk(args.input_dir):
        for f in fs:
            if f.endswith(args.format): files.append(os.path.join(root, f))
    return sorted(files)

def process_files_with_progress(file_list, fmt, stats):
    for filepath in tqdm(file_list, desc="Processing files", unit="file"):
        configs = read(filepath, format=fmt, index=":")
        for atoms in configs:
            try:
                stats["energy"].update(atoms.get_potential_energy() / len(atoms))
                forces = atoms.get_forces()
                stats["force_modulus"].update(force_modulus(forces))
                stats["force_x"].update(forces[:, 0]); stats["force_y"].update(forces[:, 1]); stats["force_z"].update(forces[:, 2])
                stats["force_components"].update(forces.flatten())
                stress = atoms.get_stress()
                stats["pressure"].update(pressure_from_stress(stress))
                for i, name in enumerate(STRESS_INDEX_LITERAL_MAP):
                    stats[f"stress_{name}"].update(stress[i])
            except Exception: continue

def get_global_ranges(file_list, fmt):
    ranges = {"energy": [np.inf, -np.inf], "force": [np.inf, -np.inf], "stress": [np.inf, -np.inf]}
    for filepath in tqdm(file_list, desc="Scanning ranges", unit="file"):
        try:
            for atoms in read(filepath, format=fmt, index=":"):
                e = atoms.get_potential_energy() / len(atoms)
                ranges["energy"][0], ranges["energy"][1] = min(ranges["energy"][0], e), max(ranges["energy"][1], e)
                f = atoms.get_forces()
                ranges["force"][0], ranges["force"][1] = min(ranges["force"][0], np.min(f)), max(ranges["force"][1], np.max(f))
                s = atoms.get_stress()
                ranges["stress"][0], ranges["stress"][1] = min(ranges["stress"][0], np.min(s)), max(ranges["stress"][1], np.max(s))
        except: continue
    return ranges

def init_stats_dynamic(nbins, ranges):
    def buffer(r):
        margin = (r[1] - r[0]) * 0.05 if r[1] > r[0] else 0.1
        return r[0] - margin, r[1] + margin
    e_min, e_max = buffer(ranges["energy"]); f_min, f_max = buffer(ranges["force"]); s_min, s_max = buffer(ranges["stress"])
    f_mod_max = np.sqrt(3 * (max(abs(f_min), abs(f_max))**2))
    stats = {
        "energy": HistogramStats(nbins, e_min, e_max),
        "force_modulus": HistogramStats(nbins, 0, f_mod_max),
        "force_x": HistogramStats(nbins, f_min, f_max),
        "force_y": HistogramStats(nbins, f_min, f_max),
        "force_z": HistogramStats(nbins, f_min, f_max),
        "force_components": HistogramStats(nbins, f_min, f_max),
        "pressure": HistogramStats(nbins, s_min, s_max)
    }
    for k in STRESS_INDEX_LITERAL_MAP: stats[f"stress_{k}"] = HistogramStats(nbins, s_min, s_max)
    return stats

def plot_histogram(stat, name, xlabel, out_dir, logscale=False, vlines_xs=[], vlines_cols=[], vlines_labs=[], legend_outside=False):
    plt.figure(figsize=(8, 6))
    centers = 0.5 * (stat.edges[:-1] + stat.edges[1:])
    plt.bar(centers, stat.hist, width=centers[1] - centers[0])
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel(xlabel); plt.ylabel("counts")
    if logscale: plt.yscale("log")
    
    y_max = np.max(stat.hist)
    for i, (x, c) in enumerate(zip(vlines_xs, vlines_cols)):
        plt.vlines(x, 1e-5 if logscale else 0, y_max, colors=c, linestyles="--", linewidth = 4, label=vlines_labs[i] if i < len(vlines_labs) else None)
    
    if vlines_labs:
        if legend_outside:
            plt.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=len(vlines_labs), fontsize = 20)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
        else:
            plt.legend(); plt.tight_layout()
    else: plt.tight_layout()
    
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{name}.pdf")); plt.close()

def main():
    parser = ArgumentParser("Plot distribution of quantities in datasets."); parser.add_argument("--input_file", type=str); parser.add_argument("--input_dir", type=str)
    parser.add_argument("--format", default="extxyz"); parser.add_argument("--out_dir", required=True); parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--logscale", action="store_true"); parser.add_argument("--E0", type=float); parser.add_argument("--ecut", type=float); parser.add_argument("--fcut", type=float)
    args = parser.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    
    files = collect_input_files(args)
    stats = init_stats_dynamic(args.nbins, get_global_ranges(files, args.format))
    process_files_with_progress(files, args.format, stats)
    
    # Plotting logic
    e_xs, e_cols, e_labs = [], [], []
    cutcolor="#065b00"
    if args.ecut is not None: e_xs.append(args.ecut); e_cols.append(cutcolor); e_labs.append(f"Energy cut={args.ecut:.1f} eV/atom")
    if args.E0 is not None: e_xs.append(args.E0); e_cols.append("red"); e_labs.append(f"E0={args.E0:.3f} eV")
    plot_histogram(stats["energy"], "energy", "eV/atom", args.out_dir, args.logscale, e_xs, e_cols, e_labs, legend_outside=True)
    
    f_xs, f_cols, f_labs = [], [], []
    if args.fcut is not None:
        f_xs = [-args.fcut, args.fcut]; f_cols = [cutcolor, cutcolor]; f_labs = [f"Force cut {args.fcut:.0f} $\\mathrm{{eV/\\AA}}$"]
    plot_histogram(stats['force_components'], "F_all", r"$\mathrm{eV/\AA}$", args.out_dir, args.logscale, f_xs, f_cols, f_labs, legend_outside=True)

    # Standard plots
    for name, xlabel in [("force_modulus", r"$eV/\AA$"), ("force_x", "Fx"), ("force_y", "Fy"), ("force_z", "Fz"), ("pressure", r"$eV/\AA^3$")]:
        plot_histogram(stats[name], name, xlabel, args.out_dir, args.logscale)
    for comp in STRESS_INDEX_LITERAL_MAP:
        plot_histogram(stats[f"stress_{comp}"], f"stress_{comp}", r"$eV/\AA^3$", args.out_dir, args.logscale)

if __name__ == "__main__":
    main()