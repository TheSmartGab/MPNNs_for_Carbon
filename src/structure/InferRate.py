import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from pprint import pprint
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# Assuming these are available in your environment
# from lammps_logfile import read_log
# from ase.io import write, Trajectory
from bayes.rate import infer_rate
from structural_analysis import (
    parallel_count_boundary_trajs,
    find_breaking_step_list_parallel,
)

def parse_args():
    parser = ArgumentParser(
        description="Bayesian inference on graphene breaking rates with robust discard logic."
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory for .traj files")
    parser.add_argument("--outdir", type=str, default="RATE_OUT")
    parser.add_argument("--ext", type=str, default=".traj")
    parser.add_argument("--normal_lens", nargs="+", type=int, default=[])
    parser.add_argument("--lens_with_discard", nargs="+", type=int, default=[])
    parser.add_argument("--discards", nargs="+", type=int, default=[])
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--cutoff", type=float, default=2.0)
    parser.add_argument("--skin", type=float, default=0.1)
    parser.add_argument("--boundary_threshold", type=int, default=3)
    parser.add_argument("--count_threshold", type=int, default=10)
    parser.add_argument("--save_counts", action="store_true")
    parser.add_argument("--nwalkers", default=16, type=int)
    parser.add_argument("--nsteps", default=10000, type=int)
    parser.add_argument("--burnin", default=1000, type=int)

    args = parser.parse_args()
    print("[INFO] Running with config:")
    pprint(vars(args))
    return args

def get_discard_map(args):
    """Pre-computes sets and dicts for O(1) lookup performance."""
    normal_set = set(args.normal_lens)
    discard_map = dict(zip(args.lens_with_discard, args.discards))
    return normal_set, discard_map

def custom_subtract(length, normal_set, discard_map):
    if length in normal_set:
        return 0
    if length in discard_map:
        return discard_map[length]
    
    print(f"[ERROR] len {length} not recognized. Check simulation steps.")
    exit(-1)

def main():
    args = parse_args()

    args.outdir = os.path.join(args.root, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Data Discovery
    traj_files, found_traj_files, found_counts = [], [], []
    for root, _, files in os.walk(args.root):
        for f in files:
            if f.endswith(args.ext):
                filepath = os.path.join(root, f)
                jsonpath = Path(filepath).with_suffix(".json")
                if jsonpath.exists():
                    with open(jsonpath, "r") as fj:
                        found_counts.append(json.load(fj))
                    found_traj_files.append(filepath)
                else:
                    traj_files.append(filepath)

    # 2. Processing
    results = parallel_count_boundary_trajs(
        traj_files, cutoff=args.cutoff, skin=args.skin, threshold=args.boundary_threshold
    )

    if args.save_counts:
        for path, value in results.items():
            savefile = Path(path).with_suffix(".json")
            with open(savefile, "w") as f:
                json.dump([int(x) for x in value["counts"]], f, indent=4)

    # Combine new and cached data
    all_traj_paths = list(results.keys()) + found_traj_files
    all_counts = [v["counts"] for v in results.values()] + found_counts

    # 3. Robust Logic for Breaking Steps
    # Get raw breaking steps (None if didn't break)
    raw_breaking_steps = np.array(
        find_breaking_step_list_parallel(all_counts, threshold=args.count_threshold), 
        dtype=object
    )
    
    lengths = np.array([len(c) for c in all_counts])
    normal_set, discard_map = get_discard_map(args)
    subtract_vals = np.array([custom_subtract(l, normal_set, discard_map) for l in lengths])

    # --- THE FILTERING BLOCK ---
    # define three categories:
    # BROKEN_VALID: Broke after the discard period.
    # CENSORED: Never broke.
    # DISCARDED: Broke during discard period (Invalid/Negative time) -> Remove completely.

    is_broken_raw = np.array([s is not None for s in raw_breaking_steps])
    
    # Calculate relative breaking steps for those that broke
    relative_breaking_steps = np.zeros_like(lengths, dtype=float)
    for i in range(len(raw_breaking_steps)):
        if is_broken_raw[i]:
            relative_breaking_steps[i] = raw_breaking_steps[i] - subtract_vals[i]

    # Masks
    mask_valid_break = is_broken_raw & (relative_breaking_steps > 0)
    mask_censored = ~is_broken_raw
    mask_invalid = is_broken_raw & (relative_breaking_steps <= 0)

    if np.any(mask_invalid):
        print(f"[WARNING] Discarding {np.sum(mask_invalid)} files that broke during burn-in/discard period.")
        for idx in np.where(mask_invalid)[0]:
            print(f"  -> {all_traj_paths[idx]}")

    # Prepare inputs for infer_rate
    # We only keep indices that are either valid breaks or censored
    keep_indices = mask_valid_break | mask_censored
    
    final_data = relative_breaking_steps[mask_valid_break] * args.timestep
    final_censored_mask = mask_censored[keep_indices]
    final_Tmax = (lengths[keep_indices] - 1 - subtract_vals[keep_indices]) * args.timestep
    
    # Validation
    if len(final_data) == 0:
        print("[ERROR] No valid breaking events found after filtering. Cannot infer rate.")
        return -1

    # 4. Bayesian Inference
    sampler = infer_rate(
        data=final_data,
        censored=final_censored_mask,
        Tmax=final_Tmax,
        tdelta=args.timestep,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps
    )
    
    samples = sampler.get_chain(discard=args.burnin, flat=True)

    # 5. Statistics and Plotting
    quantiles = np.percentile(samples, [2.5, 50, 97.5])
    mean, std = np.mean(samples), np.std(samples)

    stats = {
        "q0_025": quantiles[0], "q0_500": quantiles[1], "q0_975": quantiles[2],
        "mean": mean, "std": std, "samples": samples.tolist()
    }

    with open(os.path.join(args.outdir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=100, alpha=0.7, color='skyblue', label="Posterior")
    plt.axvline(mean, color='black', label=f"Mean: {mean:.3f}")
    plt.axvline(quantiles[0], color='red', linestyle='--', label="95% CI")
    plt.axvline(quantiles[2], color='red', linestyle='--')
    plt.xlabel("Rate [1/ps]")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Posterior Distribution of Breaking Rate")
    plt.savefig(os.path.join(args.outdir, "rate_distribution.svg"))

    return 0

if __name__ == "__main__":
    main()