"""Subsample a training extxyz file to a fixed total count.

Configurations are drawn proportionally from each config_type category.
Underrepresented categories (those that would receive zero samples under
strict proportional allocation) are guaranteed at least one configuration,
with the remainder distributed among the larger categories.

Usage example
-------------
python subsample_train.py \
    --infile train.xyz \
    --outfile train_sub.xyz \
    --n_configs 5000 \
    --config_type_keyword config_type \
    --seed 42
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from math import floor
from pprint import pprint
from typing import Dict, List, Tuple

import random

from ase.io import read, write


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(
        description="Subsample a training extxyz file to N configurations "
                    "while preserving config_type proportions and guaranteeing "
                    "at least one sample from underrepresented categories."
    )

    parser.add_argument(
        "--infile", type=str, required=True,
        help="Input training extxyz file."
    )
    parser.add_argument(
        "--outfile", type=str, required=True,
        help="Output subsampled extxyz file."
    )
    parser.add_argument(
        "--n_configs", type=int, required=True,
        help="Total number of configurations to keep."
    )
    parser.add_argument(
        "--config_type_keyword", type=str, default="config_type",
        help="Key in atoms.info that labels the configuration type "
             "(default: 'config_type')."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--save_info", action="store_true",
        help="Save a JSON with per-category allocation details next to --outfile."
    )
    parser.add_argument(
        "--no_shuffle_output", action="store_true",
        help="Keep output in category-grouped order instead of shuffling."
    )

    args = parser.parse_args()

    if args.n_configs < 1:
        parser.error("--n_configs must be >= 1.")

    print("[INFO] Running", __file__, "with config:")
    pprint(vars(args))
    print()

    return args


# ---------------------------------------------------------------------------
# Allocation logic
# ---------------------------------------------------------------------------

def proportional_allocation_with_floor(
    category_sizes: Dict[str, int],
    n_total: int,
    rng: random.Random,
) -> Dict[str, int]:
    """
    Allocate n_total slots across categories proportionally to their sizes,
    guaranteeing every category gets at least 1 slot (capped at its own size).

    Algorithm
    ---------
    1. Give every category a floor of 1 (or its full size if smaller).
    2. Distribute remaining slots proportionally among categories that still
       have room, using largest-remainder rounding to hit n_total exactly.
    3. If n_total >= total dataset size, return the full dataset (no sampling).
    """
    categories = list(category_sizes.keys())
    sizes      = {c: category_sizes[c] for c in categories}
    n_dataset  = sum(sizes.values())

    if n_total >= n_dataset:
        print(
            f"[WARN] Requested n_configs ({n_total}) >= dataset size "
            f"({n_dataset}). Returning the full dataset."
        )
        return dict(sizes)

    # --- Step 1: guarantee floor of 1 per category --------------------------
    allocation: Dict[str, int] = {}
    forced_total = 0
    for c in categories:
        floor_val = min(1, sizes[c])
        allocation[c] = floor_val
        forced_total  += floor_val

    if forced_total > n_total:
        # More categories than requested configs: keep one from each of the
        # n_total largest categories (deterministic, then random tie-break).
        sorted_cats = sorted(categories, key=lambda c: (-sizes[c], c))
        allocation  = {c: 0 for c in categories}
        chosen = sorted_cats[:n_total]
        rng.shuffle(chosen)          # random tie-break among equals
        for c in chosen:
            allocation[c] = 1
        return allocation

    remaining = n_total - forced_total

    # --- Step 2: distribute remaining slots proportionally ------------------
    # Only categories that have room left participate.
    eligible = {c: sizes[c] - allocation[c] for c in categories if sizes[c] > allocation[c]}
    eligible_total = sum(eligible.values())

    if eligible_total == 0 or remaining == 0:
        return allocation

    # Compute real-valued quotas
    quotas = {c: remaining * room / eligible_total for c, room in eligible.items()}

    # Floor each quota and track remainders
    floor_alloc:     Dict[str, int]   = {c: floor(q)       for c, q in quotas.items()}
    remainders:      Dict[str, float] = {c: quotas[c] - floor_alloc[c] for c in quotas}

    # How many extra slots do we still need to hand out?
    extra = remaining - sum(floor_alloc.values())

    # Give extras to categories with largest remainders (random tie-break)
    sorted_by_remainder = sorted(
        remainders.keys(),
        key=lambda c: (-remainders[c], rng.random())
    )
    for c in sorted_by_remainder[:extra]:
        floor_alloc[c] += 1

    # Merge and cap at available room
    for c, add in floor_alloc.items():
        allocation[c] += min(add, eligible[c])

    # Sanity: total should equal n_total (floating point may cause off-by-one)
    diff = n_total - sum(allocation.values())
    if diff != 0:
        # Nudge the largest category
        largest = max(
            (c for c in categories if allocation[c] < sizes[c]),
            key=lambda c: sizes[c],
            default=None,
        )
        if largest is not None:
            allocation[largest] = min(allocation[largest] + diff, sizes[largest])

    return allocation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"[INFO] Reading frames from '{args.infile}' …")
    frames = read(args.infile, index=":")
    n_dataset = len(frames)
    print(f"[INFO] Loaded {n_dataset} frames.")

    # ------------------------------------------------------------------
    # 2. Group by config_type
    # ------------------------------------------------------------------
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, atoms in enumerate(frames):
        ctype = str(atoms.info.get(args.config_type_keyword, "unknown"))
        groups[ctype].append(i)
    groups = dict(groups)

    category_sizes = {c: len(idxs) for c, idxs in groups.items()}

    print(f"\n[INFO] Found {len(groups)} config_type(s) in dataset:")
    print(f"  {'Config type':<40}  {'count':>7}  {'fraction':>9}")
    print("  " + "-" * 60)
    for ctype in sorted(groups):
        n = category_sizes[ctype]
        frac = n / n_dataset
        print(f"  {ctype:<40}  {n:>7}  {frac:>9.4f}")
    print()

    # ------------------------------------------------------------------
    # 3. Compute allocation
    # ------------------------------------------------------------------
    allocation = proportional_allocation_with_floor(category_sizes, args.n_configs, rng)

    n_selected = sum(allocation.values())

    # ------------------------------------------------------------------
    # 4. Sample
    # ------------------------------------------------------------------
    selected_indices: List[int] = []
    info_records: List[dict]    = []

    print(f"[INFO] Allocation (target total: {args.n_configs}, actual: {n_selected}):")
    print(f"  {'Config type':<40}  {'avail':>6}  {'alloc':>6}  {'frac_in':>9}  {'frac_out':>9}")
    print("  " + "-" * 76)

    for ctype in sorted(groups):
        avail  = category_sizes[ctype]
        n_pick = allocation[ctype]
        idxs   = groups[ctype]

        picked = rng.sample(idxs, n_pick)
        selected_indices.extend(picked)

        frac_in  = avail  / n_dataset  if n_dataset  else 0
        frac_out = n_pick / n_selected if n_selected else 0
        print(
            f"  {ctype:<40}  {avail:>6}  {n_pick:>6}  "
            f"{frac_in:>9.4f}  {frac_out:>9.4f}"
        )

        info_records.append(
            {
                "config_type":    ctype,
                "available":      avail,
                "allocated":      n_pick,
                "fraction_in":    round(frac_in,  6),
                "fraction_out":   round(frac_out, 6),
                "selected_indices": sorted(picked),
            }
        )

    print("  " + "-" * 76)
    print(f"  {'TOTAL':<40}  {n_dataset:>6}  {n_selected:>6}")
    print()

    # ------------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------------
    if not args.no_shuffle_output:
        rng.shuffle(selected_indices)
    else:
        selected_indices.sort()

    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)

    subset = [frames[i] for i in selected_indices]
    write(args.outfile, subset)
    print(f"[INFO] Wrote {len(subset)} configurations to '{args.outfile}'.")

    # ------------------------------------------------------------------
    # 6. Optionally save allocation info
    # ------------------------------------------------------------------
    if args.save_info:
        info_path = os.path.splitext(args.outfile)[0] + "_subsample_info.json"
        with open(info_path, "w") as fh:
            json.dump(
                {
                    "infile":    args.infile,
                    "outfile":   args.outfile,
                    "n_configs": args.n_configs,
                    "n_actual":  n_selected,
                    "seed":      args.seed,
                    "categories": info_records,
                },
                fh,
                indent=2,
            )
        print(f"[INFO] Allocation info saved to '{info_path}'.")


if __name__ == "__main__":
    main()