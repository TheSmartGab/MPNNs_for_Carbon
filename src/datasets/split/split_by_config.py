"""Split input file by configuration types into train / validation / test sets.

Each split preserves the ratio of config_type categories found in the full
dataset.  Categories with too few structures can be excluded from validation
and/or test via --min_val and --min_test thresholds.

Usage example
-------------
python split_extxyz.py \
    --infile all.xyz \
    --outdir splits/ \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --min_val 20 \
    --min_test 20 \
    --seed 42
"""

from __future__ import annotations

import json
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
from typing import Dict, List, Optional

from ase.io import read, write


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(
        description="Split an extxyz file into train / val / test sets "
                    "while preserving config_type ratios."
    )

    parser.add_argument(
        "--infile", type=str, required=True,
        help="Input extxyz (or any ASE-readable) file."
    )
    parser.add_argument(
        "--config_type_keyword", type=str, default="config_type",
        help="Key in the atoms.info dict that labels the configuration type "
             "(default: 'config_type')."
    )
    parser.add_argument(
        "--outdir", type=str, required=True,
        help="Directory where train.xyz / val.xyz / test.xyz are written."
    )
    parser.add_argument(
        "--info_file", type=str, default=None,
        help="Optional JSON produced by a previous inspection run that "
             "contains per-category counts and indices.  When supplied the "
             "file is read instead of re-scanning --infile."
    )

    # Split ratios
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Fraction of each category assigned to training (default: 0.8)."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of each (eligible) category assigned to validation "
             "(default: 0.1)."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Fraction of each (eligible) category assigned to testing "
             "(default: 0.1)."
    )

    # Thresholds
    parser.add_argument(
        "--min_val", type=int, default=None,
        help="Minimum category size to be included in validation.  "
             "Categories smaller than this go entirely to training."
    )
    parser.add_argument(
        "--min_test", type=int, default=None,
        help="Minimum category size to be included in testing.  "
             "Categories smaller than this go entirely to training."
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--save_info", action="store_true",
        help="Save a JSON with per-category counts and split indices alongside "
             "the output files."
    )

    args = parser.parse_args()

    # Basic validation of ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        parser.error(
            f"--train_ratio + --val_ratio + --test_ratio must sum to 1.0 "
            f"(got {total:.4f})."
        )

    print("[INFO] Running", __file__, "with config:")
    pprint(vars(args))
    print()

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_by_config_type(
    frames: List,
    keyword: str,
) -> Dict[str, List[int]]:
    """Return {config_type: [frame_index, ...]} mapping."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, atoms in enumerate(frames):
        ctype = atoms.info.get(keyword, "unknown")
        groups[str(ctype)].append(i)
    return dict(groups)


def split_indices(
    indices: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    use_val: bool,
    use_test: bool,
    rng: random.Random,
) -> Dict[str, List[int]]:
    """Randomly split a list of indices into train / val / test."""
    shuffled = indices[:]
    rng.shuffle(shuffled)
    n = len(shuffled)

    if use_val and use_test:
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            # Not enough items even after rounding — put everything in train
            return {"train": shuffled, "val": [], "test": []}
        train = shuffled[:n_train]
        val   = shuffled[n_train : n_train + n_val]
        test  = shuffled[n_train + n_val :]
    elif use_val and not use_test:
        n_val  = max(1, round(n * val_ratio / (train_ratio + val_ratio)))
        n_train = n - n_val
        if n_train < 1:
            return {"train": shuffled, "val": [], "test": []}
        train = shuffled[:n_train]
        val   = shuffled[n_train:]
        test  = []
    elif use_test and not use_val:
        n_test = max(1, round(n * test_ratio / (train_ratio + test_ratio)))
        n_train = n - n_test
        if n_train < 1:
            return {"train": shuffled, "val": [], "test": []}
        train = shuffled[:n_train]
        val   = []
        test  = shuffled[n_train:]
    else:
        train = shuffled
        val   = []
        test  = []

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load frames
    # ------------------------------------------------------------------
    print(f"[INFO] Reading frames from '{args.infile}' …")
    frames = read(args.infile, index=":")
    print(f"[INFO] Loaded {len(frames)} frames.")

    # ------------------------------------------------------------------
    # 2. Group by config_type (from info_file or by scanning)
    # ------------------------------------------------------------------
    if args.info_file is not None:
        print(f"[INFO] Loading category info from '{args.info_file}' …")
        with open(args.info_file) as fh:
            groups: Dict[str, List[int]] = json.load(fh)
    else:
        print(f"[INFO] Grouping frames by '{args.config_type_keyword}' …")
        groups = group_by_config_type(frames, args.config_type_keyword)

    print(f"[INFO] Found {len(groups)} config_type(s):")
    for ctype, idxs in sorted(groups.items()):
        print(f"       {ctype:40s}  n={len(idxs)}")
    print()

    # ------------------------------------------------------------------
    # 3. Split each category
    # ------------------------------------------------------------------
    split_map: Dict[str, Dict[str, List[int]]] = {}

    for ctype, idxs in groups.items():
        n = len(idxs)
        use_val  = (args.min_val  is None) or (n >= args.min_val)
        use_test = (args.min_test is None) or (n >= args.min_test)

        if not use_val:
            print(
                f"[WARN] '{ctype}' has {n} frames < min_val={args.min_val}: "
                "excluded from validation."
            )
        if not use_test:
            print(
                f"[WARN] '{ctype}' has {n} frames < min_test={args.min_test}: "
                "excluded from testing."
            )

        split_map[ctype] = split_indices(
            idxs,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            use_val=use_val,
            use_test=use_test,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # 4. Collect global index lists and write output files
    # ------------------------------------------------------------------
    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    for ctype, splits in split_map.items():
        train_idx.extend(splits["train"])
        val_idx.extend(splits["val"])
        test_idx.extend(splits["test"])

    # Sort to preserve original ordering within each split
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    splits_to_write = {
        "train": train_idx,
        "val":   val_idx,
        "test":  test_idx,
    }

    print()
    for split_name, idxs in splits_to_write.items():
        if not idxs:
            print(f"[INFO] {split_name}: 0 frames — skipping file.")
            continue
        out_path = os.path.join(args.outdir, f"{split_name}.xyz")
        subset = [frames[i] for i in idxs]
        write(out_path, subset)
        print(f"[INFO] {split_name:5s}: {len(subset):6d} frames → '{out_path}'")

    # ------------------------------------------------------------------
    # 5. Print per-category summary
    # ------------------------------------------------------------------
    print()
    print(f"{'Config type':<40}  {'total':>6}  {'train':>6}  {'val':>6}  {'test':>6}")
    print("-" * 68)
    for ctype, splits in sorted(split_map.items()):
        n_total = len(groups[ctype])
        n_train = len(splits["train"])
        n_val   = len(splits["val"])
        n_test  = len(splits["test"])
        print(f"{ctype:<40}  {n_total:>6}  {n_train:>6}  {n_val:>6}  {n_test:>6}")
    print("-" * 68)
    total_all   = sum(len(v) for v in groups.values())
    total_train = len(train_idx)
    total_val   = len(val_idx)
    total_test  = len(test_idx)
    print(f"{'TOTAL':<40}  {total_all:>6}  {total_train:>6}  {total_val:>6}  {total_test:>6}")

    # ------------------------------------------------------------------
    # 6. Optionally save split info JSON
    # ------------------------------------------------------------------
    if args.save_info:
        info = {
            ctype: {
                split: idxs
                for split, idxs in splits.items()
            }
            for ctype, splits in split_map.items()
        }
        info_path = os.path.join(args.outdir, "split_info.json")
        with open(info_path, "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"\n[INFO] Split info saved to '{info_path}'.")


if __name__ == "__main__":
    main()