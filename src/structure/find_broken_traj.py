"""This scripts aims to authomatically find trajectories in which graphene broke, read the last frame, and simlink them in another directory. This can be usefull if you are measuring the breaking rate of a graphene sheet, and are iterating runs to avoid running for a long time broken trajectories"""

from argparse import ArgumentParser
import os
import sys

from ase.io import read, write

from pprint import pprint
from structural_analysis import broken

from pathlib import Path

import multiprocessing
import concurrent
from tqdm import tqdm
from functools import partial

def parse_args():

    parser = ArgumentParser(
        "Find trajectories that broke, simlink them in another directory"
    )

    parser.add_argument("--inputdir", "-i", type=str)
    parser.add_argument("--outputdir", "-o")

    parser.add_argument(
        "--ext", "-e",
        type=str,
        required=False,
        default="traj",
        help="extension of files with data. beware that ase binary .traj files allows lazy loading, whereas lammps .dump files do not!"
    )

    parser.add_argument(
        "--cutoff", "-c", type=float, default=2.0, help="cutoff distance for nns."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=3,
        help="threshold of neighbours below which an atom is consider belonging to the boundary",
    )
    parser.add_argument(
        "--count_threshold", "-ct",
        type=int,
        default=10,
        help="minimum number of atoms on the boundary for the structure to be consider broken",
    )

    parser.add_argument("--ncores", "-nc", type=int, default=16)

    args = parser.parse_args()

    print("[INFO] Runing File", __file__, "with args:")
    pprint(args)

    return args

def process_file(filepath, args) -> None:
    filename = Path(filepath).stem
    atoms = read(filepath, index="-1")
    b = broken(atoms=atoms, cutoff=args.cutoff, threshold=args.threshold, counthreshold=args.count_threshold)
    if not b:
        dst = os.path.join(args.outputdir, filename+".lammps-data")
        write(dst, atoms, velocities = True, format="lammps-data")
        return

    else:
        return


def main():
    args = parse_args()

    os.makedirs(args.outputdir, exist_ok=True)

    files = os.listdir(args.inputdir)
    filepaths = [os.path.join(args.inputdir, f) for f in files if f.endswith(args.ext)]

    print("="*60)
    print("[INFO] Filepaths found:")
    pprint(filepaths)
    print("="*60)

    pool = multiprocessing.Pool(args.ncores)
    worker = partial(process_file, args=args)
    for _ in tqdm(pool.imap(worker, filepaths), total=len(filepaths)):
        pass


if __name__ == "__main__":
    main()
