import numpy as np
from ase.io import read
import os.path
import json

from argparse import ArgumentParser

DEFAULT_OUTPUT_NAME = "length.json"


def main():

    parser = ArgumentParser(
        "read lamps-data file, and print to a txt the lenghts of the primitive vectors of the cell. Used, for example, to compare equilibrium cells from different models"
    )

    parser.add_argument("--file", required=True, help="lammps-data file")
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="output file path. If None is give, it will default to length.json in the same directory as the input file",
    )

    args = parser.parse_args()

    # sanity check the path
    if not os.path.exists(args.file):
        print("[ERROR], prvided file does not exists. exit")
        exit(-1)

    file_dir = os.path.dirname(args.file)

    # manage output file
    if args.output:
        output = args.output
    else:
        output=os.path.join(file_dir, DEFAULT_OUTPUT_NAME)
    
    # read input config
    # note that the last frame is considered. if this is not the wanted behaviour, change this logic
    frame = read(args.file, index="-1", format = "lammps-data")

    lenghts = np.array([np.sqrt(np.sum(v**2)) for v in frame.cell])

    lengths_dict = {f"a_{i}" : lenghts[i] for i in  range(3)}
    with open(output, "w") as o:
        json.dump(lengths_dict, o, indent = 4)

    return 0

if __name__ == "__main__":
    main()
