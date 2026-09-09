"""Given a file with indeces and an input file, write the selected indeces to the output file. If no validation file is provided. """

from argparse import ArgumentParser
import numpy as np

from ase.io import read, write

def parse_args():
    parser = ArgumentParser(
        """Given a file with indeces and an input file, write the selected indeces to the output file"""
    )

    parser.add_argument("--input_file", type=str, required=True, help="Input file with data. must be ase-readable")
    parser.add_argument("--indices", type=str, required=True, help="File with indeces to load.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file with selected indices.")

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    indices = np.loadtxt(args.indices)

    configs = read(args.input_file, index=":")

    selected_configs = [c for i, c in enumerate(configs) if i in indices]

    write(args.output_file, selected_configs)

    return 0


if __name__ == "__main__":
    main()
