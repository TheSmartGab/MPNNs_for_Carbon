from argparse import ArgumentParser
import os
from ase.io import read, write


def main():
    parser = ArgumentParser(
        description="""This program converts all files in an input directory containing quantum espresso outout to extxyz format in a single file."""
    )

    parser.add_argument("input_dir", type=str, help="directory with input files.")
    parser.add_argument("output_file", type=str, help="output extxyz file")

    parser.add_argument("--extension", required=False, type=str, default="out", help="extension of datafiles (default is out)")
    

    args = parser.parse_args()

    # sanity check
    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] {args.input_dir} does not exists. exit.")
        exit(-1)

    files = [f for f in os.listdir(args.input_dir) if f.endswith("."+args.extension)]
    print(f"[INFO] {len(files)} files found.")

    images = []

    for file in files:
        input_file = os.path.join(args.input_dir, file)
        new_images = read(input_file, index=":")
        images.extend(new_images)

    write(args.output_file, images, format="extxyz")

if __name__ == "__main__":
    main()
