from ase.io import read, write
from ase import Atoms, Atom
import numpy as np
from argparse import ArgumentParser
import os
from datetime import datetime


def parse_args():

    parser=ArgumentParser("This script takes as input a file with ase format, and discards all configurations in which are present forces higher, in modulus, than the given cutoff. for Nequip/MACE models, a cutoff of 100/200 eV/A is reccommended, since, apparently, models are not able to achieve accuracy for both high and 'normal' forces. I experimented this personally while trying to train models on a simple 2-body function with distance as low as 0.5 A.")

    parser.add_argument("input_file", type=str, help="input file with data to modify")
    parser.add_argument("--format", type=str, help="ase compatible format of input file. will be used as output format as well")
    parser.add_argument("--f_cutoff", type=float, help="cutoff for forces. all configurations in which there are forces with greater modulus than this will be discarded.")
    parser.add_argument("--output_file", type=str, required=False, default=None, help="pass this if you do not want to replace inplace the file, but rather make a copy somewhere else. By default, a backup of the original file is created anyway in its same directory named backup.backup. The backup is not created if output file is passed")

    args=parser.parse_args()

    return args


def main():

    args = parse_args()

    # some sanity check
    if not os.path.isfile(args.input_file):
        print("[ERROR] File", args.file, "does not exits or is not a file. exit")
        exit(-1)

    input_dir = os.path.dirname(args.input_file)
    # output dir either for backup or new file
    output_dir = os.path.dirname(args.output_file) if args.output_file else input_dir
    os.makedirs(output_dir, exist_ok=True)

    input_images = read(args.input_file, index=":", format=args.format)
    print("[INFO] Found", len(input_images), "images")
    print("[INFO] Cleaning images with f_cutoff", args.f_cutoff)
    output_images = [
        # long condition, a bit pythonic
        # im.get_forces -> np.array(n_atoms, 3)
        # np.linalg.norm(forces, axis=1) -> np.array(n_atoms)
        # then, python built-in broadcast of args.cutoff
        # >= 0 -> np.array(n_atoms) array of bool, true if force over cutoff is present
        # .any() check if any bool in the array is true
        im for im in input_images if not (np.linalg.norm(im.get_forces(), axis=1) - args.f_cutoff >= 0).any()
    ]
    print("[INFO] Keeping", len(output_images), "images")

    print("[INFO] Writing output")
    if args.output_file:
        print("[INFO] Writing to", args.output_file)
        write(args.output_file, output_images, format=args.format)
    else:
        print("[INFO] Overwriting", args.input_file)
        write(args.input_file, output_images, format=args.format)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") # avoid overwriting backups
        backup_path = os.path.join(input_dir, "backup-" + timestamp + ".backup")
        print("[INFO] Saving backup to", backup_path)
        write(backup_path, input_images, format=args.format)




    return 0

if __name__ == "__main__":
    main()