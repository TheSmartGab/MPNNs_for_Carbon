from ase.io import read, write
from argparse import ArgumentParser
import os
import numpy as np

def main():

    parser = ArgumentParser(
        """This program aims to merge two distinct datasets. A new training file is created with the number of training points from each dataset specified b y <nsamples> (This could be avoided by using Nequip and MACE interfaces to SQL databases to avoid useless bloating). In the validation and test directories, some simlinks are created instead, without mixing the sets, so that these can be consulted independentely."""
    )

    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="list of directories with datasets to merge. they should be organised as split/training/training.<ext> split/validation/validation.<ext> split/test/test.<ext>, where <ext> is the extension parameter. ",
    )
    parser.add_argument(
        "--extension",
        required=True,
        type=str,
        help="extension of training, validation and test files.",
    )
    parser.add_argument(
        "--format",
        required=False,
        type=str,
        help="input data format. must be ase compatible",
        default="extxyz"
    )
    parser.add_argument(
        "--nsamples",
        required=True,
        type=int,
        nargs="+",
        help="number of samples in the training files to keep from each ataset. if you want to use all points, pass 0.",
    )

    parser.add_argument(
        "--names",
        required=True,
        nargs="+",
        type=str,
        help="naames of datasets. symbolic links will be created in validation and test with names <name>.<output_extension>"
    )

    parser.add_argument(
        "--output_directory",
        required=True,
        type=str,
        help="output directory where split will be created. Either pass an absoute path or relative path from working dir.",
    )
    parser.add_argument(
        "--output_extension",
        required = False,
        type=str,
        help = "output file extension",
        default="extxyz"
    )
    parser.add_argument(
        "--output_format",
        required=False,
        type=str,
        help="output file format. must be ase compatible",
        default="extxyz"
    )

    parser.add_argument(
        "--splits",
        required=False, 
        type=str,
        help="""splits name for input and output. default works with standard organization of the project. if you named differently directories and files chnge this at will. for example, have split/validation/validation.extxyz . if you used split/val/val.extxyz use --splits train val test. or can be used to add predict. note that what you put at index 0 is considered to be the training set, therefore, <nsamples> samples from each dataset are retained for the 0-index split.""",
        nargs="+",
        default=["training", "validation", "test"]
    )

    args=parser.parse_args()

    # Create output directory
    os.makedirs(args.output_directory, exist_ok=False)
    SPLIT_OUT_DIR = os.path.join(os.path.join(args.output_directory, "split"))
    os.makedirs(SPLIT_OUT_DIR)
    OUT_DIRS = {
        split : os.path.join(SPLIT_OUT_DIR, split) for split in args.splits
    }
    for split in args.splits:
        os.makedirs(OUT_DIRS[split])

    out_train_file = os.path.join(OUT_DIRS[args.splits[0]], args.splits[0]+"."+args.output_extension)

    # create simbolic links to non training files
    for d, name in zip(args.directories, args.names):
        for split in args.splits[1:]:

            # create symlink    
            src_abs = os.path.abspath(os.path.join(d, "split", split, f"{split}.{args.extension}"))
            dst_dir = os.path.dirname(os.path.join(OUT_DIRS[split], f"{name}.{args.output_extension}"))
            src_rel = os.path.relpath(src_abs, dst_dir)

            if not os.path.isfile(src_abs):
                print(f"[ERROR] {src_rel} does not exist")
                
            # store relative path to make project portable
            os.symlink(src_rel, os.path.join(OUT_DIRS[split], f"{name}.{args.output_extension}"))



    # create empy list for output data organized in a dict. 
    out_train_atoms = []

    for nsamples, dataset_dir in zip(args.nsamples, args.directories):
        IN_SPLIT = os.path.join(dataset_dir, "split")
        IN_DIR = os.path.join(IN_SPLIT, args.splits[0])
        IN_FILE = os.path.join(IN_DIR, args.splits[0]+"."+args.extension)

        # bool that controls training dataset
        # the 0 index split is considered as the training dataset
        images = read(IN_FILE, index=":", format=args.format)
        if nsamples != 0:
            images = [images[i] for i in np.random.randint(low=0, high=len(images), size=nsamples)]
        out_train_atoms.extend(images)

    write(out_train_file, out_train_atoms, format=args.output_format)

    return 0

if __name__ == "__main__":
    main()
