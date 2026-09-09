from lammps_logfile import read_log
import pandas
import numpy as np

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from pprint import pprint

import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm

def parse_args():

    parser = ArgumentParser(
        "This is an helper script to read and plot lammps thermo files"
    )

    parser.add_argument("--inputfile", "-if", default=None, required=False, type=str)
    parser.add_argument(
        "--inputfiles", "-ifs", nargs="+", default=None, required=False, type=str
    )
    parser.add_argument("--xcol", "-x", type=str)
    parser.add_argument("--ycol", "-y", type=str)
    parser.add_argument("--xlabel", "-xl", type=str)
    parser.add_argument("--ylabel", "-yl", type=str)

    parser.add_argument("--outdir", "-od", type=str, default="THERMO_OUT")
    parser.add_argument("--outname", "-on", type=str, default="thermo.pdf")

    args = parser.parse_args()

    print("Running", __file__, "with args:")
    pprint(args)

    return args


def block(data, nblocks):
    discard = len(data.size) % nblocks
    if len(data) % nblocks != 0:
        print("[WARNING] len of data is not divisible by n")


def plot(x, y, xlabel, ylabel, savepath=None):
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savepath:
        plt.savefig(savepath)


def process_file(filepath, xcol, ycol, xlabel, ylabel, outdir=None, outname=None):

    if outdir and outname:
        dirname = os.path.dirname(filepath)
        outpath = os.path.join(dirname, outdir)
        os.makedirs(outpath, exist_ok=True)
        outfile = os.path.join(outpath, outname)
    else:
        outfile=None

    df = read_log(filepath)

    x = df[xcol]
    y = df[ycol]

    plot(x, y, xlabel, ylabel, outfile)


def main():

    args = parse_args()

    if args.inputfile:
        process_file(
            args.inputfile,
            args.xcol,
            args.ycol,
            args.xlabel,
            args.ylabel,
            args.outdir,
            args.outname
        )

        return 0

    if args.inputfiles:

        func = partial(
            process_file,
            xcol=args.xcol,
            ycol=args.ycol,
            xlabel=args.xlabel,
            ylabel=args.ylabel
        )

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(func, fp) for fp in args.inputfiles]

            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
        
        outfile = os.path.join(args.outdir, args.outname)
        os.makedirs(args.outdir, exist_ok=True)
        plt.savefig(outfile)

        return 0


    print("[ERROR] No file nor files passed, nothing got done")
    return 1


if __name__ == "__main__":
    main()
