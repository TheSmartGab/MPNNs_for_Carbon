#!/usr/bin/env python3
import argparse
import subprocess
import shlex
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os

def run_file(args):
    """Function executed in parallel for each file."""
    infile, inner_script, output_dir, forwarded_args = args
    base = infile.stem
    output_name = output_dir / base

    cmd = ["python3", str(inner_script), str(infile),
           "--output_name", str(output_name)] + forwarded_args

    print(f"[PID {os.getpid()}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        "Multiprocessing driver for running a script over many input files."
    )

    parser.add_argument("input_dir", type=str)
    parser.add_argument("inner_script", type=str)

    parser.add_argument("--ext", default="xyz",
                        help="File extension to search for")
    parser.add_argument("--output_dir", default="output",
                        help="Directory for output files")
    parser.add_argument("--nprocs", type=int, default=cpu_count(),
                        help="Number of parallel processes to run (default: all CPUs)")

    parser.add_argument(
        "--inner_args",
        type=str,
        default="",
        help="Arguments forwarded as a single string to the inner script. "
             "Example: \"--ecut 500 --kpoints 6 4 1 --format extxyz\""
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    inner_script = Path(args.inner_script).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Forwarded arguments
    forwarded_args = shlex.split(args.inner_args)

    # List all input files
    files = sorted(input_dir.glob(f"*.{args.ext}"))
    if not files:
        print(f"No .{args.ext} files found in {input_dir}")
        return

    print(f"Total files: {len(files)}")
    print(f"Running {args.nprocs} processes in parallel...")

    # Prepare arguments for pool.map
    pool_args = [(f, inner_script, output_dir, forwarded_args) for f in files]

    # Launch multiprocessing pool
    with Pool(processes=args.nprocs) as pool:
        pool.map(run_file, pool_args)

    print("All calculations completed successfully.")

if __name__ == "__main__":
    import os
    main()
