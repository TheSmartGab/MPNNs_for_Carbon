import os
from ase.io import read, write
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Convert LAMMPS .dump files to ASE .traj format. Usefull if you intend to run simulations using ase, since .traj binary file supports lazy loading.")
    parser.add_argument("--root_dir", "-r", type=str, nargs="?", default=None,  required = False,
                        help="Root directory to search for .dump files (default: current directory)")
    
    parser.add_argument("--input_file", "-f", type=str, dafault = None, required=False, help="convert a single file")

    parser.add_argument("--check", action="store_true", help="if passed, check if the traj file exists before trying to process a dump.")
    return parser.parse_args()

def convert_single_file(args):
    """
    Worker function to convert a single .dump file to .traj.
    Args is a tuple of (dump_path, check) to support multiprocessing.
    """
    dump_path, check = args
    traj_path = os.path.splitext(dump_path)[0] + ".traj"

    if check and os.path.exists(traj_path):
        return True, f"Skipped (already exists): {traj_path}"

    try:
        # Load all frames
        atoms_objects = read(dump_path, index=":", format="lammps-dump-text")
        # Write to ASE trajectory format
        write(traj_path, atoms_objects)
        # RETURN A TUPLE: (Success_Status, Message)
        return True, f"Successfully converted {dump_path}"
    except Exception as e:
        # RETURN A TUPLE: (Success_Status, Error_Message)
        return False, f"Failed to convert {dump_path}: {e}"

def parallel_convert(root_dir, check):
    # 1. Gather all .dump file paths
    dump_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".dump"):
                dump_files.append(os.path.join(root, file))

    if not dump_files:
        print("No .dump files found.")
        return

    # 2. Process with Pool and tqdm
    num_files = len(dump_files)
    failed_files = []

    print(f"Starting conversion of {num_files} files using {cpu_count()} cores...")

    worker_args = [(path, check) for path in dump_files]

    with Pool(processes=cpu_count()) as pool:
        for success, message in tqdm(pool.imap_unordered(convert_single_file, worker_args), 
                                     total=num_files, 
                                     desc="Converting .dump to .traj",
                                     unit="file"):
            if not success:
                failed_files.append(message)
            if success:
                print(message)

    # 3. Final Summary
    print("\n--- Processing Complete ---")
    if failed_files:
        print(f"Finished with {len(failed_files)} errors:")
        for error in failed_files:
            print(f"  - {error}")
    else:
        print("All files converted successfully.")

def main():
    args = parse_args()
    parallel_convert(args.root_dir, args.check)

if __name__ == "__main__":
    main()