import os
import glob
import argparse
from multiprocessing import Pool
from ase.io import read, write, iread

def get_file_info(args):
    """Initial pass to count how many frames will be saved from a file."""
    file_path, N = args
    try:
        # We use index=':' but only get the length to avoid high memory usage
        traj_len = len(read(file_path, index=':'))
        num_frames = (traj_len - 1) // N
        return num_frames
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def convert_frames(args):
    """Actual conversion task using iterative reading for memory efficiency."""
    file_path, N, start_idx, out_dir = args
    try:
        # iread is a generator, saving memory for massive trajectories
        traj_gen = iread(file_path)
        
        current_out_idx = start_idx
        for i, atoms in enumerate(traj_gen):
            # Skip index 0 and only take every Nth frame
            if i > 0 and i % N == 0:
                output_filename = os.path.join(out_dir, f"{current_out_idx}.lammps-data")
                write(output_filename, atoms, format='lammps-data', velocities=True)
                current_out_idx += 1
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Parallelly extract frames from ASE .traj files to .lammps-data format."
    )
    parser.add_argument("-i", "--input", type=str, default=".", 
                        help="Root directory to search for .traj files (default: current)")
    parser.add_argument("-o", "--output", type=str, required=True, 
                        help="Directory where .lammps-data files will be saved")
    parser.add_argument("-n", "--interval", type=int, default=1000, 
                        help="Save every Nth frame (default: 1000)")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), 
                        help="Number of parallel processes (default: all CPUs)")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 1. Recursive search
    search_path = os.path.join(args.input, "**/*.traj")
    traj_files = sorted(glob.glob(search_path, recursive=True))
    
    if not traj_files:
        print(f"No .traj files found in {args.input}")
        return

    print(f"Found {len(traj_files)} files. Calculating global indices...")

    # 2. Count frames per file
    with Pool(args.jobs) as pool:
        counts = pool.map(get_file_info, [(f, args.interval) for f in traj_files])

    # 3. Build task list with offsets
    start_indices = []
    current_offset = 0
    for count in counts:
        start_indices.append(current_offset)
        current_offset += count

    tasks = [
        (traj_files[i], args.interval, start_indices[i], args.output) 
        for i in range(len(traj_files)) if counts[i] > 0
    ]

    # 4. Parallel Process
    print(f"Processing {current_offset} total frames across {args.jobs} workers...")
    with Pool(args.jobs) as pool:
        pool.map(convert_frames, tasks)
    
    print(f"Success. Files saved to {args.output}")

if __name__ == "__main__":
    main()