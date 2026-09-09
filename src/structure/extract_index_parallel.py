from argparse import ArgumentParser
import os
from ase.io import read, write
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from structural_analysis import broken


def parse_args():
    parser = ArgumentParser(
        "Extract a given index from multiple files and save as LAMMPS trajectory format"
    )

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=1000)
    parser.add_argument("--ext", type=str, default=".traj")
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())

    return parser.parse_args()


def process_file(f, root_dir, target_dir, index):
    input_path = os.path.join(root_dir, f)

    try:
        traj = read(input_path, index=":")

        if index >= len(traj):
            return f"[ERROR] Skipping {f}: index {index} out of range"

        atoms = traj[index]

        if broken(atoms):
            return f"[ERROR] {input_path} broke before index {index}"

        base_name = os.path.splitext(f)[0]
        output_file = base_name + ".lammps-data"
        output_path = os.path.join(target_dir, output_file)

        write(output_path, atoms, format="lammps-data", velocities=True)

        return f"Written: {output_path}"

    except Exception as e:
        return f"Error processing {f}: {e}"


def main():
    args = parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    files = os.listdir(args.root_dir)
    files = [f for f in files if f.endswith(args.ext)]

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [
            executor.submit(
                process_file, f, args.root_dir, args.target_dir, args.index
            )
            for f in files
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())


if __name__ == "__main__":
    main()