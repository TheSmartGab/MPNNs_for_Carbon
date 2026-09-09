import os
from argparse import ArgumentParser
import numpy as np
from pprint import pprint
import pandas as pd

from ase.io import read, write

SUBDIR = "all_splitted"
SUBDIR = os.path.normpath(SUBDIR)

TARGET_DIR = ""

train_dir = ""
validation_dir = ""
test_dir = ""

VALID_ENTRIES_NAME = "ValidEntries.txt"
VALID_ENTRIES_DIR = "info"
VALID_ENTRIES_PATH = ""

ensambles = ["NVE", "NVT", "NPT"]

def parse_filename(filename):
    """Extract entry_id, run_number, Force_type, dim, chemical_hill from filename, preserving leading underscores."""
    name, _ = filename.split(".")
    prefix_len = len(name) - len(name.lstrip("_"))
    underscores = "_" * prefix_len
    fields = name.split("_")
    chemical_hill = fields[-1]

    MD = False
    field = fields[-2]
    if field in ensambles:
        ensamble = field
        MD = True

    if MD:
        id_str = "_".join(fields[:-2])
    else:
        id_str = "_".join(fields[:-1])
    
    entry_id = underscores + id_str
    return entry_id, ensamble, chemical_hill

def create_dirs(args):
    global train_dir, validation_dir, test_dir, TARGET_DIR, VALID_ENTRIES_PATH

    if len(args.root_dir.split("/")) <2:
        print("[ERROR]: something is wrong with how you are trying to store data. Individual xyz data should be put in SYSTEM/ORIGIN/DATABASE/files/.") 
        exit(-1)

    l_prev_dir = args.root_dir.split("/")[:-1] # adjust this if the subdirectory is being created somwhere wrong
    s_prev_dir = "/".join(l_prev_dir)
    TARGET_DIR = os.path.join(s_prev_dir, SUBDIR)
    VALID_ENTRIES_PATH = os.path.join(s_prev_dir, VALID_ENTRIES_DIR, VALID_ENTRIES_NAME)
    print(f"[INFO] Creating target dir {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=False)

    train_dir = os.path.join(TARGET_DIR, f"split/training")
    validation_dir = os.path.join(TARGET_DIR, f"split/validation")
    test_dir = os.path.join(TARGET_DIR, f"split/test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return (
        os.path.join(train_dir, "training.xyz"),
        os.path.join(validation_dir, "validation.xyz"),
        os.path.join(test_dir, "test.xyz"),
    )

def append_file(in_file: str, a_file: str):
    with open(in_file, "r") as src, open(a_file, "a") as dst:
        dst.write(src.read())

def append_frames(in_file: str, a_file: str, n_frame: int = None, frame_freq: int = 1):
    """
    Append frames from in_file to a_file, using every `frame_freq`-th frame,
    and up to a maximum of n_frame frames if provided.
    """
    print(f"[INFO] Appending frames from {in_file} to {a_file} with frame_freq={frame_freq} and n_frame={n_frame}")
    with open(in_file, "r") as src:
        lines = src.readlines()

    n_atoms = int(lines[0].strip())
    frame_size = n_atoms + 2  # n_atoms + comment + atoms lines
    total_frames = len(lines) // frame_size

    # Determine which frames to include
    selected_frames = list(range(0, total_frames, frame_freq))
    if n_frame is not None:
        selected_frames = selected_frames[:n_frame]  # cap the number of frames

    with open(a_file, "a") as dst:
        for frame_idx in selected_frames:
            start_line = frame_idx * frame_size
            end_line = start_line + frame_size
            dst.writelines(lines[start_line:end_line])

def filter_files_by_required_fields(file_names, valid_entries_info, args):
    filtered_file_names = []

    if args.required_fields:
        print(f"[INFO] Filtering files to only keep those with required fields: {args.required_fields}")
        for fn in file_names:
            id = parse_filename(fn)[0]
            if id in valid_entries_info['ID'].values:

                entry = valid_entries_info[valid_entries_info['ID'] == id].iloc[0]
                has_all_fields = all(entry[field] for field in args.required_fields if field in entry)
                if has_all_fields:
                    if args.required_values:
                        has_required_values = all(entry[field] in args.required_values for field in args.required_fields if field in entry)
                        if has_required_values:
                            print("[INFO] Keeping", id)
                            filtered_file_names.append(fn)
                        else:
                            print(f"[INFO] Skipping {id} as it does not have required values {args.required_values} for fields {args.required_fields}")
                    else:
                        print("[INFO] Keeping", id)
                        filtered_file_names.append(fn)

            else:
                print(f"[INFO] Skipping {id} as it is not listed in {VALID_ENTRIES_NAME}")

        return filtered_file_names

    else:
        print("[INFO] No required fields specified, keeping all files.")
        return file_names

def main():
    parser = ArgumentParser(
        prog="naive_split",
        description=(
            "Naive split: randomly divide data into training, validation, test given fractions. "
            "Can be done for structures and frames. BEWARE: splitting by frames may overestimate metrics."
        ),
    )

    parser.add_argument("root_dir", help="Root dir with .extxyz files", type=str)
    parser.add_argument("-f", "--frames", action="store_true", help="Split by frames", default=False)
    parser.add_argument("-tr", "--training", help="Training fraction", type=float, default=0.7)
    parser.add_argument("-val", "--validation", help="Validation fraction", type=float, default=0.2)
    parser.add_argument("-test", "--test", help="Test fraction", type=float, default=0.1)
    parser.add_argument("-ext", "--extension", default=".xyz", help="Data file extension", type=str)
    parser.add_argument("-s", "--seed", default=42, help="Random seed", type=int)
    parser.add_argument("-nf", "--num_frames", default=None, help="Number of frames to use per file", type=int)
    parser.add_argument("-ff", "--frame_freq", default = 1, help="Frequency of frames to use per file", type=int)
    parser.add_argument("-reqf", "--required_fields", nargs = "*", default=[], help="List of required fields in the .extxyz files", type=str)
    parser.add_argument("-reqv", "--required_values", nargs = "*", default=[], help="List of required values of required fields in the .extxyz files", type=str)
    parser.add_argument("--subdir", required = True, help = "subdirectory of the split to create.", type = str)
    args = parser.parse_args()
    args.rootdir = os.path.normpath(args.root_dir)

    global SUBDIR
    SUBDIR = args.subdir
    SUBDIR = os.path.normpath(SUBDIR)

    pprint(args)

    total_frac = args.training + args.validation + args.test
    if not abs(total_frac - 1.0) < 1e-6:
        # raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")
        print("[WARNING] Fractions do not sum to 1.0")

    train_path, validation_path, test_path = create_dirs(args)

    try:
        valid_entries_info = pd.read_csv(VALID_ENTRIES_PATH, sep = ",?\s+", engine = "python")
        print(valid_entries_info.head())
        info=True
    except FileNotFoundError as e:
        print("[WARNING] No info file found")
        info=False

    file_names = [fn for fn in os.listdir(args.root_dir) if fn.endswith(args.extension)]

    if info:
        file_names = filter_files_by_required_fields(file_names, valid_entries_info, args)
    else:
        file_names = filter_files_by_required_fields(file_names, None, args)

    n_files = len(file_names)
    print("[INFO] Number of files after filtering:", n_files)
    n_working_files = int(round(n_files * total_frac))

    if not args.frames:
        n_train = int(round(n_files * args.training))
        n_val = int(round(n_files * args.validation))
        n_test = n_working_files - n_train - n_val

        print(f"[INFO] Found {n_files} data files, splitting into train: {n_train}, val: {n_val}, test: {n_test}")

        np.random.shuffle(file_names)

        train_files = file_names[:n_train]
        validation_files = file_names[n_train:n_train + n_val]
        test_files = file_names[n_train + n_val:]  # remainder

        # Write info files
        with open(os.path.join(train_dir, "training_info.txt"), "w") as f:
            f.write(f"training split: {args.training}\n")
            f.write(f"n entries: {n_train}\n\nENTRIES\n")
            f.writelines(f"{filename}\n" for filename in train_files)

        with open(os.path.join(validation_dir, "validation_info.txt"), "w") as f:
            f.write(f"validation split: {args.validation}\n")
            f.write(f"n entries: {n_val}\n\nENTRIES\n")
            f.writelines(f"{filename}\n" for filename in validation_files)

        with open(os.path.join(test_dir, "test_info.txt"), "w") as f:
            f.write(f"test split: {args.test}\n")
            f.write(f"n entries: {n_test}\n\nENTRIES\n")
            f.writelines(f"{filename}\n" for filename in test_files)

        # Append data into combined .xyz files
        for f in train_files:
            append_frames(os.path.join(args.rootdir, f), train_path, n_frame=args.num_frames, frame_freq=args.frame_freq)
        for f in validation_files:
            append_frames(os.path.join(args.rootdir, f), validation_path, n_frame=args.num_frames, frame_freq=args.frame_freq)
        for f in test_files:
            append_frames(os.path.join(args.rootdir, f), test_path, n_frame=args.num_frames, frame_freq=args.frame_freq)

    else:
        print("[INFO] Splitting by frames using ASE...")
        # Collect all frames
        all_frames = []
        for fn in file_names:
            full_path = os.path.join(args.rootdir, fn)
            try:
                frames = read(full_path, index=":")
                all_frames.extend(frames)
            except Exception as e:
                print(f"[WARNING] Could not read {fn} with ASE: {e}")

        n_total = len(all_frames)
        print(f"[INFO] Loaded {n_total} total frames from {len(file_names)} files")

        if n_total == 0:
            print("[ERROR] No frames found – aborting.")
            return

        # Shuffle frames
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_frames)

        # Compute split sizes
        n_train = int(round(n_total * args.training))
        n_val = int(round(n_total * args.validation))
        n_test = n_total - n_train - n_val

        print(f"[INFO] Splitting frames into train={n_train}, val={n_val}, test={n_test}")

        train_frames = all_frames[:n_train]
        val_frames = all_frames[n_train:n_train+n_val]
        test_frames = all_frames[n_train+n_val:]

        # Info files with counts only (no filenames since not per-file)
        with open(os.path.join(train_dir, "training_info.txt"), "w") as f:
            f.write(f"training split: {args.training}\n")
            f.write(f"n frames: {n_train}\n")

        with open(os.path.join(validation_dir, "validation_info.txt"), "w") as f:
            f.write(f"validation split: {args.validation}\n")
            f.write(f"n frames: {n_val}\n")

        with open(os.path.join(test_dir, "test_info.txt"), "w") as f:
            f.write(f"test split: {args.test}\n")
            f.write(f"n frames: {n_test}\n")

        # Write combined xyz files
        print("[INFO] Writing output datasets...")

        if train_frames:
            write(train_path, train_frames)
        if val_frames:
            write(validation_path, val_frames)
        if test_frames:
            write(test_path, test_frames)

        print("[INFO] Frame-based split complete.")


if __name__ == "__main__":
    main()
