from argparse import ArgumentParser
from ase.io import read, write
from ase import Atoms
import os


def main():
    parser = ArgumentParser(
        description=(
            "Conform the calculator result fields of a target file to match those in a "
            "reference file. Any calc result keys present in the target but missing from "
            "the reference will be permanently deleted.\n\n"
            "WARNING: This operation is destructive. You must explicitly choose whether "
            "to create a backup of the target file using --backup or to proceed without a "
            "backup using --no-backup."
        )
    )

    parser.add_argument(
        "reference",
        type=str,
        help="Reference file that defines the allowed calculator result keys.",
    )

    parser.add_argument(
        "target", type=str, help="Target file whose calc fields will be modified."
    )

    parser.add_argument(
        "--format",
        type=str,
        help="Input/output file format (default: let ASE auto-detect).",
    )

    # Mandatory: user MUST choose backup or no-backup
    backup_group = parser.add_mutually_exclusive_group(required=True)

    backup_group.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the target file as 'backup.backup'.",
    )

    backup_group.add_argument(
        "--no-backup",
        action="store_true",
        help="Proceed without creating a backup (destructive!).",
    )

    args = parser.parse_args()

    # Sanity checks
    if not os.path.isfile(args.reference):
        print(f"[ERROR]: '{args.reference}' does not exist. Exiting.")
        exit(-1)

    if not os.path.isfile(args.target):
        print(f"[ERROR]: '{args.target}' does not exist. Exiting.")
        exit(-1)

    target_dir = os.path.dirname(args.target)

    # Load reference (template)
    template = Atoms(read(args.reference, format=args.format, index=0))
    template_keys = set(template.calc.results.keys())

    # Load target images
    target_images = read(args.target, index=":", format=args.format)

    # Backup if required
    if args.backup:
        backup_path = os.path.join(target_dir, "backup.backup")
        write(backup_path, target_images, format=args.format)
        print(f"[INFO] Backup created at: {backup_path}")
    else:
        print("[INFO] Proceeding WITHOUT backup.")

    # Remove extra keys
    for atoms in target_images:
        for key in list(atoms.calc.results.keys()):
            if key not in template_keys:
                atoms.calc.results.pop(key, None)

    # Write modified file
    write(args.target, target_images, format=args.format)
    print(f"[INFO] Target file '{args.target}' updated successfully.")

    return 0


if __name__ == "__main__":
    main()
