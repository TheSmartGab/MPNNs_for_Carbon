import sys
import argparse
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(description="Convert a single-frame dump to LAMMPS data format.")
    parser.add_argument("input", help="Input temporary dump file")
    parser.add_argument("output", help="Output .lammps-data file")
    args = parser.parse_args()

    try:
        # format='lammps-dump-text' ensures ASE knows what it's looking at
        atoms = read(args.input, format='lammps-dump-text')
        
        # Write to lammps-data format
        write(args.output, atoms, format='lammps-data', velocities=True)
        print(f"Successfully converted to {args.output}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()