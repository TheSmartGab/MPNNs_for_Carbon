from argparse import ArgumentParser
from ase.io import read, write
from gpaw import GPAW, PW


def main():
    parser = ArgumentParser(
        """Compute GPAW single-point energies and forces for configurations."""
    )

    parser.add_argument("input_file", type=str,
                        help="Input ASE-readable structure file.")
    parser.add_argument("--ecut", type=float, required=True)
    parser.add_argument("--kpoints", nargs=3, type=int, required=True)
    parser.add_argument("--output_name", default="gpaw_run_single_point")
    parser.add_argument("--format", default="extxyz")

    args = parser.parse_args()

    # Load possibly multiple structurea (index=":")
    atoms_list = read(args.input_file, index=":", format=args.format)

    for i, atoms in enumerate(atoms_list):
        calc = GPAW(mode=PW(args.ecut),
                    kpts=args.kpoints,
                    xc='LDA',
                    txt=f"{args.output_name}_{i}.out")

        atoms.calc = calc

        # Trigger actual GPAW calculation
        properties = atoms.calc.implemented_properties
        atoms.calc.calculate(atoms, properties=properties)

        
    # Write out updated energies/forces computed by GPAW
    write(args.output_name + ".extxyz", atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
