from ase.build import graphene, make_supercell
from ase.io import write
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser("Generate a rectangular cell of graphene.")
    parser.add_argument("--supercell", nargs=2, type=int, required=True, help="Number of repetitions along x and y axis for rectangular graphene. Note that the primitive rectangular cell is elongated along the y axis.")

    args = parser.parse_args()
    return args

def generate_graphene(supercell):
    # Primitive graphene
    atoms = graphene(vacuum=50)
    atoms.pbc = [True, True, False]

    # Integer transformation matrix → rectangular cell
    # This converts hexagonal → orthogonal
    nx, ny = supercell
    P = np.array([
        [nx,  nx, 0],
        [-ny, ny, 0],
        [0,  0, 1]
    ])

    atoms_rect = make_supercell(atoms, P)
    atoms_rect.wrap()

    #atoms_rect.repeat(supercell)

    write(f"initial_graphene_{supercell[0]}_{supercell[1]}.lammps-data", atoms_rect)

if __name__ == "__main__":
    args = parse_args()
    generate_graphene(args.supercell)
