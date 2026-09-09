from ase.io import read, write
import numpy as np

in_file=input("Input file: ")
out_file=input("Output file: ")

# Read LAMMPS data file
atoms = read(in_file, format="lammps-data", style="atomic")

# Box height
Lz = atoms.cell[2, 2]
print("Lz", Lz)


# Get z positions
z = atoms.positions[:, 2]

# Choose a cutoff halfway through the box
z_cut = 0.5 * Lz
print("z_cut", z_cut)

# Shift wrapped layer back down
mask = z > z_cut
atoms.positions[mask, 2] -= Lz

# Optional: wrap everything back into box [0, L)
# atoms.wrap()

# Write fixed structure
write(out_file, atoms, format="lammps-data")

