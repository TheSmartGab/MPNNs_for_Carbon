from ase.lattice.hexagonal import Graphite
from ase import Atoms
import ase.io as io
import numpy as np

def atoms_in_circle(atoms, center, radius):
    """Select atoms within a circle around center (no PBC)."""
    selected = []
    for atom in atoms:
        if np.linalg.norm(atom.position[:2] - center[:2]) <= radius:
            selected.append(atom)
    return selected

def find_top_pair_near_center(atoms, max_xy_tol=0.01):
    """
    Efficiently find a pair of atoms in different layers on top of each other
    near the XY center of the system.
    
    max_xy_tol: maximum XY distance to accept a pair (Å)
    """
    positions = np.array([atom.position for atom in atoms])
    xy_com = positions[:, :2].mean(axis=0)
    
    # Compute XY distances from COM
    dist_xy = np.linalg.norm(positions[:, :2] - xy_com, axis=1)
    
    # Sort atom indices by distance from COM (closest first)
    sorted_idx = np.argsort(dist_xy)
    
    # Loop over atoms from center outwards
    for i in sorted_idx:
        pos1 = positions[i]
        for j in sorted_idx:
            if i >= j:
                continue
            pos2 = positions[j]
            # Different layers
            if abs(pos1[2] - pos2[2]) > 0.1:
                # XY distance between atoms
                dxdy = np.linalg.norm(pos1[:2] - pos2[:2])
                if dxdy < max_xy_tol:
                    # Return the XY midpoint
                    return (pos1[:2] + pos2[:2]) / 2

    raise ValueError("No top-pair found within tolerance")

# ---------------- Parameters ----------------
index1 = 100
index2 = 100
mya = 2.45942
myc = 3.54505*2

stacks = 1
radius = 50      # circular flake radius
vacuum = 50      # vacuum in all directions

# Build graphite
gra = Graphite(symbol='C', latticeconstant={'a': mya, 'c': myc},size=(index1,index2,stacks))

# Find a top-pair near XY center
center_xy = find_top_pair_near_center(gra)
print("Top-pair chosen at XY:", center_xy)

# Shift entire graphite so top-pair is at origin
for atom in gra:
    atom.position[:2] -= center_xy

# Select atoms in a circular flake
in_circle_atoms = atoms_in_circle(gra, np.array([0,0,0]), radius)
flake = Atoms(in_circle_atoms)

# Determine Z extents
positions = np.array([atom.position for atom in flake])
z_min, z_max = positions[:,2].min(), positions[:,2].max()
cell_z = (z_max - z_min) + 2*vacuum

# Determine XY extents for cell
x_min, x_max = positions[:,0].min(), positions[:,0].max()
y_min, y_max = positions[:,1].min(), positions[:,1].max()
cell_x = (x_max - x_min) + 2*vacuum
cell_y = (y_max - y_min) + 2*vacuum

# Set rectangular cell
flake.cell = np.array([[cell_x, 0, 0],
                       [0, cell_y, 0],
                       [0, 0, cell_z]])

# Shift atoms in Z to center flake
# for atom in flake:
#     atom.position[2] = atom.position[2] - (z_min+z_max)/2 + cell_z/2

# XY stays unchanged, so top-pair remains at (0,0)

# Write LAMMPS file
io.write(f'circular_graphene_r{radius:.2f}.lammps-data', flake, format='lammps-data')

print("Connected circular flake created, top-pair at XY=0, centered in Z with vacuum.")
