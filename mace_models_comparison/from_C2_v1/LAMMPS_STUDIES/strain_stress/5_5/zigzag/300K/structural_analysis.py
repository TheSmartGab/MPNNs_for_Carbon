import ase
from ase import atoms
from ase.neighborlist import neighbor_list
import numpy as np

# some default values
CUTOFF = 1.0
BOUNDARY_THRESHOLD = 3

def get_neigh_count(atoms, cutoff):
    i = neighbor_list('i', atoms, cutoff)
    return np.bincount(i)

def count_boundary(atoms, cutoff, threshold):
    i = neighbor_list('i', atoms, cutoff)
    neigh_counts = np.bincount(i, minlength=len(atoms))
    return np.count_nonzero(neigh_counts < threshold)