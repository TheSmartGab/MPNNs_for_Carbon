import os
import numpy as np
from ase.io import Trajectory
from ase.neighborlist import neighbor_list
from concurrent.futures import ProcessPoolExecutor
from numba import njit
from functools import partial

from ase.neighborlist import NeighborList
from tqdm import tqdm 


############################################################
# manage neighbors and boundary atoms
# originally written for graphene
############################################################

# ==========================
# Constants (can be overridden)
# ==========================
CUTOFF = 2.0 # cutodd distance between neighbors. note that ase check for overlap of spheres. cutoff is therefore halved when passed to ase
BOUNDARY_THRESHOLD = 3
SKIN = 0.2  # buffer for NeighborList to avoid frequent rebuilds

# ==========================
# Count boundary atoms in a single ASE Atoms object
# ==========================
def count_neighbours(atoms, cutoff=CUTOFF, skin=SKIN, nl=None):
    if nl is None:
        nl=NeighborList(cutoffs=[cutoff/2.]*len(atoms), skin=SKIN, self_interaction=False, bothways=True)
        nl.update(atoms)

    # if nl is passed, assume it is updated!
    # if you have to call many functions that requires update, this is more efficient
    cm = nl.get_connectivity_matrix(sparse=False) # sparse=False is 3 times faster to sum over rows
    return  np.sum(cm, axis=1)

def count_boundary(atoms, cutoff=CUTOFF, skin=SKIN, threshold=BOUNDARY_THRESHOLD, nl=None):
    neighbour_counts = count_neighbours(atoms, nl=nl)
    return np.sum(neighbour_counts < threshold)

# ==========================
# Count boundary atoms for a trajectory
# ==========================
def count_boundary_traj(traj, cutoff=CUTOFF, skin=SKIN,threshold=BOUNDARY_THRESHOLD):
    nl = NeighborList(cutoffs=[cutoff/2.]*len(traj[0]), skin=skin, self_interaction=False, bothways=True)
    counts = []
    updates = 0
    for atoms in traj:
        update = nl.update(atoms)
        updates += update
        counts.append(count_boundary(atoms, cutoff, skin=skin, threshold=threshold, nl=nl))
    return counts, updates

# ==========================
# Count boundary atoms for a single file
# ==========================
def count_boundary_file(filename, cutoff=CUTOFF, skin=SKIN, threshold=BOUNDARY_THRESHOLD):
    print(f"Processing file: {filename}")
    traj = Trajectory(filename)
    return count_boundary_traj(traj, cutoff, skin, threshold)

# ==========================
# Count boundary atoms for all files in a directory
# ==========================
def count_boundary_dir(directory, cutoff=CUTOFF, skin=SKIN, threshold=BOUNDARY_THRESHOLD, max_workers=16):

    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".traj")]

    if not filepaths:
        print(f"No .traj files found in directory {directory}")
        return {}

    worker_func = partial(count_boundary_file, cutoff=cutoff, skin=skin, threshold=threshold)

    counts_dict = {}
    # Limit max_workers to number of files
    with ProcessPoolExecutor(max_workers=min(max_workers, len(filepaths))) as executor:
        # Wrap map with tqdm for progress bar
        for f, result in zip(filepaths, tqdm(executor.map(worker_func, filepaths), total=len(filepaths), desc="Processing")):
            counts_dict[os.path.basename(f)] = {"counts": result[0], "updates": result[1]}

    return counts_dict


############################################################
# find graphene breaking
############################################################


def find_breaking_step(boundary_counts, threshold=15):
    """
    Find the breaking step.
    check if it is broken at the end, if not return None
    find the first step where you have the final count of boundary atoms, and return that step as breaking step.
    Note that strands of atoms can form (in graphene), but I count that as a breaking, so there is no univocous number of boundary atoms. Consider threshold with care.
    Parameters:
    - boundary_counts: list of boundary atom counts at each step
    - threshold: the minimum number of boundary atoms to consider the structure as broken at the end
    """

    if boundary_counts[-1] < threshold:
        return None

    final_count = boundary_counts[-1]
    for step, count in enumerate(boundary_counts):
        if count >= final_count:
            return step

    # this should never be reached
    return None