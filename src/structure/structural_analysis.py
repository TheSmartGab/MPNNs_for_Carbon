import os
import numpy as np
from ase.io import Trajectory
from ase.neighborlist import neighbor_list
from concurrent.futures import ProcessPoolExecutor
from numba import njit
from functools import partial

from ase.neighborlist import NeighborList
from tqdm import tqdm 

import multiprocessing as mp


############################################################
# manage neighbors and boundary atoms
# originally written for graphene
############################################################

# ==========================
# Constants (can be overridden)
# ==========================
CUTOFF = 2.0 # cutodd distance between neighbors. note that ase check for overlap of spheres. cutoff is therefore halved when passed to ase
BOUNDARY_THRESHOLD = 3
COUNT_THRESHOLD=10
SKIN = 0.1  # buffer for NeighborList to avoid frequent rebuilds

# ==========================
# Count boundary atoms in a single ASE Atoms object
# ==========================
def count_neighbours(atoms, cutoff=CUTOFF, skin=SKIN, nl=None) -> int:
    if nl is None:
        nl=NeighborList(cutoffs=[cutoff/2.]*len(atoms), skin=SKIN, self_interaction=False, bothways=True)
        nl.update(atoms)

    # LEGACY
    # ISSUE: summing over rows forces you to keep skin very small
    # loop over neighbours with larger sin
    #cm = nl.get_connectivity_matrix(sparse=False) # sparse=False is 3 times faster to sum over rows
    #return  np.sum(cm, axis=1)
    # END LEGACY

    # Efficient extraction of indices and offsets
    # This avoids calling get_distance in a loop.
    indices_i = []
    indices_j = []
    offsets = []

    for i in range(len(atoms)):
        neighbors, offset_vecs = nl.get_neighbors(i)
        indices_i.extend([i] * len(neighbors))
        indices_j.extend(neighbors)
        offsets.extend(offset_vecs)

    if not indices_i:
        return np.zeros(len(atoms), dtype=int)

    # Convert to numpy arrays for vectorized math
    idx_i = np.array(indices_i)
    idx_j = np.array(indices_j)
    offs = np.array(offsets)

    # Vectorized Distance Calculation (MIC handled by offsets)
    # dist = |(pos[j] + offset @ cell) - pos[i]|
    pos = atoms.positions
    cell = atoms.get_cell()
    
    # Calculate displacement vectors for all pairs at once
    diff = (pos[idx_j] + offs @ cell) - pos[idx_i]
    dist = np.linalg.norm(diff, axis=1)

    # Count only those within the true cutoff (filtering out skin)
    mask = dist < cutoff
    counts = np.bincount(idx_i[mask], minlength=len(atoms))
    
    return counts

def count_boundary(atoms, cutoff=CUTOFF, skin=SKIN, threshold=BOUNDARY_THRESHOLD, nl=None) -> int:
    neighbour_counts = count_neighbours(atoms, cutoff=cutoff, skin=skin, nl=nl)
    return np.sum(neighbour_counts < threshold)

# do not use this for a trajectory. In that case, use the implemented method which is efficient with ase neighbor matrix
# this is just for a single snapshot
def broken(atoms, cutoff=CUTOFF, skin=SKIN, threshold=BOUNDARY_THRESHOLD, counthreshold=COUNT_THRESHOLD) -> bool:
    nboundary = count_boundary(atoms, cutoff=cutoff, skin=SKIN, threshold=threshold)
    return (nboundary >= counthreshold)


# ==========================
# Count boundary atoms for a trajectory
# ==========================
def count_boundary_traj(traj, cutoff=CUTOFF, skin=SKIN,threshold=BOUNDARY_THRESHOLD, nl=None):
    if nl is None:
        nl = NeighborList(cutoffs=[cutoff/2.]*len(traj[0]), skin=skin, self_interaction=False, bothways=True)
    counts = []
    updates = 0
    for atoms in traj:
        update = nl.update(atoms)
        updates += update
        counts.append(count_boundary(atoms, cutoff, skin=skin, threshold=threshold, nl=nl))
    return counts, updates


def _worker_from_path(args):
    path, cutoff, skin, threshold = args
    traj = Trajectory(path)
    counts, updates = count_boundary_traj(traj, cutoff=cutoff, skin=skin, threshold=threshold, nl=None)
    return path, counts, updates  # <-- include path in return

def parallel_count_boundary_trajs(files, cutoff=CUTOFF, skin=SKIN, threshold=BOUNDARY_THRESHOLD, nproc=None):
    if nproc is None:
        nproc = mp.cpu_count()

    args = [(f, cutoff, skin, threshold) for f in files]

    results = {}
    ctx = mp.get_context("spawn")  # clean worker processes, no inherited file descriptors
    with ctx.Pool(processes=nproc) as pool:
        for path, counts, updates in tqdm(pool.imap_unordered(_worker_from_path, args), total=len(args)):
            results[path] = {"counts": counts, "updates": updates}

    return results

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


def find_breaking_step(boundary_counts, threshold=10):
    """
    Find the breaking step.
    check if it is broken at the end, if not return None
    find the first step where you have the final count of boundary atoms, and return that step as breaking step.
    Note that strands of atoms can form (in graphene), but I count that as a breaking, so there is no univocous number of boundary atoms. Consider threshold with care.
    Parameters:
    - boundary_counts: list of boundary atom counts at each step
    - threshold: the minimum number of boundary atoms to consider the structure as broken at the end
    """

    # process is non reversible
    if boundary_counts[-1] < threshold:
        return None

    for step in range(len(boundary_counts)):
        if boundary_counts[step] >= threshold:
            return step

    
    # final_count = boundary_counts[-1]
    # for step, count in enumerate(boundary_counts):
    #     if count >= final_count:
    #         return step

    # this should never be reached
    return None

def find_breaking_step_list_parallel(counts_list, threshold=10, ncpu = None):
    """
    Parallel processing of counts_list using multiprocessing.Pool.
    Returns a list of breaking steps in the same order as counts_list.
    """

    if ncpu==None:
        ncpu = mp.cpu_count()
    # Create a list of tuples containing (counts, threshold) for each item
    args = [(counts, threshold) for counts in counts_list]

    # Use a context manager to ensure the pool is closed properly
    with mp.Pool(processes=ncpu) as pool:
        # starmap unpacks the tuples into the function arguments
        results = pool.starmap(find_breaking_step, args)

    return results

