import numpy as np
from ase.build import make_supercell
from ase.optimize import BFGS
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from fractions import Fraction
import itertools

from ase.calculators import calculator
from ase.phonons import Phonons

import numpy as np
from ase import Atoms
from ase.phonons import Phonons

import numpy as np
from ase import Atoms
import ase.units as units
from ase.io import write

def energy_mode(
    pristine: Atoms,
    versor: np.ndarray,
    amplitude: float,
    supercell: tuple[int, int, int],  # e.g., (3, 3, 1)
    calc,
    kpoints: np.ndarray,              # Shape: (M, 3)
    modes: np.ndarray,                # Shape: (M, N_atoms_unit, 3)
    coefficients: np.ndarray,         # Shape: (M,)
    mode_amplitudes: np.ndarray,
    phase_shift: float = 0.0,          # The 'x' parameter from ASE's linspace(0, 2*pi)
    phases_matrix: np.ndarray = np.eye(3),
    debug: bool = False
):
    assert kpoints.shape[0] == modes.shape[0],        "[ERROR] kpoints and mode mismatch"
    assert kpoints.shape[0] == coefficients.shape[0], "[ERROR] kpoints and coefficients mismatch"
    assert modes.shape[1] == len(pristine),            "[ERROR] modes - natoms mismatch"

    # 1. Geometry Setup
    deformation = np.eye(3) + amplitude * versor
    strained = pristine.copy()
    strained.set_cell(pristine.cell @ deformation, scale_atoms=True)
    

    strained.calc = calc
    # opt = BFGS(strained)
    # opt.run(fmax = 1e-5, steps=1000)
    # write("deformed_energy_mode.extxyz", strained)

    if debug:
        print("="*70)
        print("strained positions")
        print(strained.positions)

        print("="*70)
        print(("fractional positions"))
        print(strained.get_scaled_positions())

    # ASE: atoms = self.atoms * repeat
    atoms = strained * supercell
    pos_Nav = atoms.get_positions()
    N = np.prod(supercell)

    # 2. Phase Grid Allocation
    # ASE: R_cN = np.indices(repeat).reshape(3, -1)
    R_cN = np.indices(supercell).reshape(3, -1)
    if debug:
        print("="*70)
        print("R_cN")
        print(R_cN)

    # Pre-allocate total real displacements matching pos_Nav shape
    total_real_displacements = np.zeros_like(pos_Nav, dtype=complex)

    # 3. Build & Linearly Combine Modes using ASE Architecture
    for q_c, u_av, coeff in zip(kpoints, modes, coefficients):
        # ASE: phase_N = np.exp(2.0j * pi * np.dot(q_c, R_cN))
        phase_N = np.exp(2.0j * np.pi * np.dot(q_c, R_cN))
        # ASE: phase_Na = phase_N.repeat(len(self.atoms))
        phase_Na = phase_N.repeat(len(strained))
        if debug:   
            print("="*70)
            print("phase_Na")
            print(phase_Na)

        # We construct mode_av exactly like ASE's internal assignments
        mode_av = np.zeros((len(strained), 3), dtype=complex)
        mode_av[:] = u_av  # Assumes all indices are included

        # ASE: mode_Nav = np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis]
        mode_Nav = np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis]

        # Incorporate the coefficient and the dynamic trajectory phase shift e^(1j * x)
        # ASE uses: (np.exp(1.0j * x) * mode_Nav).real
        modulated_mode = coeff * np.exp(1.0j * phase_shift) * mode_Nav
        
        # Accumulate the real part of this specific mode contribution
        total_real_displacements += modulated_mode

    # phases are just to explore, but they should not be used in production as they do not commute with dynamical matrix
    total_real_displacements @= phases_matrix
    total_real_displacements = np.array(total_real_displacements.real, dtype=np.float64)
    if debug:
        print("="*70)
        print("total_real_displacements")
        print(total_real_displacements)

    # 4. Map Energy Profiles Across Frozen Amplitudes
    modded_configs = []
    energies = []
    force_projections = []

    for mode_amp in mode_amplitudes:
        modded = atoms.copy()
        
        # Scale the accumulated real structural displacements by the global amplitude
        modded.set_positions(pos_Nav + mode_amp * total_real_displacements)
        
        modded.calc = calc
        energies.append(modded.get_potential_energy())
        fp = modded.get_forces() * total_real_displacements
        force_projections.append(fp)
        modded_configs.append(modded)

    return modded_configs, energies, force_projections

