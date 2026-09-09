#!/usr/bin/env python3
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spglib
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from ase import Atoms
from ase.build import graphene
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

# ---------------------------------------------------------------------------
# Framework Data Transformers
# ---------------------------------------------------------------------------

def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell()[:],
        scaled_positions=atoms.get_scaled_positions(),
    )


def phonopy_to_ase(ph_atoms: PhonopyAtoms, calc=None) -> Atoms:
    from ase import Atoms as AseAtoms
    a = AseAtoms(
        symbols=ph_atoms.symbols,
        cell=ph_atoms.cell,
        scaled_positions=ph_atoms.scaled_positions,
        pbc=True,
    )
    if calc is not None:
        a.calc = calc
    return a


def compute_force_constants(
    primitive: Atoms,
    calc: Calculator,
    supercell_matrix,
    symprec: float = 1e-3,
    displacement: float = 0.01,
) -> Phonopy:
    ph_primitive = ase_to_phonopy(primitive)
    ph = Phonopy(
        ph_primitive,
        supercell_matrix=supercell_matrix,
        primitive_matrix='P',
        symprec=symprec,          
        is_symmetry=True,
    )

    ph.generate_displacements(distance=displacement)
    supercells_with_displacements = ph.supercells_with_displacements

    print(f"[phonopy] Evaluating {len(supercells_with_displacements)} displaced structural configurations...")
    forces = []
    for i, sc in enumerate(supercells_with_displacements):
        ase_sc = phonopy_to_ase(sc, calc=None)
        if hasattr(calc, 'reset'):
            calc.reset()
        ase_sc.calc = calc          
        f = ase_sc.get_forces()
        forces.append(f)

    ph.forces = forces
    ph.produce_force_constants()
    return ph


# ---------------------------------------------------------------------------
# Dynamic Irrep Extractor (No Hardcoding)
# ---------------------------------------------------------------------------

def get_symmetry_inferred_irreps(ph: Phonopy, qpoint) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts the pure A1 and B1 Irrep basis functions by computing the complex
    phase-shifted normal mode coordinates across the supercell.
    """
    ph.run_qpoints([qpoint], with_eigenvectors=True)
    mesh_dict = ph.get_qpoints_dict()

    freqs = mesh_dict['frequencies'][0]   
    eigvecs = mesh_dict['eigenvectors'][0]  

    idx_unstable = np.argmin(freqs)
    freq_unstable = freqs[idx_unstable]

    print(f"[phonopy] Target unstable K1 mode detected at: {freq_unstable:.4f} THz")

    ev_prim = eigvecs[:, idx_unstable]

    sc = ph.supercell
    prim = ph.primitive
    N_sc = len(sc.masses)

    sc_pos    = sc.positions
    prim_pos  = prim.positions
    prim_cell = prim.cell

    inv_prim = np.linalg.inv(prim_cell.T)
    q_frac   = np.array(qpoint, dtype=float)

    disp_A1 = np.zeros((N_sc, 3), dtype=float)
    disp_B1 = np.zeros((N_sc, 3), dtype=float)

    sc_to_prim_map = prim.s2p_map  
    p2s_list       = list(prim.p2s_map)  

    for i_sc in range(N_sc):
        prim_sc_index = sc_to_prim_map[i_sc]
        i_prim = p2s_list.index(prim_sc_index)

        # Calculate lattice translation vector R
        R_cart = sc_pos[i_sc] - prim_pos[i_prim]
        R_frac = R_cart @ inv_prim.T   

        # Construct Bloch phase mapping: e^{i 2pi q . R}
        phase = np.exp(2j * np.pi * np.dot(q_frac, R_frac))
        ev_block = ev_prim[3*i_prim : 3*i_prim+3]

        # Standard complex mass-weighted projection
        complex_disp = (ev_block * phase) / np.sqrt(sc.masses[i_sc])
        
        # Real and Imaginary orthogonal components directly span the A1 and B1 spaces
        # of the warped potential map at the zone boundary
        disp_A1[i_sc] = np.real(complex_disp)
        disp_B1[i_sc] = np.imag(complex_disp)

    # Normalize vectors individually by their maximum atomic displacement values
    norm_A1 = np.max(np.sqrt(np.sum(disp_A1**2, axis=1)))
    if norm_A1 > 1e-12:
        disp_A1 /= norm_A1
        
    norm_B1 = np.max(np.sqrt(np.sum(disp_B1**2, axis=1)))
    if norm_B1 > 1e-12:
        disp_B1 /= norm_B1

    return disp_A1, disp_B1, freq_unstable


# ---------------------------------------------------------------------------
# Energy Scanner Function
# ---------------------------------------------------------------------------

def energy_scan_along_vector(
    sc_ase: Atoms,
    calc: Calculator,
    displacement_vector: np.ndarray,
    amplitudes: np.ndarray
) -> np.ndarray:
    pos0 = sc_ase.get_positions().copy()
    energies = np.zeros(len(amplitudes))

    for i, u in enumerate(amplitudes):
        sc_ase.set_positions(pos0 + u * displacement_vector)
        if hasattr(calc, 'reset'):
            calc.reset()
        sc_ase.calc = calc
        energies[i] = sc_ase.get_potential_energy()

    sc_ase.set_positions(pos0)   
    return energies


# ---------------------------------------------------------------------------
# Core Execution Coordinator
# ---------------------------------------------------------------------------

def scan_unstable_mode(
    atoms: Atoms,
    calc: Calculator,
    qpoint,
    amplitudes: np.ndarray,
    supercell_matrix,
    symprec: float = 1e-3,
    displacement: float = 0.01,
):
    atoms = atoms.copy()
    atoms.pbc = True

    # 1. Evaluate Force Matrices
    ph = compute_force_constants(atoms, calc, supercell_matrix, symprec=symprec, displacement=displacement)
    disp_A1, disp_B1, freq_thz = get_symmetry_inferred_irreps(ph, qpoint)
    sc_ase = phonopy_to_ase(ph.supercell)

    # 2. Trace Explicit Irrep Energy Profiles
    print("\n[Scan Engine] Scanning along symmetry-inferred A1 and B1 coordinate vectors...")
    energies_A1 = energy_scan_along_vector(sc_ase, calc, disp_A1, amplitudes)
    energies_B1 = energy_scan_along_vector(sc_ase, calc, disp_B1, amplitudes)

    zero_idx = np.abs(amplitudes).argmin()
    E0_A1 = energies_A1[zero_idx]
    E0_B1 = energies_B1[zero_idx]

    dE_A1 = (energies_A1 - E0_A1) * 1000  # convert to meV
    dE_B1 = (energies_B1 - E0_B1) * 1000  # convert to meV

    print(f"  -> Symmetry Trajectory A1 | Min dE = {dE_A1.min():+.2f} meV | Max dE = {dE_A1.max():+.2f} meV")
    print(f"  -> Symmetry Trajectory B1 | Min dE = {dE_B1.min():+.2f} meV | Max dE = {dE_B1.max():+.2f} meV")

    # 3. Export 3-Column Scan Metrics Data File
    df = pd.DataFrame({
        "Amplitude_Angstrom": amplitudes,
        "Delta_E_A1_meV": dE_A1,
        "Delta_E_B1_meV": dE_B1
    })
    csv_filename = "unstable_irreps_scan.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n[Data Export] Symmetrized coordinate curves saved to: {csv_filename}")

    # 4. Generate Separate Plot Representation
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    ax.plot(amplitudes, dE_A1, '-', color='#E07030', lw=2.2, label=r'$A_1$ Representation Profile')
    ax.plot(amplitudes, dE_B1, '-', color='#3A7DC9', lw=2.2, label=r'$B_1$ Representation Profile')

    ax.axhline(0, color='black', lw=0.6, ls='--')
    ax.axvline(0, color='black', lw=0.6, ls='--')
    
    ax.set_xlabel('Symmetry-Coordinate Amplitude $u$ (Å)', fontsize=11)
    ax.set_ylabel(r'$\Delta E$ (meV / supercell)', fontsize=11)
    ax.set_title(f'Symmetry-Inferred Irrep Landscape ($q$ = {qpoint})\nInstability Mode Frequency = {freq_thz:.3f} THz')
    
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plot_filename = "unstable_irreps_projection.svg"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"[Plotting] High-fidelity isolated projections saved to: {plot_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--angle', type=float, default=45.0)
    parser.add_argument('--rot', type=float, default=0.0)
    parser.add_argument('--amplitude', type=float, default=0.25)
    parser.add_argument('--min_amp', type=float, default=-0.5)
    parser.add_argument('--max_amp', type=float, default=0.5)
    parser.add_argument('--points', type=int, default=100)
    parser.add_argument('--symprec', type=float, default=1e-3)
    args = parser.parse_args()

    from mace.calculators import MACECalculator

    model = MACECalculator(model_path=args.model_path, device=args.device)

    angle_rad = np.deg2rad(args.angle)
    C, S = np.cos(angle_rad), np.sin(angle_rad)
    
    STRAIN = args.amplitude * np.array([[C, 0, 0], [0, S, 0], [0, 0, 0]])
    DEFORMATION = np.eye(3) + STRAIN
    ROTATION = np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])

    supercell_matrix = [[2, 1, 0], [-1, 1, 0], [0, 0, 1]]
    qpoint = np.array([1/3, 1/3, 0])
    amps = np.linspace(args.min_amp, args.max_amp, args.points)

    print("[Pipeline] Initializing clean graphene configuration...")
    atoms = graphene()
    atoms.set_cell(atoms.cell @ ROTATION.T, scale_atoms=False)
    atoms.set_positions(atoms.positions @ ROTATION.T)
    
    atoms.cell[2, 0] = 0.0; atoms.cell[2, 1] = 0.0
    atoms.cell[0, 2] = 0.0; atoms.cell[1, 2] = 0.0
    atoms.cell[2, 2] = 30.0  

    atoms.calc = model
    cell_filter = FrechetCellFilter(atoms, [1, 1, 0, 0, 0, 0])
    opt = BFGS(cell_filter)
    opt.run(fmax=1e-5, steps=1000)

    print("[Pipeline] Applying 2D Strain transformations to cell boundaries...")
    cell_matrix = atoms.get_cell().copy()
    cell_matrix[0:2, 0:2] = cell_matrix[0:2, 0:2] @ DEFORMATION[0:2, 0:2].T
    atoms.set_cell(cell_matrix, scale_atoms=True)
    
    atoms.cell[2, 0] = 0.0; atoms.cell[2, 1] = 0.0
    atoms.cell[0, 2] = 0.0; atoms.cell[1, 2] = 0.0
    atoms.cell[2, 2] = 30.0
    atoms.center(axis=2)
    atoms.pbc = [True, True, True]

    scan_unstable_mode(
        atoms=atoms,
        calc=model,
        qpoint=qpoint,
        amplitudes=amps,
        supercell_matrix=supercell_matrix,
        symprec=args.symprec,
    )