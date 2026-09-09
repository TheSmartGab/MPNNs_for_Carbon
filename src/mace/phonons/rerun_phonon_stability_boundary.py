from __future__ import annotations

import mace
import ase

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from tqdm import tqdm

from ase.phonons import Phonons

from ase import Atoms
from ase.build import graphene

from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

from mace.calculators import MACECalculator

import pandas as pd

# Logging setup

def setup_logging(outdir: Path, verbose: bool = False) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phonon_strain_map")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler - always DEBUG
    fh = logging.FileHandler(outdir / "run.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

# Graphene unit cel

def make_graphene_unitcell(model):
    """Return a pristine graphene unit cell as an ASE Atoms object."""

    # Use ase.build.graphene (a=C-C bond * sqrt(3))
    # Vacuum of 100 Å along Z to suppress interlayer interactions.
    try:
        atoms = graphene(vacuum=100)
    except Exception:
        # Fallback: build manually
        a = 2.46  # Å - experimental graphene lattice constant
        atoms = Atoms(
            symbols=["C", "C"],
            scaled_positions=[[0.0, 0.0, 0.0], [1 / 3, 2 / 3, 0.0]],
            cell=[
                [a, 0.0, 0.0],
                [a / 2, a * np.sqrt(3) / 2, 0.0],
                [0.0, 0.0, 100.0],
            ],
            pbc=[True, True, False],
        )

    atoms.calc = model

    filter = FrechetCellFilter(atoms, mask=[1,1,0,0,0,0])
    opt = BFGS(filter, logfile=None)
    converged = opt.run(fmax = 1e-5, steps=1000)

    return atoms, converged

# Strain applicatio

def biaxial_strain_tensor(angle: float, amplitude: float) -> np.ndarray:
    """
    Construct a 3dx3 deformation gradient F for biaxial strain.

    Parameters
    -
    angle     : Strain anisotropy angle in [0, π/2].
                0   → pure uniaxial along x
                π/4 → isotropic biaxial
                π/2 → pure uniaxial along y
    amplitude : Magnitude of the strain (engineering strain).

    Returns
    --
    F : (3,3) deformation gradient (I + ε)
    """
    # Strain eigenvalues along principal axes
    e_x = amplitude * np.cos(angle)
    e_y = amplitude * np.sin(angle)

    # Build strain tensor in principal frame
    eps = np.diag([e_x, e_y, 0.0])


    return np.eye(3) + eps

DEG2RAD = np.pi / 180
def rotate_atoms(atoms, rot):
    """Return a new Atoms object with positions and cell rotated by rot (radians) around Z."""
    atoms = atoms.copy()
    rot = rot * DEG2RAD
    R = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot),  np.cos(rot), 0],
        [0,            0,           1],
    ])
    new_cell = atoms.cell @ R.T
    atoms.set_cell(new_cell, scale_atoms=False)
    new_positions = atoms.positions @ R.T
    atoms.set_positions(new_positions)
    return atoms


def apply_strain_to_atoms(atoms_pristine, F: np.ndarray):
    """Return a *new* Atoms object with the deformation gradient F applied."""
    atoms = atoms_pristine.copy()
    new_cell = atoms.cell @ F
    # Fractional coordinates are invariant under homogeneous deformation
    atoms.set_cell(new_cell, scale_atoms=True)
    return atoms

# MACE calculator factor

def build_calculator(model_path: str, device: str, logger: logging.Logger):
    """Return a MACE ASE calculator. Supports mace_mp and mace_off models."""

    mp = str(model_path)

    logger.info("Loading custom MACE model: %s", mp)
    calc = MACECalculator(model_paths=mp, device=device, default_dtype="float32")

    return calc

# Relaxatio

def relax_structure(
    atoms,
    calc,
    traj_path: Path,
    fmax: float = 1e-5,
    steps: int = 1000,
    logger: Optional[logging.Logger] = None,
) -> tuple:
    """
    Relax the structure at fixed cell (cell was pre-strained).

    Returns
    --
    (relaxed_atoms, converged: bool, energy: float, stress: np.ndarray)
    """
    from ase.io import Trajectory
    from ase.optimize import LBFGS

    atoms = atoms.copy()
    atoms.calc = calc

    traj = Trajectory(str(traj_path), "w", atoms)
    opt = LBFGS(atoms, logfile=None)
    opt.attach(traj.write, interval=1)

    try:
        opt.run(fmax=fmax, steps=steps)
        converged = opt.get_number_of_steps() < steps
    except Exception as exc:
        if logger:
            logger.warning("Relaxation raised exception: %s", exc)
        converged = False
    finally:
        traj.close()

    energy = float(atoms.get_potential_energy())
    # Voigt stress in GPa: ASE returns eV/Å³
    try:
        stress_voigt = atoms.get_stress(voigt=True)  # eV/Å³
        stress_voigt = stress_voigt * 160.21766  # eV/Å³ → GPa
    except Exception:
        stress_voigt = np.zeros(6)

    return atoms, converged, energy, stress_voigt

# Phonon calculation  (ASE-native Phonons object

from typing import Optional, Tuple
from ase.dft.kpoints import BandPath
from ase.dft.kpoints import BandPath, monkhorst_pack

def run_phonons(
    atoms,
    calc,
    supercell_matrix: list,
    outdir: Path,
    delta: float = 0.01,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, np.ndarray, Phonons]:
    """
    Compute phonons with the ASE ``Phonons`` finite-difference engine.

    Outputs written to *outdir*
    
    mesh_frequencies.npy  - Full grid of calculated phonon frequencies (THz)
    cache/                - raw ASE Phonons cache files (``*.json`` per
                            displacement); retained for reproducibility.

    Returns
    --
    min_freq_thz : float
        Minimum optical-branch frequency across both the uniform BZ mesh 
        AND the high-symmetry bandpath (THz). Negative → imaginary (unstable).
    min_kpt: np.ndarray
        The fractional k-point coordinates where the minimum frequency was found.
    ph: Phonons
        The computed ASE Phonons object.
    """

    outdir.mkdir(parents=True, exist_ok=True)
    cache_prefix = str(outdir / "cache" / "phonon")
    (outdir / "cache").mkdir(exist_ok=True)

    # 1. Build Phonons with explicit cache path
    ph = Phonons(
        atoms,
        calc,
        supercell=tuple(supercell_matrix),
        delta=delta,
        name=cache_prefix
    )

    if logger:
        logger.info("[INFO] ph.run()")
    ph.run()

    if logger:
        logger.debug("[INFO] ph.cache: %s", ph.cache)

    # 2. Read + apply acoustic sum rule, then clean raw displacements
    if logger:
        logger.info("[INFO] ph.read(acoustic=True)")
    ph.read(acoustic=True)  # impose acoustic sum rule
    ph.clean()

    # 3. Define a uniform mesh strictly mapped to the true first Brillouin Zone
    mesh_size = [50, 50, 1] 
    
    if logger:
        logger.info(f"[INFO] Generating uniform full Brillouin Zone k-point mesh {mesh_size}")

    # monkhorst_pack generates a uniform grid centered around Gamma, 
    # automatically bound within [-0.5, 0.5) fractional space of the Brillouin Zone
    mesh_kpts = monkhorst_pack(mesh_size)

    # 4. Generate the high-symmetry bandpath from the cell
    cell_bandpath = atoms.cell.bandpath()
    cell_bandpath.interpolate(npoints = 200)
    path_kpts = cell_bandpath.kpts
    
    if logger:
        logger.info(f"[INFO] Generated high-symmetry bandpath with {len(path_kpts)} points")

    # 5. Combine mesh and path k-points to evaluate them in a single calculator pass
    combined_kpts = np.vstack([mesh_kpts, path_kpts])
    
    # Evaluate arbitrary flat array coordinates using an empty path string
    custom_combined_path = BandPath(path="", cell=atoms.cell, kpts=combined_kpts)
    
    # Extract the frequencies along our combined coordinates
    bs = ph.get_band_structure(custom_combined_path, verbose=False)
    combined_freqs = bs.energies[0]  # Shape: (n_combined_kpts, n_bands)

    # 6. Separate frequencies back out to keep your original file export intact
    n_mesh_points = len(mesh_kpts)
    mesh_freqs = combined_freqs[:n_mesh_points, :]
    path_freqs = combined_freqs[n_mesh_points:, :]

    # Save the raw mesh data as originally expected
    mesh_path = outdir / "mesh_frequencies.npy"
    np.save(mesh_path, mesh_freqs)
    if logger:
        logger.info("  Mesh frequencies saved → %s", mesh_path)
    path_path = outdir / "path_frequencies.npy"
    np.save(path_path, path_freqs)
    if logger:
        logger.info("  Path frequencies saved → %s", path_path)

    # 7. Find global absolute minimum across BOTH meshes and all bands
    idx_flat = np.argmin(combined_freqs)
    k_idx, band_idx = np.unravel_index(idx_flat, combined_freqs.shape)
    
    min_freq = combined_freqs[k_idx, band_idx]
    min_kpt = combined_kpts[k_idx]

    if logger:
        origin = "mesh" if k_idx < n_mesh_points else "bandpath"
        logger.info(
            "Global Min freq %.4f THz found in [%s] at k-point %s (band %d)", 
            min_freq, origin, np.round(min_kpt, 4), band_idx
        )   

    return min_freq, min_kpt, ph

DONE_SENTINEL = "DONE"

def check_done(strain_dir: Path) -> bool:
    """Return True if the calculation has already completed successfully."""
    sentinel = strain_dir / ".done"
    return sentinel.exists()


def mark_done(strain_dir: Path) -> None:
    (strain_dir / ".done").write_text(DONE_SENTINEL)


def evaluate_stability(
    atoms_pristine,
    calc,
    path,
    amplitude: float,
    angle: float,
    rot: float,
    supercell_matrix: list,
    fmax: float,
    steps: int,
    threshold: float,
    strain_dir: Path,
    logger: logging.Logger,
    delta: float = 0.01,
    angle_records: Optional[list] = None,
) -> dict:
    """
    Apply strain, relax, compute phonons, return stability record.

    Resume if already done.
    """
    strain_dir.mkdir(parents=True, exist_ok=True)
    result_file = strain_dir / "result.json"

    # Resume capability
    if check_done(strain_dir) and result_file.exists():
        logger.debug("    [SKIP] %s already complete.", strain_dir)
        with open(result_file) as f:
            record = json.load(f)
        if angle_records is not None:
            angle_records.append(record)
        return record

    t0 = time.time()

    # 1. Apply strain
    F = biaxial_strain_tensor(angle, amplitude)
    atoms_strained = apply_strain_to_atoms(atoms_pristine, F)

    # 2. Relax
    traj_path = strain_dir / "relaxation.traj"
    atoms_relaxed, converged, energy, stress = relax_structure(
        atoms_strained, calc, traj_path, fmax=fmax, steps=steps, logger=logger
    )

    # 3. Phonons
    try:
        min_freq, min_kpt, _ = run_phonons(
            atoms_relaxed, calc, supercell_matrix,
            strain_dir / "phonons",
            delta=delta,
            logger=logger,
        )
        phonon_ok = True
    except Exception as exc:
        logger.warning("    Phonon calculation failed: %s", exc)
        logger.debug(traceback.format_exc())
        min_freq = float("nan")
        min_kpt = np.array([float("nan"), float("nan"), float("nan")])
        phonon_ok = False

    # 4. Stability criterion
    if phonon_ok:
        stable = bool(min_freq > -threshold)
    else:
        stable = False

    elapsed = time.time() - t0
    record = {
        "amplitude": amplitude,
        "angle_rad": angle,
        "rot_rad": rot,
        "stable": stable,
        "min_freq_thz": min_freq,
        "min_kpt": min_kpt.tolist(),
        "relaxation_converged": converged,
        "energy_ev": energy,
        "stress_voigt": stress.tolist(),
        "elapsed_s": elapsed,
        "phonon_ok": phonon_ok,
    }

    with open(result_file, "w") as f:
        json.dump(record, f, indent=2)

    mark_done(strain_dir)

    logger.debug(
        "    amp=%.5f  stable=%s  min_freq=%.4f THz  conv=%s  (%.1fs)",
        amplitude,
        stable,
        min_freq if phonon_ok else float("nan"),
        converged,
        elapsed,
    )
    
    if angle_records is not None:
        angle_records.append(record)
        
    return record

# Bisection searc

def bisection_search(
    atoms_pristine,
    calc,
    path,
    angle: float,
    rot: float,
    min_strain: float,
    max_strain: float,
    convergence_threshold: float,
    supercell_matrix: list,
    fmax: float,
    steps: int,
    freq_threshold: float,
    angle_dir: Path,
    logger: logging.Logger,
    delta: float = 0.01,
    angle_records: Optional[list] = None,
) -> dict:
    """
    Binary search for the critical strain amplitude along one angle.

    Returns a summary dict with critical amplitude and boundary stress tensor.
    """
    lo, hi = min_strain, max_strain
    lo_record = hi_record = None
    iteration = 0

    logger.info(
        "  Bisection for angle=%.4f rad (%.1f°): searching [%.4f, %.4f]",
        angle,
        np.degrees(angle),
        lo,
        hi,
    )

    # First: verify boundary conditions
    def _eval(amp: float) -> dict:
        strain_dir = angle_dir / f"strain_{amp:.6f}"
        return evaluate_stability(
            atoms_pristine, calc, path, amp, angle, rot,
            supercell_matrix, fmax, steps, freq_threshold,
            strain_dir, logger,
            delta=delta,
            angle_records=angle_records
        )
    
    grid_strains = np.linspace(min_strain, max_strain, 21, endpoint = True)
    grid_recs = []

    for s in grid_strains:
        grid_recs.append(_eval(s))

    if not grid_recs[0]["stable"]:
        logger.warning(
            "  min_strain=%.4f is already UNSTABLE. Boundary is at or below min_strain.", lo
        )
        return {
            "angle_rad": angle,
            "critical_amplitude": lo,
            "boundary_found": False,
            "reason": "min_strain_unstable",
            "lo_record": grid_recs[0],
            "hi_record": grid_recs[-1],
        }

    found = False
    for i, (s, gr) in enumerate(zip(grid_strains, grid_recs)):
        if not gr["stable"]:
            hi = s
            lo = grid_strains[i-1]
            lo_record = grid_recs[i-1]
            hi_record = grid_recs[i]
            logger.info("found instability in range %f %f", lo, hi)
            found = True
            break
    
    # here we can actually check that the system was NEVER unstable during the scan
    if not found:
        logger.warning(
            "  max_strain=%.4f is still STABLE. The system is never unstable on a grid. Either try a finer grid, or consider increasing the max_strain"
        )
        return {
            "angle_rad": angle,
            "critical_amplitude": hi,
            "boundary_found": False,
            "reason": "max_strain_stable",
            "lo_record": grid_recs[0],
            "hi_record": grid_recs[-1],
        }

    # Bisection loop
    while (hi - lo) > convergence_threshold:
        mid = 0.5 * (lo + hi)
        iteration += 1
        logger.info(
            "  [iter %2d] lo=%.5f  hi=%.5f  mid=%.5f  (gap=%.5f)",
            iteration, lo, hi, mid, hi - lo,
        )
        rec = _eval(mid)
        if rec["stable"]:
            lo = mid
            lo_record = rec
        else:
            hi = mid
            hi_record = rec

    critical = 0.5 * (lo + hi)
    logger.info(
        "  → Critical amplitude: %.5f (converged to within %.1e)",
        critical,
        convergence_threshold,
    )

    # Use the hi (first-unstable) record for stress tensor at boundary
    boundary_rec = hi_record

    return {
        "angle_rad": angle,
        "angle_deg": float(np.degrees(angle)),
        "critical_amplitude": critical,
        "lo_amplitude": lo,
        "hi_amplitude": hi,
        "boundary_found": True,
        "boundary_min_freq_thz": boundary_rec.get("min_freq_thz"),
        "boundary_min_kpt": boundary_rec.get("min_kpt"),
        "boundary_stress_voigt": boundary_rec.get("stress_voigt"),
        "boundary_energy_ev": boundary_rec.get("energy_ev"),
        "n_iterations": iteration,
    }
# Per-angle summary CSV

def write_angle_summary(angle_dir: Path, records: list[dict]) -> None:
    """Write all evaluated strain points for one angle to summary.csv."""
    csv_path = angle_dir / "summary.csv"
    if not records:
        return
    fieldnames = [
        "amplitude", "stable", "min_freq_thz", "boundary_min_kpt", "relaxation_converged",
        "energy_ev",
        "stress_xx", "stress_yy", "stress_zz",
        "stress_yz", "stress_xz", "stress_xy",
        "elapsed_s", "phonon_ok",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            sv = r.get("stress_voigt", [0] * 6)
            writer.writerow({
                "amplitude": r.get("amplitude"),
                "stable": r.get("stable"),
                "min_freq_thz": r.get("min_freq_thz"),
                "boundary_min_kpt": r.get("min_kpt"),
                "relaxation_converged": r.get("relaxation_converged"),
                "energy_ev": r.get("energy_ev"),
                "stress_xx": sv[0],
                "stress_yy": sv[1],
                "stress_zz": sv[2],
                "stress_yz": sv[3],
                "stress_xz": sv[4],
                "stress_xy": sv[5],
                "elapsed_s": r.get("elapsed_s"),
                "phonon_ok": r.get("phonon_ok"),
            })
# Main drive

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map the phonon instability boundary of graphene under biaxial strain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_g = p.add_argument_group("Model")
    model_g.add_argument(
        "--model_path",
        type=str,
        default="medium",
        help=(
            "Path to a MACE .model file, or a shorthand for MACE-MP-0 sizes "
            "('small', 'medium', 'large') or 'mace_off'."
        ),
    )
    model_g.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")

    # Relaxation
    relax_g = p.add_argument_group("Relaxation")
    relax_g.add_argument("--fmax", type=float, default=1e-5,
                         help="Force convergence criterion (eV/Å).")
    relax_g.add_argument("--steps", type=int, default=1000,
                         help="Maximum number of relaxation steps.")

    # Phonons
    phonon_g = p.add_argument_group("Phonons")
    phonon_g.add_argument("--supercell", type=int, nargs=3, default=[8, 8, 1],
                          metavar=("SX", "SY", "SZ"),
                          help="ASE Phonons supercell dimensions.")
    phonon_g.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Finite-difference displacement magnitude (Å) passed to ASE Phonons.",
    )
    phonon_g.add_argument(
        "--kpath",
        type=str,
        default="GMKG",
        help=(
            "Special-point path string for the band structure "
            "(passed to atoms.cell.bandpath). "
            "Graphene default: 'GMKG'."
        ),
    )
    phonon_g.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Imaginary frequency threshold (THz). Stable if min_freq > -threshold.",
    )

    # Search
    search_g = p.add_argument_group("Search")
    search_g.add_argument("--min_strain", type=float, default=0.0,
                           help="Minimum biaxial strain amplitude.")
    search_g.add_argument("--max_strain", type=float, default=0.20,
                           help="Maximum biaxial strain amplitude.")
    search_g.add_argument("--rot", type=float, default=0.0,
                           help="Fixed rotation of strain axes around Z (radians).")
    search_g.add_argument(
        "--convergence_threshold",
        type=float,
        default=1e-4,
        help="Bisection convergence: stop when hi-lo < this value.",
    )

    restart_g = p.add_argument_group("Restart parameters")
    restart_g.add_argument("--final_boundary_file")

    # Output
    io_g = p.add_argument_group("Output")
    io_g.add_argument("--outdir", type=Path, default=Path("phonon_strain_results"),
                       help="Root output directory.")
    io_g.add_argument("--verbose", action="store_true",
                       help="Enable DEBUG-level console output.")

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(outdir, verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("Phonon Instability Boundary Restart — graphene under biaxial strain")
    logger.info("=" * 70)
    logger.info("Parameters:")
    for k, v in vars(args).items():
        logger.info("  %-28s = %s", k, v)
    logger.info("=" * 70)

    # Build calculator
    calc = build_calculator(args.model_path, args.device, logger)

    # Build pristine graphene unit cell
    atoms_pristine, converged = make_graphene_unitcell(calc)
    atoms_pristine = rotate_atoms(atoms_pristine, args.rot)
    logger.info(
        "Pristine graphene: %d atoms, cell=\n%s, converged=%d",
        len(atoms_pristine),
        atoms_pristine.cell,
        converged
    )

    path = atoms_pristine.cell.bandpath("GMKG", npoints=200)

    # 1. Load original dataframe
    df_master = pd.read_csv(args.final_boundary_file, sep=",")
    
    # 2. Filter out ONLY the entries that failed (where boundary_found is False)
    rerun_entries = df_master[df_master['boundary_found'] == False]
    
    logger.info("Found %d entries to re-run out of %d total records.", len(rerun_entries), len(df_master))

    # Main loop: iterating over the filtered dataframe using its original index
    for progress_idx, (orig_df_idx, row) in tqdm(enumerate(rerun_entries.iterrows()), total=len(rerun_entries)):
        
        angle = row['angle_rad'] 
        angle_deg = row['angle_deg']
        
        angle_dir = outdir / f"angle_{orig_df_idx:03d}"
        angle_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("━" * 70)
        logger.info(
            "Rerun Progress %d/%d (Original Index %d) : %.4f rad = %.2f°",
            progress_idx + 1, len(rerun_entries), orig_df_idx, angle, angle_deg,
        )
        logger.info("━" * 70)

        angle_records: list[dict] = []

        try:
            boundary = bisection_search(
                atoms_pristine=atoms_pristine,
                calc=calc,
                path=path,
                angle=angle,
                rot=args.rot,
                min_strain=args.min_strain,
                max_strain=args.max_strain,
                convergence_threshold=args.convergence_threshold,
                supercell_matrix=args.supercell,
                fmax=args.fmax,
                steps=args.steps,
                freq_threshold=args.threshold,
                angle_dir=angle_dir,
                logger=logger,
                delta=args.delta,
                angle_records=angle_records,
            )
        except Exception as exc:
            logger.error("Bisection for original index %d FAILED: %s", orig_df_idx, exc)
            logger.debug(traceback.format_exc())
            boundary = {
                "angle_rad": float(angle),
                "angle_deg": float(angle_deg),
                "critical_amplitude": float("nan"),
                "boundary_found": False,
                "lo_amplitude": float("nan"),
                "hi_amplitude": float("nan"),
                "boundary_min_freq_thz": float("nan"),
                "boundary_min_kpt": None,
                "boundary_stress_voigt": [float("nan")] * 6,
                "boundary_energy_ev": float("nan"),
                "n_iterations": 0,
                "reason": f"exception: {exc}",
            }

        boundary["angle_index"] = orig_df_idx
        
        # 3. Inject the results back into the master DataFrame
        df_master = _update_dataframe_row(df_master, orig_df_idx, boundary)

        # Write per-angle summary
        write_angle_summary(angle_dir, angle_records)

        # 4. Save progress update out to target file safely
        df_master.to_csv(outdir / "final_boundary.csv", index=False)

        logger.info(
            "  ✓ Critical amplitude = %s  (boundary_found=%s)",
            f"{boundary.get('critical_amplitude'):.5f}" if not pd.isna(boundary.get('critical_amplitude')) else "NaN",
            boundary.get("boundary_found"),
        )

    # Final summary displays
    df_master.to_csv(outdir / "final_boundary.csv", index=False)
    logger.info("")
    logger.info("=" * 70)
    logger.info("Run complete. Results in: %s", outdir.resolve())
    logger.info("Final boundary CSV updated in place: %s", outdir / "final_boundary.csv")
    logger.info("=" * 70)
    
    _print_boundary_table(df_master.to_dict(orient='records'), logger)

    return 0


def _update_dataframe_row(df: pd.DataFrame, idx: int, record: dict) -> pd.DataFrame:
    """Unpacks dictionary keys explicitly to clear cached/residual row data elements."""
    sv = record.get("boundary_stress_voigt")
    if sv is None or not isinstance(sv, (list, np.ndarray)) or len(sv) < 6:
        sv = [float("nan")] * 6
    
    df.at[idx, "angle_index"] = record.get("angle_index", idx)
    df.at[idx, "angle_deg"] = record.get("angle_deg")
    df.at[idx, "angle_rad"] = record.get("angle_rad")
    df.at[idx, "critical_amplitude"] = record.get("critical_amplitude")
    df.at[idx, "boundary_found"] = record.get("boundary_found")
    df.at[idx, "lo_amplitude"] = record.get("lo_amplitude")
    df.at[idx, "hi_amplitude"] = record.get("hi_amplitude")
    df.at[idx, "boundary_min_freq_thz"] = record.get("boundary_min_freq_thz")
    df.at[idx, "n_iterations"] = record.get("n_iterations")
    
    # Explicitly clear or overwrite string columns to please strict Arrow engines
    min_kpt = record.get("boundary_min_kpt")
    if min_kpt is not None and not (isinstance(min_kpt, float) and np.isnan(min_kpt)):
        df.at[idx, "boundary_min_kpt"] = str(min_kpt)
    else:
        df.at[idx, "boundary_min_kpt"] = None

    reason = record.get("reason")
    df.at[idx, "reason"] = str(reason) if reason is not None else None

    # Write stresses cleanly
    df.at[idx, "boundary_stress_xx"] = sv[0]
    df.at[idx, "boundary_stress_yy"] = sv[1]
    df.at[idx, "boundary_stress_zz"] = sv[2]
    df.at[idx, "boundary_stress_yz"] = sv[3]
    df.at[idx, "boundary_stress_xz"] = sv[4]
    df.at[idx, "boundary_stress_xy"] = sv[5]
    
    df.at[idx, "boundary_energy_ev"] = record.get("boundary_energy_ev", float("nan"))
        
    return df


def _print_boundary_table(records: list[dict], logger: logging.Logger) -> None:
    logger.info("")
    logger.info("  %-6s  %-10s  %-12s  %-10s  %-10s", "Idx", "Angle(°)", "Crit.Amp.", "Found", "MinFreq(THz)")
    logger.info("  " + "-" * 56)
    for r in records:
        min_freq = r.get('boundary_min_freq_thz')
        try:
            freq_str = f"{float(min_freq):.4f}" if (min_freq is not None and not pd.isna(min_freq)) else "N/A"
        except (ValueError, TypeError):
            freq_str = "N/A"

        crit_amp = r.get("critical_amplitude")
        amp_str = f"{float(crit_amp):.5f}" if (crit_amp is not None and not pd.isna(crit_amp)) else "N/A"

        logger.info(
            "  %-6s  %-10.2f  %-12s  %-10s  %-10s",
            r.get("angle_index", "?"),
            r.get("angle_deg", float("nan")),
            amp_str,
            r.get("boundary_found", "?"),
            freq_str,
        )


if __name__ == "__main__":
    sys.exit(main())