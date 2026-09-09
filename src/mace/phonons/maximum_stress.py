import ase
from mace.calculators import MACECalculator
import os
import pandas as pd
from argparse import ArgumentParser
from ase.build import graphene
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize_scalar

from ase.optimize import BFGS
from ase import Atoms, Atom
from ase.io import write

DEG2RAD = np.pi / 180.0

def parse_args():
    parser = ArgumentParser(
        description="Find maximum strain using coarse scan + precise optimization"
    )
    parser.add_argument("--min_strain", type=float, default=0.15)
    parser.add_argument("--max_strain", type=float, default=0.30)
    parser.add_argument("--nstrains", type=int, default=30)
    parser.add_argument("--ncombinations", type=int, default=45)
    parser.add_argument("--theta", type=float, default=0)
    parser.add_argument("--supercell", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--lattice_parameter", "-lp", type=float, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", "-o", type=str, default="MAXIMUM_STRESS")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument(
        "--kcell",
        action="store_true",
        help="Use the K-cell (6-atom, 3x1 supercell of the primitive cell along a1).",
    )
    parser.add_argument(
        "--objective",
        choices=["frobenius", "lateral"],
        default="frobenius",
        help="Quantity to maximise during the strain scan.",
    )
    return parser.parse_args()

# Cell builders 

def generate_primitivecell(latticeparam=2.46):
    a = latticeparam
    cell = np.array([
        [0.5 * np.sqrt(3) * a,  0.5 * a, 0],
        [0,                      a,       0],
        [0,                      0,       100.0],
    ])
    positions_frac = np.array([[1/3, 1/3, 0],
                                 [2/3, 2/3, 0]])
    xyz = positions_frac @ cell
    return Atoms("C2", positions=xyz, cell=cell, pbc=(True, True, True))

def generate_kcell(latticeparam):
    l0 = latticeparam / np.sqrt(3)
    KCELL0 = 3*l0* np.array([
        [1,0,0],
        [0.5, np.sqrt(3)/2, 0],
        [0,0,100 / (3*l0)]
    ])
    KBASE0 = (1/3.) * np.array([
        [1, 0, 0], [2, 0, 0], [2, 1, 0],
        [1, 2, 0], [0, 1, 0], [0, 2, 0]
    ])
    katoms = Atoms("C6", positions = KBASE0 @ KCELL0, cell=KCELL0, pbc = [True, True, True])
    return katoms

# Strain mechanics 

def rotate_strain(matrix, theta_rad):
    cos, sin = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    return R.T @ matrix @ R

def frobenius(matrix):
    return np.sqrt(np.trace(matrix.T @ matrix))

def lateral_stress(matrix):
    return matrix[0, 0] + matrix[1, 1]

def apply_strain(atoms_template, cell0, s, angle, theta_rad, calc, optimize, save=None):
    atoms = atoms_template.copy()
    atoms.calc = calc

    exx_loc = s * np.cos(angle)
    eyy_loc = s * np.sin(angle)
    E_local = np.array([[exx_loc, 0, 0],
                        [0,       eyy_loc, 0],
                        [0,       0,       0]])
    E_global = rotate_strain(E_local, theta_rad)
    F = np.eye(3) + E_global
    atoms.set_cell(cell0 @ F, scale_atoms=True)
    atoms.pbc = [True, True, True]

    if optimize:
        perturbation = np.random.normal(0, 1e-2, size = (len(atoms), 3))
        atoms.positions += perturbation
        converged = BFGS(atoms, logfile=None).run(fmax=1e-4, steps = 1000)

    # Note: 'save' parameter is kept for signature consistency but ignored here
    # as we handle writing in the main loop now.
    return atoms.get_stress(voigt=False), atoms.get_cell()[:], atoms, converged


# Per-angle worker 

def process_angle(angle, atoms_template, cell0, theta_rad, strain_magnitudes, calc, optimize, objective_fn):
    norms    = []
    stresses = []
    cells    = []
    coarse_atoms = []
    convergences = []

    for s in strain_magnitudes:
        stress, cell, atoms_out, converged = apply_strain(atoms_template, cell0, s, angle, theta_rad, calc, optimize)
        norm = objective_fn(stress)

        norms.append(norm)
        stresses.append(stress.copy())
        cells.append(cell.copy())
        coarse_atoms.append(atoms_out)
        convergences.append(converged)


        if len(norms) >= 5:
            peak_idx_so_far  = np.argmax(norms)
            points_past_peak = len(norms) - 1 - peak_idx_so_far
            if norms[-1] < 0.80 * norms[peak_idx_so_far] and points_past_peak >= 3:
                break

    norms        = np.array(norms)
    strains_used = strain_magnitudes[:len(norms)]
    peak_idx     = np.argmax(norms)

    s_lo = strains_used[max(0, peak_idx - 1)]
    s_hi = strains_used[min(len(strains_used) - 1, peak_idx + 1)]

    def neg_objective(s):
        stress, _, _, _ = apply_strain(atoms_template, cell0, s, angle, theta_rad, calc, optimize)
        return -objective_fn(stress)

    res = minimize_scalar(
        neg_objective, bounds=(s_lo, s_hi), method="bounded",
        options={"xatol": 1e-4, "maxiter": 20},
    )

    refined_s = res.x
    refined_stress, refined_cell, peak_atoms, converged = apply_strain(
        atoms_template, cell0, refined_s, angle, theta_rad, calc, optimize
    )

    sa = np.array(stresses)
    ca = np.array(cells)

    curve_df = pd.DataFrame({
        "strain_magnitude": strains_used,
        "stress_norm":      norms,
        "sxx": sa[:, 0, 0], "sxy": sa[:, 0, 1], "sxz": sa[:, 0, 2],
        "syx": sa[:, 1, 0], "syy": sa[:, 1, 1], "syz": sa[:, 1, 2],
        "szx": sa[:, 2, 0], "szy": sa[:, 2, 1], "szz": sa[:, 2, 2],
        "cell_a1x": ca[:, 0, 0], "cell_a1y": ca[:, 0, 1], "cell_a1z": ca[:, 0, 2],
        "cell_a2x": ca[:, 1, 0], "cell_a2y": ca[:, 1, 1], "cell_a2z": ca[:, 1, 2],
        "cell_a3x": ca[:, 2, 0], "cell_a3y": ca[:, 2, 1], "cell_a3z": ca[:, 2, 2],
        "converged" : convergences
    })

    rc, rs = refined_cell, refined_stress
    summary = {
        "angle":              angle,
        "coarse_peak_strain": strains_used[peak_idx],
        "coarse_peak_norm":   norms[peak_idx],
        "coarse_peak_converged": convergences[peak_idx],
        "max_strain_mag":     refined_s,
        "max_exx":            refined_s * np.cos(angle),
        "max_eyy":            refined_s * np.sin(angle),
        "peak_norm":          -res.fun,
        "peak_converged": converged,
        "sxx": rs[0, 0], "sxy": rs[0, 1], "sxz": rs[0, 2],
        "syx": rs[1, 0], "syy": rs[1, 1], "syz": rs[1, 2],
        "szx": rs[2, 0], "szy": rs[2, 1], "szz": rs[2, 2],
        "cell_a1x": rc[0, 0], "cell_a1y": rc[0, 1], "cell_a1z": rc[0, 2],
        "cell_a2x": rc[1, 0], "cell_a2y": rc[1, 1], "cell_a2z": rc[1, 2],
        "cell_a3x": rc[2, 0], "cell_a3y": rc[2, 1], "cell_a3z": rc[2, 2],
    }

    return summary, curve_df, coarse_atoms, peak_atoms


# Main 

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.kcell:
        atoms_template = generate_kcell(args.lattice_parameter)
        cell_type = "kcell"
    else:
        atoms_template = generate_primitivecell(args.lattice_parameter)
        cell_type = "primitive"

    if args.supercell != [1, 1, 1]:
        atoms_template = atoms_template.repeat(args.supercell)

    cell0     = atoms_template.get_cell()[:]
    Lz        = cell0[2, 2]
    theta_rad = args.theta * DEG2RAD

    calc = MACECalculator(args.model, device=args.device)
    objective_fn = frobenius if args.objective == "frobenius" else lateral_stress

    combinations_angle = np.linspace(0, np.pi / 2.0, args.ncombinations)
    strain_magnitudes  = np.linspace(args.min_strain, args.max_strain, args.nstrains)

    results_data = []
    structure_storage = []

    for angle in tqdm(combinations_angle, desc="angles"):
        summary, curve_df, coarse_list, peak_atoms = process_angle(
            angle, atoms_template, cell0, theta_rad,
            strain_magnitudes, calc, args.optimize, objective_fn
        )
        results_data.append(summary)

        angle_deg  = np.degrees(angle)
        curve_path = os.path.join(args.outdir, f"stress_strain_angle{angle_deg:.2f}deg.csv")
        curve_df.to_csv(curve_path, index=False)
        
        # Store atoms for batch writing
        structure_storage.append((angle_deg, coarse_list, peak_atoms))

        # Batch writing at the end
        print("\nWriting relaxed structures to extxyz format...")
        angle_dir = os.path.join(args.outdir, f"RELAXED_ANGLE_{angle_deg:.2f}deg")
        os.makedirs(angle_dir, exist_ok=True)
        
        for i, atoms in enumerate(coarse_list):
            fname = os.path.join(angle_dir, f"step_{i:03d}.extxyz")
            write(fname, atoms, format="extxyz")
            
        write(os.path.join(angle_dir, "REFINED_PEAK.extxyz"), peak_atoms, format="extxyz")

    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(args.outdir, "results.csv"), index=False)

    # Color plot 
    color_values = (df["sxx"] + df["syy"]) / np.sqrt(2) * Lz
    cbar_label   = r"$(\sigma_{xx}+\sigma_{yy})\,/\,\sqrt{2}\ \times\ L_z$" + "\n[eV/Å²]"

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        df["max_exx"], df["max_eyy"],
        c=color_values, cmap="viridis",
        s=40, edgecolors="k", linewidths=0.4,
    )
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel(r"$\varepsilon_{xx}$ (at failure)")
    ax.set_ylabel(r"$\varepsilon_{yy}$ (at failure)")
    ax.set_title(f"Instability Boundary (θ={args.theta}°, {cell_type}, obj={args.objective})")
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 0.30)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "instability_boundary.pdf"), dpi=150)

    print(f"Done! Results and structures saved in {args.outdir}")


if __name__ == "__main__":
    main()