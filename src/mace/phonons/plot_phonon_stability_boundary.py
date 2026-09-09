import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ase import Atoms
from ase.build import graphene
from ase.dft.kpoints import monkhorst_pack
from ase.geometry import find_mic
import os
from argparse import ArgumentParser
from pprint import pprint
from scipy.spatial import Voronoi

DEG2RAD = np.pi / 180.

def parse_args():
    parser = ArgumentParser(description="Separates phonon stability plots and adds background meshgrid/bandpath details.")
    parser.add_argument("--summary_file", "-f", required=True, help="Path to the summary CSV file")
    parser.add_argument("--rot", type=float, default=0.0, help="Rotation angle applied to pristine atoms (degrees)")
    
    args = parser.parse_args()
    print("Running with parameters:")
    print("="*70)
    pprint(vars(args))
    print("="*70)
    return args

def get_pristine_graphene():
    try:
        atoms = graphene(vacuum=100)
    except Exception:
        a = 2.46  # lp is not really important for the plot since kpoints are handled through fractional coordinates, and the symmetries are the same
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
    return atoms

def rotate_atoms(atoms, rot_deg):
    atoms = atoms.copy()
    rot_rad = rot_deg * DEG2RAD
    R = np.array([
        [np.cos(rot_rad), -np.sin(rot_rad), 0],
        [np.sin(rot_rad),  np.cos(rot_rad), 0],
        [0,                0,               1],
    ])
    atoms.set_cell(atoms.cell @ R.T, scale_atoms=False)
    atoms.set_positions(atoms.positions @ R.T)
    return atoms

def biaxial_strain_tensor(angle_rad, amplitude):
    e_x = amplitude * np.cos(angle_rad)
    e_y = amplitude * np.sin(angle_rad)
    eps = np.diag([e_x, e_y, 0.0])
    return np.eye(3) + eps

def apply_strain_to_atoms(atoms_pristine, F):
    atoms = atoms_pristine.copy()
    atoms.set_cell(atoms.cell @ F, scale_atoms=True)
    return atoms

def get_exact_wigner_seitz_2d(b1, b2):
    """
    Computes the exact 2D Brillouin Zone polygon using a Voronoi 
    decomposition safely without hardcoded index assumptions.
    """
    shifts = [-2, -1, 0, 1, 2]
    points = []
    for m in shifts:
        for n in shifts:
            points.append(m * b1 + n * b2)
    points = np.array(points)
    
    vor = Voronoi(points)
    
    # Dynamically find the index closest to [0,0] to protect against shearing re-ordering
    gamma_idx = np.argmin(np.linalg.norm(points, axis=1))
    
    region_idx = vor.point_region[gamma_idx]
    v_vertices_idx = vor.regions[region_idx]
    
    # Filter out any infinite region markers (-1) if they exist
    v_vertices_idx = [v for v in v_vertices_idx if v != -1]
    
    vertices = vor.vertices[v_vertices_idx]
    angles = np.arctan2(vertices[:, 1], vertices[:, 0])
    sorted_vertices = vertices[np.argsort(angles)]
    
    # Close the polygon loop cleanly
    return np.vstack([sorted_vertices, sorted_vertices[0]])



def clean_kpt_string(kpt_str):
    """Parses standard string configurations like '[0.1, 0.2, 0.0]' safely into an array."""
    cleaned = kpt_str.strip().replace('[', '').replace(']', '').replace(',', ' ')
    return np.fromstring(cleaned, sep=' ')

def generate_separated_plots(df, atoms_base, output_dir):
    df_valid = df[df['boundary_found'] == True].dropna(subset=['critical_amplitude']).copy()
    n_points = len(df_valid)
    
    if n_points == 0:
        print("No valid boundary points found to plot.")
        return

    df_valid['xs'] = df_valid['critical_amplitude'] * np.cos(DEG2RAD * df_valid['angle_deg'])
    df_valid['ys'] = df_valid['critical_amplitude'] * np.sin(DEG2RAD * df_valid['angle_deg'])
    
    df_valid['stress_norm'] = np.sqrt(
        df_valid['boundary_stress_xx']**2 +
        df_valid['boundary_stress_yy']**2 +
        2 * (df_valid['boundary_stress_xy']**2)
    ) * 100

    # -------------------------------------------------------------------------
    # PLOT 1: The Macro Mechanical Stability Boundary
    # -------------------------------------------------------------------------
    fig_macro, ax_main = plt.subplots(figsize=(8, 6))
    sc = ax_main.scatter(
        df_valid['xs'], df_valid['ys'], 
        c=df_valid['stress_norm'], cmap='viridis', 
        s=30, edgecolors='k', zorder=3
    )
    cbar = fig_macro.colorbar(sc, ax=ax_main)
    cbar.set_label('Stress Frobenius Norm')
    ax_main.set_xlabel('Critical Amplitude $\epsilon_x$')
    ax_main.set_ylabel('Critical Amplitude $\epsilon_y$')
    ax_main.grid(True, linestyle='--', alpha=0.5)
    ax_main.axis('equal')
    
    macro_path = os.path.join(output_dir, "boundary_macro.svg")
    plt.tight_layout()
    plt.savefig(macro_path, bbox_inches='tight')
    plt.close(fig_macro)
    print(f"Saved boundary map to: {macro_path}")

    # -------------------------------------------------------------------------
    # PLOT 2: The Brillouin Zone Breakdown Grid
    # -------------------------------------------------------------------------
    colors = sc.to_rgba(df_valid['stress_norm']) 
    discard=5
    selected = df_valid.iloc()[::discard]
    nplots = n_points // 5
    n_cols = 4
    n_rows = int(np.ceil(nplots / n_cols))
    
    fig_bz, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    # Generate the base uniform mesh grid in fractional coordinates
    mesh_size = [50, 50, 1]
    mesh_kpts_frac = monkhorst_pack(mesh_size)

    for i, (idx, row) in enumerate(selected.iterrows()):
        ax = axes[i]
        
        angle_rad = row['angle_rad']
        amplitude = row['critical_amplitude']
        F = biaxial_strain_tensor(angle_rad, amplitude)
        strained_atoms = apply_strain_to_atoms(atoms_base, F)
        
        # Pull reciprocal cell vectors
        rcell = strained_atoms.cell.reciprocal()
        b1, b2 = rcell[0][:2], rcell[1][:2] 
        
        # Calculate dynamic safely-indexed Wigner-Seitz boundary
        bz_polygon = get_exact_wigner_seitz_2d(b1, b2)
        ax.plot(bz_polygon[:, 0], bz_polygon[:, 1], color='dimgray', linestyle='--', linewidth=0.9, alpha=0.8, zorder=3)

        # Convert fractional k-points to Cartesian using matrix multiplication (@)
        mesh_kpts_cart = mesh_kpts_frac @ rcell
        wrapped_mesh_cart, _ = find_mic(mesh_kpts_cart, rcell, pbc=[True, True, False])
        ax.scatter(wrapped_mesh_cart[:, 0], wrapped_mesh_cart[:, 1], color='lightgray', s=1.2, alpha=0.3, zorder=1)

        # Wrap high symmetry baseline path using actual ASE minimum image convention helper
        bp = strained_atoms.cell.bandpath(npoints=200)
        path_kpts_cart = bp.kpts @ rcell
        ax.scatter(path_kpts_cart[:, 0], path_kpts_cart[:, 1], color='darkgray', s=2.5, alpha=0.6, zorder=2)
        
        # Wrap minimum boundary phonon point 
        try:
            kpt_frac = clean_kpt_string(str(row['boundary_min_kpt']))
            kpt_frac[2] = 0.0  # Strip out-of-plane elements
            kpt_cart = kpt_frac @ rcell
            # Reshape context for find_mic expectation matrix structure
            wrapped_crit_cart, _ = find_mic(kpt_cart.reshape(1, 3), rcell, pbc=[True, True, False])
            
            # ax.scatter(wrapped_crit_cart[0, 0], wrapped_crit_cart[0, 1], color=colors[i], edgecolors='k', s=35, zorder=5)
            ax.scatter(wrapped_crit_cart[0, 0], wrapped_crit_cart[0, 1], marker = "x", s=100, color = "red")
        except Exception:
            pass

        # Gamma crosshair marker
        ax.plot(0, 0, marker='+', color='red', alpha=0.5, markersize=7, zorder=4)
        
        ax.set_title(f"$\phi$ = {row['angle_deg']:.0f}°\n$\epsilon$ = {amplitude:.4f}", fontsize=20)
        
        # Dynamically fit plot scale limits to the generated polygon scale bounds
        max_r = np.max(np.linalg.norm(bz_polygon, axis=1)) * 1.1
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        fig_bz.delaxes(axes[j])

    bz_path_svg = os.path.join(output_dir, "boundary_kpoints_bz.svg")
    bz_path_png = os.path.join(output_dir, "boundary_kpoints_bz.png")
    bz_path_pdf = os.path.join(output_dir, "boundary_kpoints_bz.pdf")
    plt.tight_layout()
    plt.savefig(bz_path_svg, bbox_inches='tight')
    plt.savefig(bz_path_png, bbox_inches='tight', dpi=300)
    plt.savefig(bz_path_pdf, bbox_inches='tight')
    plt.close(fig_bz)
    print(f"Saved precise Brillouin Zone sampling maps to:\n -> {bz_path_svg}\n -> {bz_path_png}\n -> {bz_path_pdf}")

def main():
    args = parse_args()
    
    df = pd.read_csv(args.summary_file)
    output_dir = os.path.dirname(args.summary_file) or '.'
    
    atoms_pristine = get_pristine_graphene()
    atoms_rotated = rotate_atoms(atoms_pristine, args.rot)
    
    generate_separated_plots(df, atoms_rotated, output_dir)
    return 0

if __name__ == "__main__":
    main()