from ase.build import graphene
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
from ase import Atom, Atoms

def parse_args():
    parser = ArgumentParser(description="Plot phonon dispersion given amplitude, angle, rot, and a .npy of frequencies along the path.")
    parser.add_argument("--freqs", "-f", type=str, required=True, help="Path to the .npy file containing frequencies")
    parser.add_argument("--amplitude", "-am", type=float, default=0.0, help="Strain amplitude")
    parser.add_argument("--angle", "-an", type=float, default=0.0, help="Strain angle in radians")
    parser.add_argument("--rot", "-r", type=float, default=0.0, help="Rotation angle in degrees")
    return parser.parse_args()

def make_graphene_unitcell():
    """Return a pristine graphene unit cell as an ASE Atoms object."""
    try:
        atoms = graphene(vacuum=100)
    except Exception:
        # Fallback: build manually if ase.build.graphene fails
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
    return atoms

def biaxial_strain_tensor(angle: float, amplitude: float) -> np.ndarray:
    """
    Construct a 3x3 deformation gradient F for biaxial strain.
    angle     : Strain anisotropy angle in degrees [0, 90].
    amplitude : Magnitude of the strain (engineering strain).
    """
    angle = np.deg2rad(angle)
    e_x = amplitude * np.cos(angle)
    e_y = amplitude * np.sin(angle)
    eps = np.diag([e_x, e_y, 0.0])
    return np.eye(3) + eps

DEG2RAD = np.pi / 180
def rotate_atoms(atoms, rot_degrees):
    """Return a new Atoms object with positions and cell rotated around Z."""
    atoms = atoms.copy()
    rot_rad = rot_degrees * DEG2RAD
    R = np.array([
        [np.cos(rot_rad), -np.sin(rot_rad), 0],
        [np.sin(rot_rad),  np.cos(rot_rad), 0],
        [0,                0,               1],
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

def build_system(amplitude, angle, rot):
    atoms = make_graphene_unitcell()
    atoms = rotate_atoms(atoms, rot)
    deformation = biaxial_strain_tensor(angle, amplitude)
    atoms = apply_strain_to_atoms(atoms, deformation)
    return atoms

def main():
    args = parse_args()

    # Load your computed phonon frequencies 
    # Shape is (45, 6) -> 45 k-points, 6 branches
    freqs = np.load(args.freqs)
    n_kpoints = freqs.shape[0] 

    # Rebuild the atomic system
    atoms = build_system(args.amplitude, args.angle, args.rot)
    
    # Extract the default BandPath object
    path = atoms.cell.bandpath()
    
    # --- THE FIX ---
    # Generate the standard un-interpolated k-axis to grab the labels and tick positions
    original_x_coords, x_ticks, x_labels = path.get_linear_kpoint_axis()
    
    # Force create a perfectly matched 1D X-axis from the start to the end of the path
    x_coords = np.linspace(original_x_coords[0], original_x_coords[-1], n_kpoints)
    
    # Rescale the tick positions so they match our new clean linspace axis
    # (Since the relative distances between high-symmetry points remain constant)
    scale_factor = original_x_coords[-1] if original_x_coords[-1] != 0 else 1
    # --------------

    # Plotting the phonon dispersion
    plt.figure(figsize=(8, 6))
    
    # This will now safely map (45,) against (45, 6)
    plt.plot(x_coords, freqs, color='b', lw=1.5)
    
    # Add vertical lines at high-symmetry points
    for tick in x_ticks:
        plt.axvline(tick, color='gray', linestyle='--', alpha=0.7)
    
    # Format labels (replace 'G' with LaTeX Gamma symbol)
    formatted_labels = [r'$\Gamma$' if l == 'G' else l for l in x_labels]
    
    plt.xticks(x_ticks, formatted_labels)
    plt.xlim(x_coords[0], x_coords[-1])
    plt.ylim(bottom=0) 
    
    plt.xlabel("Wave Vector (k)")
    plt.ylabel("Frequency")
    plt.title(r"Phonon Dispersion ($\epsilon$={:.3f}, $\theta$={:.2f}rad, rot={:.1f}$^\circ$)".format(
        args.amplitude, args.angle, args.rot))
    plt.grid(axis='y', alpha=0.3)
    
    output_plot = Path(args.freqs).with_suffix('.png')
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()