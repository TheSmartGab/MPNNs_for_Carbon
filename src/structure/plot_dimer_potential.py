import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# Atomic numbers for ZBL calculation
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
}

def get_zbl_potential(r, z1, z2):
    """Calculates the ZBL repulsion potential in eV."""
    # Constants
    e_squared = 14.3996  # eV * Å
    a_0 = 0.529177       # Bohr radius in Å
    
    # Universal screening length
    a = 0.8854 * a_0 / (z1**0.23 + z2**0.23)
    x = r / a
    
    # Screening function phi(x)
    phi = (0.18175 * np.exp(-3.1998 * x) + 
           0.50986 * np.exp(-0.94229 * x) + 
           0.28022 * np.exp(-0.4029 * x) + 
           0.02817 * np.exp(-0.20162 * x))
    
    return (z1 * z2 * e_squared / r) * phi

def main():
    parser = argparse.ArgumentParser(description="Dimer Analysis with ZBL Repulsion")
    parser.add_argument("--potential", type=str, required=True, help="Path to .npy potential file")
    parser.add_argument("--pddf", type=str, required=True, help="Path to .txt PDDF file")
    parser.add_argument("--species", type=str, default="C", help="Chemical symbol (e.g., C, Si, H)")
    parser.add_argument("--out", type=str, default="dimer_analysis.png", help="Output plot filename")
    args = parser.parse_args()

    # Load Data
    pddf_data = np.loadtxt(args.pddf, comments='#')
    r_pddf, freq = pddf_data[:, 0], pddf_data[:, 1]
    
    pot_data = np.load(args.potential)
    r_pot, u_val = pot_data[:, 0], pot_data[:, 1]/2 # energy per atom

    # Calculate Binding Energy
    u_min = np.min(u_val)
    u_inf = u_val[-1]  # Energy at largest distance
    binding_energy = u_inf - u_min
    r_min = r_pot[np.argmin(u_val)]

    # Calculate ZBL
    z = ATOMIC_NUMBERS.get(args.species.capitalize(), 6)
    u_zbl = get_zbl_potential(r_pot, z, z)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # PDDF (Background)
    # ax1.plot(r_pddf, freq, color='gray', alpha=0.3, label='PDDF $P(r)$')
    binwidth=r_pddf[1] - r_pddf[0]
    ax1.bar(r_pddf, width = binwidth, height=freq, color='gray', alpha=0.25, align='center')
    ax1.set_xlabel('Distance $r$ (Å)', fontsize=14)
    ax1.set_ylabel('Counts', fontsize = 14)
    
    # Potential (Scatter)
    ax2 = ax1.twinx()
    ax2.scatter(r_pot, u_val, color='#1f77b4', s=20, alpha=0.6, label='Potential Energy $U(r)$ ')
    
    # ZBL (Dotted Line)
    ax2.plot(r_pot, u_zbl, 'r:', lw=2, label=f'ZBL Repulsion ({args.species}-{args.species})')
    
    # Highlight Minimum
    ax2.scatter(r_min, u_min, color='black', marker='x', s=100, linewidths=2, zorder=5, label='Global Minimum')
    
    # Formatting
    ax2.set_ylabel('Energy [eV/atom]', fontsize=12)
    ax2.set_ylim(u_inf - (binding_energy*1.1), u_inf + (binding_energy*1.1)) # Focus on the well
    
    #title = f"Dimer Analysis for {args.species}\nBinding Energy: {binding_energy:.4f} eV"
    #plt.title(title, fontsize=14, fontweight='bold')
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    print("-" * 30)
    print(f"Species:        {args.species}")
    print(f"Minimum Energy: {u_min:.6f}")
    print(f"Energy at Inf:  {u_inf:.6f}")
    print(f"Binding Energy: {binding_energy:.6f}")
    print("-" * 30)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()