import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("relaxed_file")
    parser.add_argument("rigid_file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    relaxed = np.loadtxt(args.relaxed_file)
    rigid = np.loadtxt(args.rigid_file)

    # Visual Enhancement: Using a clean style
    plt.style.use('seaborn-v0_8-muted') 
    fig, ax = plt.subplots(figsize=(8, 6))

    print("plotting")
    
    # Use markers + lines (optional) for better clarity
    ax.plot(relaxed[:,0], relaxed[:,1], 'o', color="red", label="Relaxed", markersize=4, alpha=0.8)
    ax.plot(rigid[:,0], rigid[:,1], 'x', color="blue", label="Rigid", markersize=4, alpha=0.8)

    ax.set_xlabel(r"COM distance [$\AA$]", fontsize=14)
    ax.set_ylabel(r"Energy [eV]", fontsize=14)
    
    ax.set_xlim((7.5, 14))

    # Calculate minimum for the inset
    mini = np.argmin(rigid[:,1])
    mind = rigid[mini,0]
    mine = rigid[mini,1]

    # Create Inset - [left, bottom, width, height] as fractions of parent axes
    ax_in = ax.inset_axes([0.35, 0.35, 0.55, 0.55]) 
    ax_in.plot(relaxed[:,0], relaxed[:,1], 'o', color="red", markersize=3)
    ax_in.plot(rigid[:,0], rigid[:,1], 'x', color="blue", markersize=3)

    # Set zoom limits around the minimum
    ax_in.set_xlim(mind - 1.5, mind + 3)
    ax_in.set_ylim(mine - 0.01, mine + 0.5)
    ax_in.grid(True, linestyle='--', alpha=0.6)

    ax_in.set_xlabel(r"COM distance [$\AA$]", fontsize=14)
    ax_in.set_ylabel(r"Energy [eV]", fontsize=14)
    # Add box to parent and lines connecting to inset
    #ax.indicate_inset_zoom(ax_in, edgecolor="black")

    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig("fullerene_comparison.svg")
    print("Done. Saved to fullerene_comparison.svg")
    return 0

if __name__ == "__main__":
    main()
