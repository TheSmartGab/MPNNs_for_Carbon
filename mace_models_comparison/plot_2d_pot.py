import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import os

def parse_args():
    parser = ArgumentParser("Basic script to plot 2D function")
    
    parser.add_argument("data_file", help="Path to data file")
    parser.add_argument(
        "--columns", required=False, nargs=3, type=int, default=[0, 1, 2],
        help="Column indexes for x, y, z (0-based)"
    )
    parser.add_argument(
        "--zlabel", required=False, default=None, type=str
    )
    
    return parser.parse_args()


def plot_2d_function(data_file, columns, args):
    # Load data
    data = np.loadtxt(data_file)
    x = data[:, columns[0]]
    y = data[:, columns[1]]
    z = data[:, columns[2]]

    dir = os.path.dirname(data_file)
    # ---------- Scatter plot ----------
    plt.figure(figsize=(6,5))
    sc = plt.scatter(x, y, c=z, cmap="viridis", s=50)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=args.zlabel)
    plt.xlabel(r"x [$\AA$]")
    plt.ylabel(r"y [$\AA$]")
    plt.tight_layout()
    #plt.show()
    outfile = os.path.join(dir,"graphite_potential_scatter.svg") 
    plt.savefig(outfile)

    # ---------- Contour plot ----------
    # Create a grid for contouring
    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate z values on the grid
    from scipy.interpolate import griddata
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    plt.figure(figsize=(6,5))
    contour = plt.contourf(Xi, Yi, Zi, levels=20, cmap="viridis")
    plt.colorbar(contour, label=args.zlabel) 
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel(r"x [$\AA$]")
    plt.ylabel(r"y [$\AA$]")
    plt.tight_layout()
    #plt.show()
    outfile = os.path.join(dir, "graphite_potential_contour.svg")
    plt.savefig(outfile)


def main():
    args = parse_args()
    plot_2d_function(args.data_file, args.columns, args)


if __name__ == "__main__":
    main()
