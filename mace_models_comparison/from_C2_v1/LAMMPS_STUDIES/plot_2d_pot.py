import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser("Basic script to plot 2D function")
    
    parser.add_argument("data_file", help="Path to data file")
    parser.add_argument(
        "--columns", required=False, nargs=3, type=int, default=[0, 1, 2],
        help="Column indexes for x, y, z (0-based)"
    )
    
    return parser.parse_args()


def plot_2d_function(data_file, columns):
    # Load data
    data = np.loadtxt(data_file)
    x = data[:, columns[0]]
    y = data[:, columns[1]]
    z = data[:, columns[2]]

    # ---------- Scatter plot ----------
    plt.figure(figsize=(6,5))
    sc = plt.scatter(x, y, c=z, cmap="viridis", s=50)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label="z")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter plot colored by z")
    plt.tight_layout()
    #plt.show()
    plt.savefig("graphite_potential_scatter.pdf")

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
    plt.colorbar(contour, label="z") 
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour plot of z(x, y)")
    plt.tight_layout()
    #plt.show()
    plt.savefig("graphite_potential_contour.pdf")


def main():
    args = parse_args()
    plot_2d_function(args.data_file, args.columns)


if __name__ == "__main__":
    main()
