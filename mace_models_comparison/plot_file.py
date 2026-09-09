#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Global font settings
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def parse_args():
    parser = argparse.ArgumentParser(description="Fully customizable CLI plotter")

    # Required argument: filepath
    parser.add_argument("filepath", help="Path to data file (columns: x y)")

    # Arbitrary plot options (can pass any matplotlib keyword)
    parser.add_argument("--xlabel", type=str, help="X-axis label")
    parser.add_argument("--ylabel", type=str, help="Y-axis label")
    parser.add_argument("--title", type=str, help="Plot title")
    parser.add_argument("--xlim", type=float, nargs=2, help="X-axis limits: min max")
    parser.add_argument("--ylim", type=float, nargs=2, help="Y-axis limits: min max")
    parser.add_argument("--color", type=str, default="blue", help="Line/scatter color")
    parser.add_argument("--marker", type=str, default="o", help="Scatter marker style")
    parser.add_argument("--linestyle", type=str, default="-", help="Line style")
    parser.add_argument("--scatter_alpha", type=float, default=0.8, help="Scatter transparency")
    parser.add_argument("--line_alpha", type=float, default=1.0, help="Line transparency")
    parser.add_argument("--save", type=str, default=None, help="File path to save figure")
    parser.add_argument("--grid", action="store_true", help="Enable grid")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    data = np.genfromtxt(args.filepath)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Data file must have at least 2 columns: x y")

    x, y = data[:,0], data[:,1]

    plt.figure(figsize=(8,5))

    # Scatter
    plt.scatter(x, y, color=args.color, marker=args.marker, alpha=args.scatter_alpha, label="data")
    # Line
    plt.plot(x, y, color=args.color, linestyle=args.linestyle, alpha=args.line_alpha)

    minidx = np.argmin(y)
    xmin = x[minidx]
    ymin = y[minidx]

    diff = y[-1] - ymin
    print("last - min:", diff)

    plt.scatter(xmin, ymin, marker="x", color="orange", label=f"Minimum ({xmin:.2f}, {ymin:.2f})")

    # Labels and title
    if args.xlabel:
        plt.xlabel(args.xlabel)
    if args.ylabel:
        plt.ylabel(args.ylabel)
    if args.title:
        plt.title(args.title)

    # Axis limits
    if args.xlim:
        plt.xlim(args.xlim)
    if args.ylim:
        plt.ylim(args.ylim)

    # Grid
    if args.grid:
        plt.grid(True)

    plt.legend()
    plt.tight_layout()

    # Save or show
    if args.save:
        plt.savefig(args.save)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

