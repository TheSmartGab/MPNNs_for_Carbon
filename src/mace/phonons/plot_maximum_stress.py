from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--save", default="instability_boundary.pdf")
    parser.add_argument("--title", type=str, default="Instability Boundary")
    # Updated default label to match your requested formula
    parser.add_argument("--clabel", type=str, default=r"$(\sigma_{xx} + \sigma_{yy}) L_z / (\sqrt{2} )$")
    parser.add_argument("--xlim", type=float, default=0.30)
    parser.add_argument("--ylim", type=float, default=0.30)
    parser.add_argument("--markersize", type=float, default=50) # Increased for visibility
    return parser.parse_args()

def main():
    args = parse_args()

    if len(args.results) != len(args.labels):
        raise ValueError(f"--results and --labels must match.")

    dfs = []
    for r in args.results:
        df = pd.read_csv(r)
        
        df["exx_plot"] = df["max_exx"]
        df["eyy_plot"] = df["max_eyy"]
        
        # 2. Custom Coloring Logic
        # Formula: (sxx + syy) / (sqrt(2) * cell_a3z)
        # Note: cell_a3z is used as Lz
        df["color_metric"] = (df["sxx"] + df["syy"]) * df["cell_a3z"] / (np.sqrt(2)) 
        #df["color_metric"] = (df["sxx"] + df["syy"]) / (np.sqrt(2)) 

        
        dfs.append(df)

    # Global color scaling based on the new metric
    vmin = min(df["color_metric"].min() for df in dfs)
    vmax = max(df["color_metric"].max() for df in dfs)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["--", ":", "-.", "-"]

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = None
    for i, (df, label) in enumerate(zip(dfs, args.labels)):
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]

        # Scatter plot for data points
        sc = ax.scatter(
            df["exx_plot"],
            df["eyy_plot"],
            c=df["color_metric"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=args.markersize,
            marker=marker,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
            label=label,
        )

        # Plot line connecting the points
        df_sorted = df.sort_values("angle")
        ax.plot(
            df_sorted["exx_plot"],
            df_sorted["eyy_plot"],
            linestyle=ls,
            linewidth=1.5,
            color="black",
            alpha=0.5,
            zorder=2,
        )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(args.clabel, fontsize=11)
    
    ax.legend(loc="upper right", framealpha=0.9, fontsize=14)

    ax.set_xlabel(r"Strain $\varepsilon_{xx}$", fontsize=14)
    ax.set_ylabel(r"Strain $\varepsilon_{yy}$", fontsize=14)
    # ax.set_title(args.title, fontsize=14, pad=15)
    
    ax.set_xlim(0, args.xlim)
    ax.set_ylim(0, args.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.save)
    print(f"Success! Plot saved to {args.save}")

if __name__ == "__main__":
    main()