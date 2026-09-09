import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib.ticker import AutoMinorLocator


DEFAULT_HPARAMS = [
    "r_max",
    "num_bessels",
    "bessel_trainable",
    "polynomial_cutoff_p",
    "num_layers",
    "l_max",
    "parity",
    "num_features",
    "radial_mlp_depth",
    "radial_mlp_width",
]

DEFAULT_HPARAMS_UDM = [r"$\AA$", "", "", "", "", "", "", "", "", ""]


def plot_all(
    output_dir: str,
    metrics_udm: dict[str, str],
    metrics: dict[str, list[float]],
    hparams: dict[str, list],
    nrows: int,
    ncols: int,
    colormap=None,
    par_color: str = "",
    hparams_keys: list[str] = DEFAULT_HPARAMS,
    hparams_udm: list[str] = DEFAULT_HPARAMS_UDM,
    hlines_args: dict = {},
):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })

    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)

    # Prepare par_color values
    use_color = False
    color_vals = None
    cmap = None
    norm = None
    category_map = None
    numeric_color = False

    if par_color and par_color in hparams:
        raw_col = np.array(hparams[par_color], dtype=object)

        def is_intlike(x):
            try:
                f = float(x)
                return f.is_integer()
            except:
                return False

        all_intlike = all(is_intlike(v) for v in raw_col)

        if all_intlike:
            # Categorical
            labels_cat = [str(v) for v in raw_col]
            uniques = sorted(set(labels_cat))
            category_map = {u: i for i, u in enumerate(uniques)}
            mapped_vals = np.array([category_map[v] for v in labels_cat], dtype=int)
            cmap = plt.get_cmap(colormap or "tab10")
            color_vals = mapped_vals
            use_color = True
        else:
            # Numeric colormap
            numeric_vals = np.array([float(v) for v in raw_col])
            finite = numeric_vals[np.isfinite(numeric_vals)]
            vmin = float(np.nanmin(finite)) if finite.size > 0 else 0
            vmax = float(np.nanmax(finite)) if finite.size > 0 else 1
            if vmin == vmax:
                vmax += 1.0
            cmap = plt.get_cmap(colormap or "viridis")
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            color_vals = numeric_vals
            use_color = True
            numeric_color = True

    # Main loop per hyperparameter
    for hp, hp_udm in zip(hparams_keys, hparams_udm):
        if hp not in hparams:
            print(f"[WARNING] hyperparameter {hp} not found.")
            continue

        # Flatten x values
        xvals = []
        for v in hparams[hp]:
            if isinstance(v, (int, float, str)):
                xvals.append(v)
            elif isinstance(v, (list, tuple, np.ndarray)):
                xvals.append(str(v[0]) if len(v) > 0 else np.nan)
            else:
                xvals.append(str(v))
        xvals = np.array(xvals)

        pdf_path = os.path.join(output_dir, f"{hp}.pdf")
        print("Saving", pdf_path)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(5 * ncols, 4 * nrows),
                                constrained_layout=True)
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        else:
            axs = axs.flatten()

        for ax, metric in zip(axs, metric_names):
            yvals = np.array(metrics[metric], dtype=float)

            # Scatter points
            if use_color:
                if numeric_color:
                    sc = ax.scatter(xvals, yvals, c=color_vals, cmap=cmap, norm=norm, marker="o")
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.set_label(par_color)
                else:
                    sc = ax.scatter(xvals, yvals, c=color_vals, cmap=cmap, marker="o")
            else:
                ax.scatter(xvals, yvals, marker="o")

            # Labels
            ax.set_title(metric)
            ax.set_xlabel(f"{hp}[{hp_udm}]" if hp_udm else hp)
            ax.set_ylabel(f"{metric}[{metrics_udm.get(metric, '')}]" if metrics_udm.get(metric) else metric)

           # Build legend handles/labels
            handles, labels_list = [], []

            # Add par_color legend first
            if use_color and not numeric_color and category_map:
                nuniq = len(category_map)
                denom = max(1, nuniq - 1)
                for cat, idx in category_map.items():
                    frac = idx / denom
                    rgba = cmap(frac)
                    handle = plt.Line2D([], [], marker="o", linestyle="None",
                                        markerfacecolor=rgba, markeredgecolor="k")
                    handles.append(handle)
                    labels_list.append(cat)

            # Then add hline legend below
            if hlines_args and "label" in hlines_args:
                hline_handle = ax.axhline(**hlines_args)
                handles.append(hline_handle)
                labels_list.append(hlines_args["label"])

            # Set legend with par_color title
            if handles:
                ax.legend(handles, labels_list, title=par_color if par_color else None,
                        bbox_to_anchor=(1.05, 1), loc="upper left")

        # Hide unused axes
        for ax in axs[len(metric_names):]:
            ax.axis("off")

        # Enable major grid
        ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.7)

        # Set minor ticks: 10 subdivisions per major tick
        ax.yaxis.set_minor_locator(AutoMinorLocator(10))

        # Enable minor grid for y-axis
        ax.grid(True, which='minor', axis='y', linestyle=':', alpha=0.3)

        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)

    print(f"All hparam PDFs saved to: {output_dir}")
