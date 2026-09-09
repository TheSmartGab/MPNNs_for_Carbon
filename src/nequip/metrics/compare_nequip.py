import os, argparse, numpy as np, yaml, json, matplotlib.pyplot as plt, matplotlib.colors as colors
from pprint import pprint
import re
import pandas as pd  # Added for reliable CSV parsing

DEFAULT_METRICS = [
    "forces_mae",
    "forces_rmse",
    "per_atom_energy_mae",
    "per_atom_energy_rmse",
]

DEFAULT_METRICS_UDM = [
    r"$\mathrm{eV/\AA}$",
    r"$\mathrm{eV/\AA}$",
    "eV",
    "eV"
]

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

DEFAULT_HPARAMS_UDM = [
    r'$\AA$',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    ''
]

# assume that models are stored in v* directories, extract the version number
VERSION_NUMBERS = []

def extract_version_number(path):
    digits = re.findall(r"v(\d+)", path)
    return int(digits[-1]) if digits else -1

METRICS = {}
HPARAMS = {}
BEST_VERSIONS = {}

RESULTS_NAME = "test_results.json"
HPARAMS_NAME = "hparams.yaml"
SKIP_FILE = "SKIP" # put a file called SKIP in a directory to avoid processing it

# Plotting
from scipy.stats import gaussian_kde

def violin_density_scatter(
    ax,
    xvals,
    yvals,
    color_vals=None,
    cmap=None,
    norm=None,
    max_width=4,
    alpha=0.3,
    s=40,
):
    """
    Violin-like scatter:
    - points spread horizontally according to y-density
    - coloring comes ONLY from color_vals (par_color)
    """
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)

    uniq_x = np.unique(xvals)

    for x in uniq_x:
        mask = xvals == x
        ys = yvals[mask]

        if len(ys) == 0:
            continue

        # central spine
        ax.vlines(
            x,
            ymin=np.min(ys),
            ymax=np.max(ys),
            color="black",
            alpha=0.25,
            linewidth=1,
            zorder=1,
        )

        if len(ys) == 1:
            x_offsets = np.array([0.0])
        else:
            kde = gaussian_kde(ys)
            dens = kde(ys)
            dens = dens / dens.max() * max_width

            # symmetric offsets
            signs = np.random.choice([-1, 1], size=len(ys))
            x_offsets = signs * dens * np.random.rand(len(ys))

        ax.scatter(
            x + x_offsets,
            ys,
            c=None if color_vals is None else color_vals[mask],
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            s=s,
            zorder=2,
        )


def plot_all(output_dir, metrics_udm, hparams_udm, nrows=2, ncols=2,
             colormap=None, par_color=None):

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "figure.titlesize": 18,
        }
    )

    metrics_list = list(METRICS.keys())
    hparams_list = list(HPARAMS.keys())

    nrows = int(nrows)
    ncols = int(ncols)

    for hp, hp_udm in zip(hparams_list, hparams_udm):
        pdf_path = os.path.join(output_dir, f"{hp}.pdf")
        print("saving", pdf_path)

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5 * ncols, 4 * nrows),
            constrained_layout=True,
        )

        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        xvals = np.array(HPARAMS[hp], dtype=object)

        # =========================================================================
        # COLORING (STRICTLY FROM par_color)
        # =========================================================================
        use_color = False
        color_vals = None
        numeric_color = False
        category_map = None
        cmap = None
        norm = None
        nuniq = 0

        if par_color:
            raw_vals = HPARAMS.get(par_color)

            if raw_vals is not None:
                raw_arr = np.asarray(raw_vals, dtype=object)

                def is_intlike(x):
                    try:
                        return float(x).is_integer()
                    except Exception:
                        return False

                if all(is_intlike(v) for v in raw_arr):
                    # categorical
                    labels = np.array([str(v) for v in raw_arr], dtype=object)
                    uniques = list(dict.fromkeys(labels))
                    category_map = {u: i for i, u in enumerate(uniques)}
                    nuniq = len(uniques)

                    color_vals = np.array([category_map[v] for v in labels], dtype=int)
                    cmap = plt.get_cmap(colormap or "tab10")

                    use_color = True
                    numeric_color = False

                else:
                    # numeric
                    color_vals = np.array(raw_arr, dtype=float)
                    finite = color_vals[np.isfinite(color_vals)]

                    vmin, vmax = (0, 1) if finite.size == 0 else (finite.min(), finite.max())
                    if vmin == vmax:
                        vmin -= 0.5
                        vmax += 0.5

                    cmap = plt.get_cmap(colormap or "viridis")
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)

                    use_color = True
                    numeric_color = True

        # =========================================================================
        # PLOTTING
        # =========================================================================
        for ax, metric, m_udm in zip(axs, metrics_list, metrics_udm):
            yvals = np.array(METRICS[metric], dtype=float)

            # If pre-calculated (like via --csv_val), use it. Otherwise compute default minimum.
            if metric in BEST_VERSIONS:
                best = BEST_VERSIONS[metric]
            else:
                idx = np.argmin(yvals)
                best = {
                    "index": idx,
                    "value": yvals[idx],
                    "version": VERSION_NUMBERS[idx],
                }
                BEST_VERSIONS[metric] = best

            try:
                max_width = np.min(xvals)/2.
            except Exception:
                max_width = 0.4

            # violin-like scatter
            violin_density_scatter(
                ax,
                xvals,
                yvals,
                color_vals=color_vals if use_color else None,
                cmap=cmap,
                norm=norm,
                alpha=0.35,
                max_width=max_width
            )

            # best point cross (Now guarantees alignment with whichever selection route chosen)
            # ax.scatter(
            #     xvals[best["index"]],
            #     best["value"],
            #     color="red",
            #     marker="x",
            #     s=90,
            #     zorder=5,
            #     label=f"best version {best['version']}",
            # )

            # legends / colorbars
            if use_color:
                if numeric_color:
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                    cbar = fig.colorbar(sm, ax=ax)
                    cbar.set_label(par_color)
                else:
                    handles, labels = [], []
                    denom = max(1, nuniq - 1)
                    for cat, idx in category_map.items():
                        rgba = cmap(idx / denom)
                        handles.append(
                            plt.Line2D([], [], marker="o", linestyle="None",
                                       markerfacecolor=rgba, markeredgecolor="k")
                        )
                        labels.append(cat)

                    ax.legend(handles, labels, title=par_color,
                              bbox_to_anchor=(1.05, 1), loc="upper left")

            ax.set_title(metric)
            ax.set_xlabel(f"{hp}[{hp_udm}]" if hp_udm else hp)
            ax.set_ylabel(f"{metric}[{m_udm}]" if m_udm else metric)

        for ax in axs[len(metrics_list):]:
            ax.axis("off")

        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)

    print(f"All hparam PDFs saved to: {output_dir}")


# Main
def main():
    parser = argparse.ArgumentParser(
        "Compare metrics on the test set (test_results.json) for different models."
    )

    parser.add_argument("root", help="root directory containing model subdirectories")
    parser.add_argument("--metrics", nargs="*", required=False, default=DEFAULT_METRICS, help="metrics to compare")
    parser.add_argument("--hparams", nargs="*", required=False, default=DEFAULT_HPARAMS, help="hyperparameters to compare")
    parser.add_argument("--out", required=False, default="./plots/", help="output directory for PDF figures")
    parser.add_argument("--nrows", required=False, default=2, type=int, help="nrows in plot.")
    parser.add_argument("--ncols", required=False, default=2, type=int, help="ncols in plot.")
    parser.add_argument("--metrics_udm", required=False, default=DEFAULT_METRICS_UDM, nargs="*")
    parser.add_argument("--hparams_udm", required=False, default=DEFAULT_HPARAMS_UDM, nargs="*")
    parser.add_argument("--par", required=False, help="hyper parameter required", default=None)
    parser.add_argument("--par_val", required=False, help="value require of parameter par", default=None)
    parser.add_argument("--par_color", required=False, help="Color differently data based on this parameters", default=None)
    parser.add_argument("--cmap", required=False, help="colormap name to use", default=None)
    
    # Validation flag
    parser.add_argument(
        "-csv_val", "--csv_val",
        action="store_true",
        help="Instruct script to look for the best RMSE on the validation set during training via metrics.csv"
    )

    args = parser.parse_args()

    # sanity check
    if args.par and not args.par_val:
        print("[ERROR] --par was passed but not par_val. Must provide parameter value, exit")
        exit(-1)
    if args.par_val and not args.par:
        print("[ERROR] --par_val was passed but not par. Must provide parameter, exit")
        exit(-1)
    if args.par_color and args.par_color not in args.hparams:
        print("[ERROR]: --par_color passed is not in the hparams. exit")
        exit(-1)

    global METRICS, HPARAMS, VERSION_NUMBERS, BEST_VERSIONS
    METRICS = {key: [] for key in args.metrics}
    BEST_VERSIONS = {}
    HPARAMS = {key: [] for key in args.hparams}

    walk = os.walk(args.root)
    csv_best_val_trackers = {key: [] for key in args.metrics}

    for root, dirs, files in walk:
        if not dirs:
            if RESULTS_NAME in files and HPARAMS_NAME in files:
                if SKIP_FILE in files:
                    continue
                results_file = os.path.join(root, RESULTS_NAME)
                hparams_file = os.path.join(root, HPARAMS_NAME)

                print("=" * 120)
                print(f"results and hyper parameters found in {root}")

                with open(results_file, "r") as fr, open(hparams_file) as fh:
                    results = json.load(fr)
                    hparams = yaml.load(fh, Loader=yaml.SafeLoader)

                    if args.par:
                        if not str(hparams.get("model", {}).get(args.par, None)) == str(args.par_val):
                            continue

                    # Process CSV validation minimums if the flag is enabled
                    if args.csv_val:
                        v_num = extract_version_number(results_file)
                        csv_path = os.path.join(root, "logger", f"version_{v_num}", "metrics.csv")
                        if not os.path.exists(csv_path):
                            csv_path = os.path.join(root, "metrics.csv")

                        if os.path.exists(csv_path):
                            try:
                                df = pd.read_csv(csv_path)
                                for m in args.metrics:
                                    val_col_base = "forces_rmse" if "forces" in m else "per_atom_energy_rmse"
                                    val_col = f"val0_epoch/{val_col_base}"
                                    
                                    if val_col in df.columns:
                                        min_val = df[val_col].min()
                                        csv_best_val_trackers[m].append(min_val)
                                    else:
                                        csv_best_val_trackers[m].append(float('inf'))
                            except Exception as e:
                                print(f"[WARNING] Failed parsing CSV {csv_path}: {e}")
                                for m in args.metrics: csv_best_val_trackers[m].append(float('inf'))
                        else:
                            for m in args.metrics: csv_best_val_trackers[m].append(float('inf'))

                    VERSION_NUMBERS.append(extract_version_number(results_file))
                    for m in METRICS.keys():
                        METRICS[m].append(results.get(m, None))
                    for h in HPARAMS.keys():
                        HPARAMS[h].append(hparams.get("model", {}).get(h, None))

    print("[INFO] found", len(HPARAMS[list(HPARAMS.keys())[0]]), "models")

    # Step 1: Pre-populate BEST_VERSIONS with integer indices *before* running plots
    if args.csv_val:
        for metric in args.metrics:
            vals = csv_best_val_trackers[metric]
            if vals and min(vals) != float('inf'):
                idx = np.argmin(vals)
                BEST_VERSIONS[metric] = {
                    "index": idx,
                    "value": METRICS[metric][idx], 
                    "version": VERSION_NUMBERS[idx],
                    "validation_best_rmse": vals[idx]
                }

    # Step 2: Draw plots while indices in BEST_VERSIONS are still integers
    plot_all(args.out, args.metrics_udm, args.hparams_udm, args.nrows, args.ncols, args.cmap, args.par_color)

    # solves int64 error when saving to json
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # Step 3: Now transform the index parameter into a tracking dictionary safely for info.json output
    for metric in list(METRICS.keys()):
        if metric in BEST_VERSIONS:
            idx = BEST_VERSIONS[metric]["index"]
            hparams_summary = {}
            for hp in list(HPARAMS.keys()):
                hparams_summary[hp] = HPARAMS[hp][idx]
            
            all_metrics = {}
            for all_m in METRICS.keys():
                all_metrics[all_m] = METRICS[all_m][idx]
                
            BEST_VERSIONS[metric]["index"] = hparams_summary
            BEST_VERSIONS[metric]["all_metrics"] = all_metrics

    out_info = os.path.join(args.out, "info.json")
    with open(out_info, "w") as f:
        json.dump(BEST_VERSIONS, f, indent=4, cls=NpEncoder)

    return 0

if __name__ == "__main__":
    main()