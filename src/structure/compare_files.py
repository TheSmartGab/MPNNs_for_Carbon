#!/usr/bin/env python3
import os
import sys
import json
import yaml
import math
import numpy as np
from argparse import ArgumentParser
import json

DIRNAME = os.path.abspath(os.path.dirname(__file__))
DIRS = DIRNAME.split("/")
SRC = "/".join(DIRS[:-1])
sys.path.append(SRC)

from tools.plot_metric_vs_hparams import plot_all, DEFAULT_HPARAMS, DEFAULT_HPARAMS_UDM

CONFIG_NAME = "config.yaml"  # config file containing hyperparameters


def compute_grid(n):
    """
    Automatically compute a reasonable nrows/ncols for n plots
    """
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def main():
    parser = ArgumentParser(
        "Collect metrics and plot them vs hyperparameters."
    )
    parser.add_argument("--root", required=True, help="Root directory to search runs")
    parser.add_argument(
        "--relative_path",
        required=True,
        help="Relative path to JSON metrics file within each run",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to store plots"
    )
    parser.add_argument(
        "--metric_keys",
        required=False,
        nargs="+",
        help="List of metric keys to plot; if omitted, all metrics in the first JSON are used",
    )
    parser.add_argument(
        "--metrics_udm",
        required=False,
        type=json.loads,
        help='JSON dict with units for metrics, e.g. \'{"a1": "\\\\AA"}\'',
    )
    parser.add_argument(
        "--par_color",
        required=False, 
        type=str,
        help = "parameter which determines the colormap in all the plots"
    )

    parser.add_argument(
        "--hlines_args",
        default=None,
        required=False,
        type=json.loads, # this will load a string as dict
        help="draw an horizontal line with given args. use as --hlines_args '{\"y\" : 2.443350, \"ls\":\"--\", \"label\"=\"DFT value\"}' "
    )

    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print("[ERROR] root is not a directory")
        exit(-1)

    # collect metrics and hyperparameters
    metrics_dict = {}
    hparams_dict = {}
    first_json = True
    n_runs = 0

    for root, dirs, files in os.walk(args.root):
        if CONFIG_NAME in files:
            data_path = os.path.join(root, args.relative_path)
            if not os.path.exists(data_path):
                print(f"[WARNING] Metrics file not found: {data_path}, skipping")
                continue

            with open(data_path, "r") as f:
                data = json.load(f)

            if first_json:
                # determine metric keys
                if args.metric_keys:
                    keys = args.metric_keys
                else:
                    keys = list(data.keys())
                metrics_dict = {k: [] for k in keys}
                first_json = False

            for k in metrics_dict.keys():
                metrics_dict[k].append(data.get(k, np.nan))

            # load config for hyperparameters
            config_path = os.path.join(root, CONFIG_NAME)
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            # flatten model hyperparameters
            model_hparams = cfg.get("training_module", {}).get("model", {})

            for hp in DEFAULT_HPARAMS:
                val = model_hparams.get(hp, None)
                if val is not None:
                    if hp not in hparams_dict:
                        hparams_dict[hp] = []
                    # wrap scalars in list to match dict[str, list] signature
                    hparams_dict[hp].append(val)
                else:
                    # will skip missing keys later
                    pass

            n_runs += 1

    if n_runs == 0:
        print("[ERROR] No runs found with metrics.")
        exit(-1)

    print(f"Collected {n_runs} runs.")
    print(f"Found {len(metrics_dict)} metrics and {len(hparams_dict)} hyperparameters.")

    # compute grid
    n_metrics = len(metrics_dict)
    nrows, ncols = compute_grid(n_metrics)
    print(f"Using subplot grid: nrows={nrows}, ncols={ncols}")

    # prepare units for filtered hyperparameters
    filtered_udm = []
    filtered_keys = []
    filtered_hparams = {}
    for i, hp in enumerate(DEFAULT_HPARAMS):
        if hp in hparams_dict:
            filtered_hparams[hp] = hparams_dict[hp]
            filtered_keys.append(hp)
            filtered_udm.append(DEFAULT_HPARAMS_UDM[i])
        else:
            print(f"[WARNING] hyperparameter {hp} not found in model, skipping.")

    # finally, plot
    plot_all(
        output_dir=args.output_dir,
        metrics_udm=args.metrics_udm or {},
        metrics=metrics_dict,
        hparams=filtered_hparams,
        nrows=nrows,
        ncols=ncols,
        colormap="viridis",
        par_color = args.par_color,
        hparams_keys=filtered_keys,
        hparams_udm=filtered_udm,
        hlines_args=args.hlines_args
    )

    print("[OK] Plots saved.")


if __name__ == "__main__":
    main()
