"""Inside extxyz there may be a string like config_type that specify the type of a certain configuration. It may be insightfull to count the number of such configurations, an eventually to separate them. This script was originally written to inspect the GAP2020 dataset"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase import Atoms, Atom
from pprint import pprint
import json
import os


def parse_args():

    parser = ArgumentParser("This script counts the number of configurations for each found type in the given file.")

    parser.add_argument("--input_file", type=str, required=True, help = "input file with data")
    parser.add_argument("--config_type_keyword", type=str, required=True, help = "keyword in extxyz file used to indicata config_type")
    parser.add_argument("--out_dir", type=str, help="output directory with count info, and eventually splitted files")
    parser.add_argument("--split", action="store_true", required=False, help="if given, one file for each found config_type will be created in the out_dir")

    args = parser.parse_args()
    print("="*100)
    print("[INFO] Running", __file__, "with args")
    pprint(args)
    print("="*100)

    return args

def plot_config_type_histogram(config_types, args):
    print("[INFO] Plotting config_type histogram")

    labels = list(config_types.keys())
    counts = [config_types[k]["count"] for k in labels]

    plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
    plt.bar(labels, counts)

    plt.xlabel("config_type")
    plt.ylabel("count")
    plt.title("Configuration type distribution")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_file = os.path.join(args.out_dir, "config_type_histogram.pdf")
    print("[INFO] Saving histogram to", out_file)
    plt.savefig(out_file)
    plt.close()

def split(configs, info, args):
    print("[INFO] Splitting dataset")
    for ct in info.keys():
        out_file = os.path.join(args.out_dir, ct+".extxyz")
        atoms = [configs[i] for i in info[ct]["indexes"]]
        print("[INFO] writing", info.get(ct).get("count"), ct, "configs to", out_file)
        write(out_file, atoms, format="extxyz")

def plot_config_type_histogram(config_types, key, args):
    print("[INFO] Plotting config_type histogram")

    labels = list(config_types.keys())
    counts = np.array([config_types[k][key] for k in labels])

    plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
    plt.bar(labels, counts)

    plt.yscale("log")  # <-- log scale for up to 3 orders of magnitude

    plt.xlabel("config_type")
    plt.ylabel(f"{key} (log scale)")
    plt.title("Configuration type distribution")
    plt.xticks(rotation=45, ha="right")

    # Avoid log(0) issues if someone ever adds an empty category
    plt.ylim(bottom=0.8)

    plt.tight_layout()

    out_file = os.path.join(args.out_dir, f"config_type_{key}.pdf")
    print("[INFO] Saving histogram to", out_file)
    plt.savefig(out_file)
    plt.close()



def main():

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    configs = read(args.input_file, format="extxyz", index=":")
    print("[INFO] Found", len(configs), "configs")
    config_types = {}
    # to be organised as 
    # {config_type : {count : count, indexes: [indexes], local_env_count : local_env_count}}
    

    for idx, atoms in enumerate(configs):
        ct = atoms.info.get(args.config_type_keyword)
        if ct in config_types.keys():
            config_types[ct]["count"]+=1
            config_types[ct]["indexes"].append(idx)
            config_types[ct]["local_env_count"]+=len(atoms)
        else:
            config_types[ct] = {}
            config_types[ct]["count"] = 1
            config_types[ct]["indexes"] = [idx]
            config_types[ct]["local_env_count"]=len(atoms)
            print("[INFO] found config_type", ct)

    # sort wrt counts
    config_types = dict(sorted(config_types.items(), key=lambda item: item[1]['count']))

    output_file = os.path.join(args.out_dir, "config_type_info.json")
    print("[INFO] writing info to\n\b", output_file)
    with open(output_file, "w") as f:
        json.dump(config_types, f, indent=4)
    if args.split:
        split(configs, config_types, args)

    plot_config_type_histogram(config_types,"count", args)
    plot_config_type_histogram(config_types,"local_env_count", args)

    return 0

if __name__ == "__main__":
    main()