"""merge data from different .json files generated with inspect_extxyz_categories.py"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description=(
            "Merge data from different .json files generated with "
            "inspect_extxyz_categories.py. Useful to compare dataset splits "
            "or different datasets with same config_types."
        )
    )

    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        required=True,
        help=".json files with information.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        required=True,
        help="Labels of the files, e.g. train val test.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where histograms are saved.",
    )

    return parser.parse_args()


def load_data(files, labels):
    if len(files) != len(labels):
        raise ValueError("Number of files and labels must match.")

    data = {}
    for label, file in zip(labels, files):
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} not found.")

        with open(file) as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                content = {}

        # Ensure empty files become empty dict
        if content is None:
            content = {}

        data[label] = content

    return data


def parse_config_types(data):
    """
    Collect all unique config_types across datasets.
    Handles empty dictionaries safely.
    """
    ct = set()

    for dataset in data.values():
        if not dataset:
            continue
        for key in dataset.keys():
            ct.add(key)

    return sorted(list(ct))


def build_matrix(config_types, data, key="count"):
    labels = list(data.keys())
    matrix = np.zeros((len(labels), len(config_types)))

    for i, label in enumerate(labels):
        dataset = data[label]

        for j, ct in enumerate(config_types):
            if ct in dataset and isinstance(dataset[ct], dict):
                matrix[i, j] = dataset[ct].get(key, 0)
            else:
                matrix[i, j] = 0

    return labels, matrix



def plot(config_types, labels, matrix, output_file=None):
    x = np.arange(len(config_types))
    width = 0.8 / len(labels)

    plt.figure(figsize=(12, 6))

    for i, label in enumerate(labels):
        plt.bar(x + i * width, matrix[i], width, label=label)

    plt.xticks(x + width * (len(labels) - 1) / 2, config_types, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.xlabel("Config Types")
    plt.legend()
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def main():
    args = parse_args()

    data = load_data(args.files, args.labels)

    config_types = parse_config_types(data)

    if not config_types:
        print("No config_types found in provided files.")
        return 0

    keys = ["count", "local_env_count"]
    for key in keys:
        labels, matrix = build_matrix(config_types, data, key)

        output_file = os.path.join(args.output_dir, key + ".pdf")
        plot(config_types, labels, matrix, output_file)

    return 0


if __name__ == "__main__":
    main()
