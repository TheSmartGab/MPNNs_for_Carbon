# =====================================
# Script Name: generate_configs.py
# Purpose: Generate atomic configurations or simulation setups programmatically
# Usage: Run with YAML/JSON configuration files to create test structures, initial configurations
#        for MD simulations, or grid scans of structural parameters.
# =====================================

"""Configuration generation utility module.

This script generates atomic configurations or simulation setups programmatically using YAML or JSON
configuration files. It is useful for creating test structures, initial configurations for molecular
dynamics simulations, or grid scans of structural parameters for machine learning interatomic potential
training and validation workflows.
"""

import os
import sys
from argparse import ArgumentParser
import yaml
import json
import itertools
from copy import deepcopy


def load_maybe_json_yaml(path):
    """Load YAML or JSON automatically."""
    with open(path, "r") as f:
        text = f.read()

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse {path} as YAML or JSON.")


def set_nested(config, key, value):
    """
    Set a nested key like 'training_module.model.num_layers' inside the config.
    """
    parts = key.split(".")
    c = config
    for p in parts[:-1]:
        if p not in c:
            raise KeyError(f"Key '{key}': missing intermediate '{p}'.")
        c = c[p]
    c[parts[-1]] = value


def main():

    parser = ArgumentParser(
        "Generate config files from a template by performing grid search on given hyperparameters."
    )

    parser.add_argument(
        "--template",
        required=True,
        help="Path to template config.yaml. Other configs will be created next to it."
    )
    parser.add_argument(
        "--hparams_dict",
        required=True,
        help="YAML/JSON file of form: {param.name: [values...]}"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to create generated configs. Default: directory of template."
    )

    args = parser.parse_args()

    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Template file '{args.template}' not found.")
    if not os.path.exists(args.hparams_dict):
        raise FileNotFoundError(f"hparams_dict file '{args.hparams_dict}' not found.")

    # Load template config
    with open(args.template, "r") as f:
        template_cfg = yaml.safe_load(f)

    # Load hyperparameter grid
    hparams = load_maybe_json_yaml(args.hparams_dict)

    # Where to write results
    base_dir = (
        os.path.dirname(os.path.abspath(args.template))
        if args.output_dir is None
        else args.output_dir
    )
    os.makedirs(base_dir, exist_ok=True)

    # Build cartesian product of hyperparameters
    keys = list(hparams.keys())
    values = list(hparams.values())
    grid = itertools.product(*values)

    print(f"[INFO] Generating configurations in: {base_dir}/")

    counter=0
    for combo in grid:
        # Example directory name: config_l_max=0_num_layers=2_r_max=4.0
        name_parts = [f"{k}={v}" for k, v in zip(keys, combo)]
        run_dir_name = f"v{counter}"
        run_dir_path = os.path.join(base_dir, run_dir_name)
        os.makedirs(run_dir_path, exist_ok=True)

        new_cfg = deepcopy(template_cfg)

        # Set hyperparameter values
        for k, v in zip(keys, combo):
            set_nested(new_cfg, k, v)

        # Update trainer checkpoint path
        if "trainer" in new_cfg and isinstance(new_cfg["trainer"], dict):
            if "callbacks" in new_cfg["trainer"]:
                for cb in new_cfg["trainer"]["callbacks"]:
                    if isinstance(cb, dict) and cb.get("_target_") == "lightning.pytorch.callbacks.ModelCheckpoint":
                        cb["dirpath"] = os.path.join(run_dir_path, "checkpoints")

        # Write the YAML config
        output_config_path = os.path.join(run_dir_path, "config.yaml")
        with open(output_config_path, "w") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False)

        print(f"Wrote: {output_config_path}")
        counter+=1

    print("[DONE] All configs generated.")


if __name__ == "__main__":
    main()
