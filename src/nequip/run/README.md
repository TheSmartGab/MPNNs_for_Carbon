# NequIP Run Directories (`src/nequip/run`)

This directory contains shell scripts for running NequIP (Neural Equivariant Interatomic Potentials) evaluations across multiple configuration directories. These utilities automate the process of evaluating trained NequIP models on sets of atomic configurations stored in various subdirectories.

## Contents:

### `run_dirs.sh`
Batch evaluation script that searches a target root directory for NequIP configuration files (`config.yaml`) and runs `nequip-train -cn config.yaml` across all found directories. 

**Key functionality:**
- Recursively searches a given directory tree for `config.yaml` files
- Activates the appropriate conda environment (via `SetConda.sh`)
- Runs NequIP evaluation or training commands in each configuration directory
- Provides status output for each processed directory

## Usage:

This script is used to:
1. Evaluate trained NequIP models across multiple test configurations simultaneously
2. Run batch evaluations of model performance on different structural datasets
3. Automate the process of computing energies, forces, and stresses for sets of atomic structures using pre-trained NequIP models

The output from these runs feeds into the metrics comparison scripts in `src/nequip/metrics/` for visualizing model performance against DFT or MACE reference data.
