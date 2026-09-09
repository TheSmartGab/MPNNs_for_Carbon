# Quantum ESPRESSO Pair Scan Gridsearch Configurations (`nequip/qe_pair_gridsearch`)

This directory contains NequIP model training configurations for pair scan grid searches comparing against Quantum ESPRESSO (QE) DFT reference calculations. These configurations test various hyperparameter combinations including cutoff radii, basis functions, and training protocols using QE as the ground truth reference data for two-body interaction potentials.

## Structure:

Each version subdirectory (`v0/`, `v1/`, `v2/`, ..., `v20+`) represents a distinct NequIP model training run or hyperparameter configuration, following this consistent structure:

### Per-Version Subdirectories:
- **`2_atoms_potential/`** - Two-atom potential energy computations and comparisons using QE reference data
- **`checkpoints/`** - Model checkpoint files and trained weights at various training epochs
- **`config.yaml`** - Version-specific configuration file specifying hyperparameters, architecture settings, and dataset paths
- **`logger/`** - Training logs, loss history, and optimization tracking data
- **`outputs/`** - Generated model outputs and prediction results

## Related QE Gridsearch Directories:

This gridsearch structure is replicated across several Quantum ESPRESSO-related configurations:
- `qe_pair_gridsearch/` - Standard QE pair scan gridsearch
- `qe_pair_183_gridsearch/` - Variant with specific 183 configuration or dataset version
- `qe_pair_scan_gridsearch/`, `qe_pair_scan_gridsearch_offset/` - Extended scan configurations with energy offset corrections

## Usage:

These QE gridsearch configurations are used to:
1. Systematically explore hyperparameter space for optimal NequIP model architecture using Quantum ESPRESSO as DFT reference
2. Compare performance across different pair interaction potentials and cutoff radius settings
3. Validate ML potential accuracy against QE ground truth calculations for carbon, graphene, and related 2D materials

Generated performance plots from these QE gridsearch runs are stored in the `plots/` directory under `qe_pair_scan/`, `qe_pair_scan_offset/`, and `qe_183_models/`.