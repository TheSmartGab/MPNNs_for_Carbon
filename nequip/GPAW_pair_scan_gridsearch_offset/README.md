# GPAW Pair Scan Gridsearch Offset Configurations (`nequip/GPAW_pair_scan_gridsearch_offset`)

This directory contains NequIP model training configurations for pair scan grid searches comparing against GPAW (GBA-Wave Package) reference calculations with energy offset corrections. These configurations test various hyperparameter combinations including cutoff radii, basis functions, and ZBL (Ziegler-Biersack-Littmark) potential inclusion/exclusion variants.

## Structure:

Each version subdirectory (`v0/`, `v1/`, `v2/`, ..., `v119+`) represents a distinct NequIP model training run or hyperparameter configuration, following this consistent structure:

### Per-Version Subdirectories:
- **`2_atoms_potential/`** - Two-atom potential energy computations and comparisons for the specific version
- **`checkpoints/`** - Model checkpoint files and trained weights at various training epochs
- **`config.yaml`** - Version-specific configuration file specifying hyperparameters, architecture settings, and dataset paths
- **`logger/`** - Training logs, loss history, and optimization tracking data
- **`outputs/`** - Generated model outputs and prediction results

## Variants:

This directory structure is replicated across several related configurations:
- `GPAW_pair_scan_gridsearch_offset/` - Standard GPAW pair scan with energy offsets
- `GPAW_pair_scan_gridsearch_offset_NOZBL/` - Same configuration without ZBL (Ziegler-Biersack-Littmark) potential
- `GPAW_pair_scan_gridsearch_offset_rcut_10/`, `GPAW_pair_scan_gridsearch_offset_rcut_10_NOZBL/` - Alternative cutoff radius (r_max=10) experiments
- `GPAW_pair_scan_gridsearch_offset_v78/` - Version 78 specific configuration or reference implementation

## Usage:

These gridsearch configurations are used to:
1. Systematically explore hyperparameter space for optimal NequIP model architecture and training protocols
2. Compare performance across different energy offset correction methodologies
3. Validate ML potential accuracy against GPAW DFT reference calculations for two-body interaction potentials

Generated performance plots and benchmark visualizations from these gridsearch runs are stored in the `plots/` directory under `nequip_GPAW_C2_f_cutoff_100/`, `NO_ZBL_nequip_C2_gridsearch/`, and related subdirectories. Merged comparison results across multiple versions are available in the `merged/` directory.