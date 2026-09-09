# NequIP (Neural Equivariant Interatomic Potentials) Workflows (`nequip`)

This directory contains workflows for evaluating, analyzing, and comparing NequIP machine learning interatomic potentials.

## Subdirectories:

### [`2_atoms_potential/`](./2_atoms_potential/)
Two-atom potential computations and comparisons:
- **`compute_pot.py`**: Compute two-atom potentials using NequIP models
- **`plot_comparison.py`**: Plot comparison between NequIP and reference (DFT/MACE) potential energies
- **`plot_merge_errors.py`**, **`plot_merge.py`**: Visualization utilities for merged model comparisons and error analysis
- **`plot.py`**: General plotting utilities for two-atom potentials

### [`automatic_phonon_dispersion/`](./automatic_phonon_dispersion/)
Automated phonon dispersion calculations:
- **`plot_band_vel.py`**: Plot band velocity from NequIP phonon dispersion data

### [`metrics/`](./metrics/)
NequIP metric evaluation and visualization:
- **`compare_nequip.py`**: Compare NequIP model predictions against reference data
- **`plot_dir_nequip.py`**, **`plot_nequip.py`**: Plot NequIP metrics across directories or runs
- **`results.py`**: Results processing and aggregation for NequIP evaluations

### [`eval_configs.py`](../nequip/eval_configs.py)
Evaluate configurations using NequIP models. This script is typically used to compute energies, forces, and stresses for a set of atomic structures using a trained NequIP model.

## Usage

The NequIP directory contains tools for evaluating pre-trained neural equivariant interatomic potentials, comparing their predictions against DFT or MACE references, and visualizing performance metrics across different structural configurations or dataset splits.