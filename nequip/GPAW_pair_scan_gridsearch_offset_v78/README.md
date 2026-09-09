# NequIP GPAW Pair Scan Gridsearch with Offset Corrections (`nequip/GPAW_pair_scan_gridsearch_offset_v78`)

This directory contains NequIP (Neural Equivariant Interatomic Potentials) model training configurations and outputs from gridsearch experiments comparing against GPAW reference pair scan calculations with energy offset corrections. These configurations systematically explore hyperparameter spaces for carbon (C2) systems to optimize ML potential accuracy against first-principles DFT references.

## Activities Performed - GridSearch Workflow:

### Hyperparameter Exploration Activities:
The gridsearch workflow performs systematic exploration of NequIP model architectures by varying:
1. Interaction layer configurations and feature dimensions
2. Cutoff radii and radial basis function parameters
3. Energy offset correction methodologies for pair scan calculations
4. Training batch sizes and learning rate schedules

Each version subdirectory (v0 through v30+) represents a distinct combination of hyperparameters evaluated against the GPAW reference data for C2 diatomic carbon systems with force cutoff of 100 eV/Å.

## Contents - Configuration Files:

### Primary Configuration Files:
- **`config.yaml`**: Primary NequIP training configuration using Hydra/OmegaConf framework, defining data pipelines with ASE extxyz file loading, neighbor list transforms (r_max=5.0 Å), and chemical species to atom type mappers for carbon systems
- **`hparams.yaml`**: Hyperparameter configurations including model architecture settings, cutoff radius specifications, and training hyperparameters

## Activities Performed - Version Subdirectory Workflows (`v0/`, `v1/`, ..., `v30+`):

Each version subdirectory contains structured outputs from specific NequIP model training runs:

### `2_atoms_potential/` Directory Activities:
This subdirectory contains scripts and outputs for two-atom potential computation workflows:
- **`compute_over_dir.sh`**, **`compute_pot.sh`**: Shell orchestration scripts for computing potential energies across directories of C2 configurations
- **`compute_pot.py`**: Python script evaluating NequIP model predictions for two-atom potential energy surfaces, comparing against GPAW reference calculations
- **`eval_diff_over_dir.sh`**: Script for evaluating differences between NequIP predictions and DFT reference values across configuration directories
- **Visualization outputs**: `force.pdf`, `potential_energy.pdf`, `plot.py`, `plot_merge.py`, `plot_merge_errors.py` - Generated plots showing force distributions, potential energy curves, and error comparisons

### `checkpoints/` Directory Activities:
This subdirectory contains model weight files saved during training:
- **`best.ckpt`**: Optimal model checkpoint selected based on validation performance metrics (lowest RMSE/MAE for energies and forces)
- **`last.ckpt`**: Final model state at completion of the 100-epoch training run
- **`deployed_pair.nequip.pth`**: Compiled NequIP model artifact prepared for production molecular dynamics simulations via LAMMPS integration

### `logger/`, `outputs/`, and Supporting Directories:
- **`logger/`**: Training log files documenting loss progression, validation metrics, and convergence behavior across epochs
- **`outputs/`**: Final evaluation results and performance metrics computed on test sets after training completion
- **`packaged_model.nequip.zip`**: Compiled NequIP model artifacts ready for deployment in production MD simulation environments

### Input and Working Directories:
- **`INPUT/`**: Input configuration files and atomic structures used as starting points for two-atom potential computations
- **`WORKING/`**: Active working directory containing intermediate computation states and temporary files during the evaluation pipeline

## Data Sources Used in GridSearch Activities:

The configurations reference GPAW-complemented datasets from `DATASETS/C/nomad/GRAPHENE_MLIP_GPAW_COMPLEMENTS/C2_f_cutoff_100/all/split/`, containing:
- Training extxyz files with force cutoff of 100 eV/Å applied
- Validation and test extxyz splits for model evaluation and generalization testing

## Usage in MLIP Workflows:

These gridsearch configurations are used to optimize NequIP model architectures for carbon systems against GPAW DFT references. The systematic hyperparameter exploration identifies optimal architectural choices (interaction layers, cutoff radii, energy offset corrections) that maximize prediction accuracy while maintaining computational efficiency. Results inform production NequIP models documented in the `nequip/` directory and enable accurate predictions for graphene, diamond, and other carbon allotropes evaluated through the structural analysis utilities in `src/nequip/metrics/`.
