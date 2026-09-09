# FULLBATCH Training Configuration (`KAGGLE_DOWNLOAD/from_C2_v1/FULLBATCH`)

This directory contains the full-batch training configuration and outputs for the MACE model trained on C2 (diatomic carbon) configurations from version 1 of the Kaggle competition dataset. This represents a comprehensive training run using all available data without mini-batch sampling strategies.

## Activities Performed:

### Training Process:
The FULLBATCH directory documents a complete MACE model training workflow where:
1. All C2 (diatomic carbon) configurations from the `from_C2_v1` dataset are used for training
2. No mini-batch sampling stratification is applied - the entire dataset is processed in full batches
3. Model weights and optimization states are checkpointed at regular intervals throughout the 100-epoch training process

### Evaluation Metrics Computed:
After training completion, the following performance metrics are evaluated and stored:
- Energy prediction accuracy (MAE, RMSE, Q95 quantiles)
- Force prediction accuracy (MAE, RMSE, relative MAE/RMSE)
- Stress tensor prediction quality (when applicable)
- Training loss progression across epochs

## Contents:

### Subdirectories and Activities:
- **`checkpoints/`**: Contains model checkpoint files saved during the full-batch training process. These checkpoints capture model weights at various epochs or evaluation points, allowing for:
  - Recovery from interrupted training runs
  - Comparison of model performance across different training stages
  - Selection of optimal models based on validation metrics

- **`logs/`**: Training log files documenting loss progression, validation metrics, and convergence behavior. These logs contain:
  - Per-step or per-epoch loss values for energy, forces, and stress components
  - Learning rate scheduling information
  - Stochastic Weight Averaging (SWA) phase transitions

- **`results/`**: Final evaluation results and performance metrics computed on validation/test sets after training completion. This directory contains:
  - PDF visualizations of model performance across different metric types
  - Training loss curves showing convergence behavior
  - Comparative plots between different model configurations

## Usage in MLIP Workflows:

These full-batch training outputs are used as a baseline comparison against other MACE model configurations (like smaller batch sizes or different architectural variants) to evaluate the impact of training strategy on ML potential accuracy and convergence behavior. The checkpoints, logs, and results inform the production-ready models documented in the `mace/` directory and help identify optimal training strategies for carbon-based systems.
