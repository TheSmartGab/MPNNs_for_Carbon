# General Purpose Tools (`tools`)

This directory contains general utility scripts for common tasks across the project, including configuration generation and metric visualization.

## Key Scripts:

- **`generate_configs.py`**: Generate atomic configurations or simulation setups programmatically. Useful for creating test structures, initial configurations for MD simulations, or grid scans of structural parameters.

- **`plot_metric_vs_hparams.py`**: Plot how model metrics (RMSE, MAE, etc.) vary with hyperparameters or model configuration choices. This script is useful for comparing different model architectures, training sizes, or preprocessing strategies.

## Usage

These utility scripts are designed to be reusable across different parts of the workflow and can be called from shell scripts or Jupyter notebooks as needed for specific analysis tasks.