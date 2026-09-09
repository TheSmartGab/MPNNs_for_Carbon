# NequIP Merged Results (`nequip/merged`)

This directory contains merged comparison results and performance visualizations from multiple NequIP model training runs or configurations. These files aggregate error distributions, quantile analyses, and performance metrics across different models or hyperparameter settings for comprehensive benchmarking against DFT reference data.

## Contents:

### Performance Plots:
- `energy_errors.pdf` - Aggregated energy prediction error distribution plot across merged model runs
- `force_errors.pdf` - Aggregated force prediction error distribution plot across merged model runs
- `overlapped_errors.pdf` - Combined visualization showing overlapped energy and force error distributions

### Error Quantile Analyses:
- `errors_quantiles_energy_force.txt` - Tabular data containing quantile statistics (e.g., 2.5%, 50%, 97.5%) for energy and force prediction errors across the merged model ensemble or dataset splits.

## Usage:

These merged results are used to:
1. Compare overall model performance across different NequIP training configurations or hyperparameter settings
2. Generate comprehensive error distributions for ML potential validation against DFT references
3. Benchmark custom-trained models against production baselines or alternative architectures

Related gridsearch configurations generating these merged results are stored in `GPAW_pair_scan_gridsearch_offset*/` and `qe_pair_*_gridsearch/` directories, with visual plots also available in `plots/nequip_models_GRAPHENE_MLIP/`.