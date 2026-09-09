# NequIP Pair Scan Configurations (`nequip/pair_scans`)

This directory serves as a logical grouping for NequIP pair scan gridsearch configurations and outputs. Pair scans compute two-body interaction potentials across different atom distances, used to validate ML potential accuracy against DFT reference data.

## Related Directories:

The actual pair scan configuration directories are located in the `nequip/` root directory:
- `GPAW_pair_scan_gridsearch_offset*` - NequIP grid searches with GPAW references and energy offsets
- `qe_pair_*_gridsearch` - Quantum ESPRESSO pair scan gridsearch configurations
- `merged/` - Merged results from multiple pair scan runs

## Contents:

Pair scan configurations include:
- Grid search hyperparameter settings for cutoff radii, basis functions
- Energy offset corrections for consistent reference levels
- ZBL (Ziegler-Biersack-Littmark) potential inclusion/exclusion variants (_NOZBL directories)

## Usage:

These pair scan configurations are used to validate NequIP two-body interaction potentials against DFT ground truth from GPAW and Quantum ESPRESSO. Generated plots and performance visualizations are stored in the `plots/` directory under `nequip_GPAW_C2_f_cutoff_100/`, `qe_pair_scan/`, and related subdirectories.