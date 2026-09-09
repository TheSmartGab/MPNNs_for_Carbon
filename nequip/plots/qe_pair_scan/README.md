# Quantum ESPRESSO Pair Scan Plots (`plots/qe_pair_scan`)

This directory contains generated plot outputs and performance visualizations from Quantum ESPRESSO pair scan calculations. These scans compute pairwise interaction energies and forces for atom pairs across different distances, used to validate ML potential accuracy against DFT reference data.

## Contents:

Generated visualization files for:
- QE pair scan energy vs distance curves
- Pair interaction force distributions
- Reference DFT potentials for MLIP validation

## Usage:

These plots are generated from Quantum ESPRESSO pair scan calculations stored in `nequip/qe_pair_scan_gridsearch/` and `nequip/qe_pair_scan_gridsearch_offset/`. The reference pair potentials are used to validate MLIP predictions (MACE, NequIP) against DFT ground truth for two-body interaction energies.