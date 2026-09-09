# Quantum ESPRESSO Pair Scan Offset Plots (`plots/qe_pair_scan_offset`)

This directory contains generated plot outputs from Quantum ESPRESSO pair scan calculations with energy offset corrections. These scans compute pairwise interaction energies with adjusted reference levels, used to validate ML potential accuracy against DFT reference data with consistent zero-energy baselines.

## Contents:

Generated visualization files for:
- QE pair scan energy vs distance curves with offset corrections
- Adjusted reference DFT potentials for consistent E0 alignment
- Pair interaction force distributions with shifted energy references

## Usage:

These plots are generated from Quantum ESPRESSO pair scan calculations stored in `nequip/qe_pair_scan_gridsearch_offset/`. The offset-corrected pair potentials ensure consistent energy referencing across different atom types and configurations, enabling fair comparison between MLIP predictions and DFT ground truth for multi-element systems.