# Diamond Phonon Reference Data (`EXTERNAL_DATA/diamond_phonons`)

This directory contains reference phonon dispersion calculations for diamond, used as a benchmark for validating ML potential (MACE, NequIP) and DFT phonon calculations.

## Contents:

Reference phonon data for diamond including:
- Band structure calculations along high-symmetry k-path directions
- Phonon density of states (DOS) reference values
- Stability boundary analyses for diamond lattice

## Usage

These reference phonon calculations are used to:
1. Validate MLIP-predicted phonon dispersions against DFT references
2. Check for imaginary frequencies that indicate structural instabilities
3. Benchmark ML potential accuracy for covalent bond vibrations

Comparison scripts in `src/mace/phonons/` and `src/nequip/metrics/` use this reference data to generate dispersion plots and stability boundary analyses.