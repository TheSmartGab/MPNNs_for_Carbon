# Graphene Phonon Reference Data (`EXTERNAL_DATA/graphene_phonons`)

This directory contains reference phonon dispersion calculations for graphene, used as a benchmark for validating ML potential (MACE, NequIP) and DFT phonon calculations of 2D carbon materials.

## Contents:

Reference phonon data for graphene including:
- Band structure calculations along the GMKG high-symmetry k-path (typical for hexagonal lattices)
- Acoustic and optical phonon branch references
- In-plane and out-of-plane (flexural) mode dispersions
- Phonon density of states (DOS) reference values

## Usage

These graphene phonon references are used to:
1. Validate MLIP-predicted phonon dispersions against DFT/GPAW/Quantum ESPRESSO references
2. Check for imaginary frequencies in the acoustic branches that indicate lattice instabilities
3. Benchmark ML potential accuracy for 2D material vibrations and flexural modes

Comparison scripts in `src/mace/phonons/compute_dispersion.py` and `src/nequip/automatic_phonon_dispersion/plot_band_vel.py` use similar k-path conventions (e.g., `GMKG`) for hexagonal systems.