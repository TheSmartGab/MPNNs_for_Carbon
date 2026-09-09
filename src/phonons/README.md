# Phonon Analysis (`phonons`)

This directory contains utilities for phonon dispersion calculations, stability analysis, and visualization using ASE, Phonopy, and ML potential calculators (MACE, NequIP).

## Key Scripts:

- **`plot_phonons.py`**: Main phonon plotting utility for visualizing phonon dispersions and density of states.

### [`lib/`](./lib/)
Core phonon library modules:
- Unit conversions, format handlers, kpoints management, and interfaces with ASE, Phonopy, and Materials Project data formats.

## Usage

These scripts are used to compute and visualize phonon band structures, check for imaginary frequencies (instabilities), and analyze the density of states. The `phonons/lib/` subdirectory contains reusable utility functions that are imported by higher-level workflow scripts in the `mace/phonons/` and `nequip/automatic_phonon_dispersion/` directories.