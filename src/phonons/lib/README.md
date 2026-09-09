# Phonon Library Utilities (`phonons/lib`)

This subdirectory contains core library functions for phonon analysis, including conversions between different unit systems, file format handlers, kpoints management, and interfaces with various phonon calculation packages (Phonopy, ASE, Materials Project).

## Key Modules:

- **[`conversions.py`](conversions.py)**: Unit conversion utilities for phonon frequencies, wave vectors, and other physical quantities. Ensures consistent units across different phonon calculation tools.

- **[`formats.py`](formats.py)**: File format handlers for various phonon output formats (e.g., band.yaml, eigs.phonopy).

- **[`__init__.py`](__init__.py)**: Module initialization, exports core phonon library functions.

- **[`kpoints_manager.py`](kpoints_manager.py)**: K-point mesh generation and management for Brillouin zone sampling in phonon calculations. Ensures proper q-point grids for Fourier interpolation.

- **[`ph_ase.py`](ph_ase.py)**: Interface with ASE (Atomic Simulation Environment) phonon calculator utilities.

- **[`ph_mp.py`](ph_mp.py)**: Utilities for Materials Project (MP) phonon data formats and conversions.

- **[`ph_phonopy.py`](ph_phonopy.py)**: Functions to plot Phonopy-style data (band.yaml format). Includes kpoint and frequency conversion functions, and utilities to extract q-points and frequencies from Phonopy band structure data.

## Usage

These library modules are imported by higher-level phonon analysis scripts in the `phonons/` directory and MACE/NequIP phonon workflow scripts. They provide standardized interfaces for reading, converting, and visualizing phonon dispersion data across different calculation packages.