# Graphene Layer Separation Studies (`KAGGLE_DOWNLOAD/from_C2_v1/TESTING_LAMMPS/graphene_layer_sep`)

This directory contains LAMMPS simulation files and outputs from graphene interlayer distance scanning studies. These tests evaluate the MACE model's ability to accurately predict van der Waals interactions and binding energies between stacked graphene layers, which is critical for modeling 2D material heterostructures and graphite systems.

## Contents:

### Input Configuration Files:
- **`2_layer_graphite.lammps-data`, `fixed_relaxed_graphene.lammps-data`, `single_2_layer_graphite.lammps-data`**: LAMMPS data files containing double-layer graphene and single-layer configurations for interlayer distance studies

### LAMMPS Output Files:
- **`log.lammps`**, **`graphene_distance_scan.out`**: LAMMMS execution logs and scan output files documenting energy evaluations across different interlayer separations
- **`relaxed_graphene.lammps-data`, `single_relaxed_graphene.lammps-data`**: Relaxed graphene configurations used as reference states

### Generated Visualizations:
- **`BM_graphene_interlayer_distance.pdf`**, **`interlayer_distance.pdf`**: Publication-quality plots showing binding energy curves versus interlayer distance (Buckingham or Morse potential fits)
- **`diamond_lp_scan.svg`, `interlayer_distance_scan.svg`**: SVG visualizations of lattice parameter scans and interlayer distance evaluations
- **`plot.ipynb`**, **`relaxed_scan.lammpstrj`**: Jupyter notebook for analysis and LAMMPS trajectory from relaxed scan simulations

### Additional Directories:
- **`DUMP/`**: Directory containing LAMMPS dump files from various interlayer separation configurations
- **`old_runs/`**: Historical run outputs and reference calculations for comparison studies

## Usage:

These graphene layer separation studies validate the ML potential's treatment of weak van der Waals interactions between 2D layers. The results inform interlayer coupling parameters used in heterostructure simulations documented in `src/mace/MD/LAMMPS_SCRIPTS/graphene/interlayerdistance/` and ensure accurate modeling of graphite and multilayer graphene systems.
