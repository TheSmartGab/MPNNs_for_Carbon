# Graphene Strain Scan Simulations (`KAGGLE_DOWNLOAD/from_C2_v1/graphene_strain_scan`)

This directory contains LAMMPS simulation files and outputs from graphene strain scan studies. These simulations evaluate the MACE model's predictions for graphene's elastic properties, mechanical response under tensile/compressive stress, and structural stability limits under various deformation configurations.

## Contents:

### Input Configuration Files:
- **`graphene.lammps-data`, `relaxed_graphene.lammps-data`**: LAMMPS data files containing initial and relaxed graphene atomic configurations for strain scan simulations

### Simulation Output Files:
- **`dump.dump`, `log.lammps`**: LAMMPS dump trajectory file and execution log documenting the strain scan simulation progress, timing, and energy evaluations

### Strained Configuration Outputs (`OUTPUT/`):
Directory containing extxyz files of graphene configurations at various strain levels, named with strain values (e.g., `-0.0003463203567064532.extxyz`, `0.0013764631429276086.extxyz`, etc.). These files represent atomic configurations at specific strain percentages ranging from compressive to tensile deformations.

- **`strain_e.out`**: Output file containing strain-energy data extracted from the simulation for elastic property analysis

## Usage:

These graphene strain scan simulations validate ML potential accuracy under mechanical deformation and inform elastic constant calculations documented in `src/mace/cell_scan/e_vs_lp_graphene.py`. The strained configurations in the OUTPUT/ directory are used for training or validating MACE models' treatment of graphene's flexural modes and elastic properties under stress.
