# NequIP Compilation Scripts (`src/nequip/compile`)

This directory contains scripts for compiling and delivering NequIP (Neural Equivariant Interatomic Potentials) models. These utilities handle the compilation of trained neural network architectures into optimized formats suitable for production molecular dynamics simulations.

## Contents:

### `compile_and_deliver.sh`
Shell script that compiles NequIP models and delivers them to appropriate directories or systems for use in MD simulations. This script handles the conversion of training outputs into runtime-compatible model files that can be used with LAMMPS or other simulation packages.

**Key functionality:**
- Compiles trained NequIP neural network architectures
- Optimizes models for production MD simulation performance
- Delivers compiled models to appropriate directories or compute systems

## Usage:

This script is part of the NequIP workflow pipeline, taking training outputs from `src/nequip/metrics/` evaluations and preparing them for use in LAMMPS simulations via the `LAMMPS_SCRIPTS/` utilities or other MD frameworks. The compiled models enable efficient evaluation of energies, forces, and stresses during production molecular dynamics runs.
