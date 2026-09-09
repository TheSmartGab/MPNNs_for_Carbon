# Structural Analysis (`structure`)

This directory contains utilities for structural analysis of molecular dynamics trajectories, including broken bond detection, rate inference from trajectory data, and visualization of potential energy surfaces.

## Key Scripts:

- **`structural_analysis.py`**: General structural analysis library for identifying motifs, coordination numbers, and local symmetry in atomic configurations. USed by other scripts.

- **`broken.py`**: Utilities for determining whether a graphene sheet is broken or not. It has been used in production runs to determine what runs to stop and which to carry on simulating.

- **`ArreniusPlot.py`**: Generate Arrhenius plots (ln(rate) vs 1/kT) from reaction rate data to extract activation energies.

- **`compare_rates.py`**: Compare files or computed rates across different simulations or models.

- **`dump2traj.py`**: Convert dump files (LAMMPS format) to trajectory formats (e.g., .traj using ASE).

- **`extract_frame.py`**, **`extract_frames_parallel.py`**, **`extract_frame_v2.py`**: Extract specific frames or configurations from molecular dynamics trajectories. Parallel versions for processing large datasets efficiently.

- **`extract_index_parallel.py`**: Extract configurations by index in parallel.

- **`find_broken_traj.py`**: Find broken bonds across entire MD trajectories.

- **`InferRate.py`**, **`InferRateDriver.py`**: Infer reaction rates from trajectory data using event counting or kinetic analysis methods.

- **`lammps_cell_length.py`**: Extract cell length information from LAMMPS simulation outputs.

- **`plot_cell_vs_distance.py`**, **`plot_dimer_potential.py`**, **`plot_ef_vs_distance.py`**, **`plot_thermo.py`**: Visualization utilities for cell parameters, dimer potentials, energy vs distance curves, and thermodynamic properties.

- **`rate_vs_beta.py`**: Analyze how reaction rates vary with inverse temperature (beta = 1/kT).
