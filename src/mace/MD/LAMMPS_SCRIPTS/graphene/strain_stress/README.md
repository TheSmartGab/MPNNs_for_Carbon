# Graphene Strain and Stress LAMMPS Simulations

This directory contains LAMMPS simulation input scripts and shell orchestration scripts for performing strain and stress analyses on graphene systems using machine learning interatomic potentials (MLIPs). These scripts enable systematic evaluation of graphene's elastic properties, breaking rates, and mechanical response under various strain configurations.

## Content

- The **mliap** notation indicate that the script is thought for the mace mliap interface in lammps. I deemed it necessary to write separate files since there are a few syntattical differences with the legacy interface.
- **batch** scripts are automation scripts used in an environment where you intend to run multiple simulations in parallel on distinct gpus.
- **strain_stress** run the initial strain_stress from an initial to a final strain in multiple steps. It is recommended to use it for the initial strain scan to find the approximate strain at which a system breaks.
- **change_strain** runs a single strain increase. It was used for the breaking rate inference simulations. Technically, you could use the strain_stress scripts, adapting their parameters to achieve the same result, but I found it more practical to save this version with some logic tweaks to provide a more immediate argument setting. Equivalently, you could reproduce the strain_stress script by concatenating change_strain runs. change_strain was also used to prolong runs without changing the strain by setting to 0 the number of steps at the next strain.
- the **restart** scripts are used to restart strain_stress simulations.
