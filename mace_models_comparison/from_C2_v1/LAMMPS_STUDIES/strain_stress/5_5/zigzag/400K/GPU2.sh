#!/bin/bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

cd 0.208

source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ change_strain_mliap.conf prolong_run_mliap_change.conf

cd ../0.210 

source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ change_strain_mliap.conf prolong_run_mliap_change.conf
