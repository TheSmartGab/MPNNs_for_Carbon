#!/bin/bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/generate_independent_configurations_mliap.sh -n 100 -i STRAINSTRESS/190150.lammps-data -o RESTARTS_0.20 -r prolong_run_mliap_change.conf

export CUDA_VISIBLE_DEVICES='1'
cd 0.204
ln -s ../RESTARTS_0.20 RESTARTS_0
source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ change_strain_mliap.conf prolong_run_mliap_change.conf &

cd ../0.206 
ln -s ../RESTARTS_0.20 RESTARTS_0
source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ change_strain_mliap.conf prolong_run_mliap_change.conf &

export CUDA_VISIBLE_DEVICES='2'
cd ../0.208
ln -s ../RESTARTS_0.20 RESTARTS_0
source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ change_strain_mliap.conf prolong_run_mliap_change.conf &

cd ../0.210
ln -s ../RESTARTS_0.20 RESTARTS_0
source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ change_strain_mliap.conf prolong_run_mliap_change.conf &
