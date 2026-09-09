#!/bin/bash

DIRS=("0.205" "0.206" "0.207" "0.208" "0.209" "0.210")

for d in ${DIRS[@]};
do
	cd $d
	source /home/gabri/Thesis/PROJECT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ run_mliap_change.conf  prolong_run_mliap_change.conf
	cd ..
done
