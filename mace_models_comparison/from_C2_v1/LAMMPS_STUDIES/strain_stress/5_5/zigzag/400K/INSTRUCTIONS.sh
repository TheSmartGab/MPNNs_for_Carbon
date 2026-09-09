#!/bin/bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

# 1. INITIAL GENERATION (GPU 1)
# This runs in the foreground. The script will wait here until it finishes.
export CUDA_VISIBLE_DEVICES='1'
echo "Step 1: Generating independent configurations on GPU 1..."
#source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/generate_independent_configurations_mliap.sh -n 100 -i STRAINSTRESS/190150.lammps-data -o RESTARTS_0.20 -r prolong_run_mliap_change.conf

echo "Step 1 Complete. Starting parallel GPU branches..."

# 2. GPU 1 BRANCH (Serial: 0.204 -> 0.206)
(
    export CUDA_VISIBLE_DEVICES='1'
    
    # Task 0.204
    cd 0.204 || exit
    pwd
    cat *.conf
    ln -snf ../RESTARTS_0.20 RESTARTS_0
    source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ run_mliap_change.conf prolong_run_mliap_change.conf
    
    # Task 0.206 (Runs ONLY after 0.204 finishes)
    cd ../0.206 || exit
    ln -snf ../RESTARTS_0.20 RESTARTS_0
    source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ run_mliap_change.conf prolong_run_mliap_change.conf
) &
# 3. GPU 2 BRANCH (Serial: 0.208 -> 0.210)
(
    export CUDA_VISIBLE_DEVICES='2'
    
    # Task 0.208
    cd 0.208 || exit
    pwd
    cat *conf
    ln -snf ../RESTARTS_0.20 RESTARTS_0
    source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ run_mliap_change.conf prolong_run_mliap_change.conf
    
    # Task 0.210 (Runs ONLY after 0.208 finishes)
    cd ../0.210 || exit
    pwd
    cat *.conf
    ln -snf ../RESTARTS_0.20 RESTARTS_0
    source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/change_strain_mliap_multiple.sh 5 RESTARTS_ OUTPUTS_ run_mliap_change.conf prolong_run_mliap_change.conf
) &

# Wait for both GPU branches to finish before exiting the main script
wait
echo "All tasks finished successfully."
