#!/bin/bash

#SBATCH -p light
#SBATCH -t 05:00:00
#SBATCH --ntasks 1
#SBATCH -c 8
#SBATCH --account GNPGi
#SBATCH --mem 8GB
#SBATCH -J graphene_strain_stress
#SBATCH --get-user-env
#SBATCH -D /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/MD_RUNS/from_C2_v1/TESTING_LAMMPS/graphene_strain_stress/5_5
#SBATCH -e /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/MD_RUNS/from_C2_v1/TESTING_LAMMPS/graphene_strain_stress/5_5/restart_slurm_error_strain_stress_xx.err
#SBATCH -o /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/MD_RUNS/from_C2_v1/TESTING_LAMMPS/graphene_strain_stress/5_5/restart_slurm_output_strain_stress_xx.out

# activate envirorment, export libraries and bin, export OMP_NUM_THREADS
conda activate /exa/projects/GNPGi/programs/sources/conda_envs/mace_cpu
source /exa/projects/GNPGi/programs/sources/mace_lammps_cpu/build_mace_lammps_cpu/export.sh
export OMP_NUM_THREADS=8

# parameters for the script
SCRIPT=/exa/projects/GNPGi/PROJECT/src/mace/MD/LAMMPS_SCRIPTS/graphene/restart_strain_stress.in
INPUT_CONFIG=restart_config.lammps-data
MODEL_FILE=../../../MACE.model-lammps.pt
MAX_EXX=0.25
NSTRAINS=25
TEMP=300

# restart variables
CURRENT_STEP=7245
CURRENT_STRAIN=20
CURRENT_EXX=0.20

OUTSTREAM="restart_strain_stress_xx.out"

srun lmp_mace -in $SCRIPT -var INPUT_CONFIG ${INPUT_CONFIG} -var TEMP ${TEMP} -var MODEL ${MODEL_FILE} -var max_exx ${MAX_EXX} -var NSTRAINS ${NSTRAINS} -var TEMP ${TEMP} -var current_step ${CURRENT_STEP} -var current_exx ${CURRENT_EXX} -var current_strain ${CURRENT_STRAIN} > ${OUTSTREAM}
