#!/bin/bash

#SBATCH -p light
#SBATCH -t 01:00:00
#SBATCH --ntasks 1
#SBATCH -c 8
#SBATCH --account GNPGi
#SBATCH --mem 32GB
#SBATCH -J graphene_strain_stress
#SBATCH --get-user-env
#SBATCH -D /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/MD_RUNS/from_C2_v1/TESTING_LAMMPS/graphene_strain_stress/5_5
#SBATCH -e /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/MD_RUNS/from_C2_v1/TESTING_LAMMPS/graphene_strain_stress/5_5/slurm_error.out
#SBATCH -o /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/MD_RUNS/from_C2_v1/TESTING_LAMMPS/graphene_strain_stress/5_5/slurm_output.out

# activate envirorment, export libraries and bin, export OMP_NUM_THREADS
conda activate /exa/projects/GNPGi/programs/sources/conda_envs/mace_cpu
source /exa/projects/GNPGi/programs/sources/mace_lammps_cpu/build_mace_lammps_cpu/export.sh
export OMP_NUM_THREADS=8

# parameters for the script
SCRIPT=/exa/projects/GNPGi/PROJECT/src/mace/MD/LAMMPS_SCRIPTS/thermalize.in
INPUT_CONFIG=relaxed_graphene_5_5.lammps-data
MODEL_FILE=../../../MACE.model-lammps.pt
OUT_CONFIG=thermalized_graphene_5_5.lammps-data
TEMP=300

srun lmp_mace -in $SCRIPT -var INPUT_CONFIG ${INPUT_CONFIG} -var OUT_CONFIG ${OUT_CONFIG} -var TEMP ${TEMP} -var MODEL ${MODEL_FILE}
