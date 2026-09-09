#!/bin/bash

# output and general management
# MODIFY at will
# output file (stdout)
#SBATCH -o /home/gabri/Thesis/PROJECT/DFT_tutorials/exercise/C2_pair_distance_potential/logs/qe_c2_closer.%j.%N.out
# output error (stderr)
#SBATCH -e /home/gabri/Thesis/PROJECT/DFT_tutorials/exercise/C2_pair_distance_potential/logsqe_c2_closer.%j.%N.err
# working directory
#SBATCH -D /home/gabri/Thesis/PROJECT/DFT_tutorials/exercise/C2_pair_distance_potential/
#SBATCH --job-name=QE_C2_CLOSER
#SBATCH --get-user-env # import user variables.

# resources
#SBATCH --time=01:00:00
#SBARCH --nodes=1
#SBATCH --cpus-per-task=2 # for each of the n parallel task, use this many cpus. I find this better than -ncpus (-c) for array parallelization
#SBATCH --ntasks 2
#SBATCH --mem-per-cpu=800MB # for each of the cpus used, allocate this much memory
#SBATCH --array=0-3%2 # if you have N files, pass 0-(N-1). %n runs n tasks in parallel. 10 is the maximum number of parallel jobs I can submit apparently. this is somewhat weird, with a bash for loop, I could allocate, say, 32 cores, and run in background 32 parallel instances of the same script.

# mail

# run with  
# sbatch --export=ALL,INPUT_DIR=inputs_scan/,OUTPUT_DIR=outputs_scan/,ECUT=500,KPTS="1 1 1",FORMAT=extxyz,SCRIPT=/home/gabri/Thesis/PROJECT/src/DFT/GPAW/gpaw_run_single_point.py /home/gabri/Thesis/PROJECT/src/DFT/GPAW/slurm_array.sh
# or similar. in this program, some command line parameters are read:
# INPUT_DIR = directory with input files
# OUTPUT_DIR = directory to store output
# in the example are the paths in my local prohect directory for the original use of this script. substitute with paths in your project folder.

# Load file list
# look for files in input dir
# printf the file names with newlines at the end
# mapfile stores the output of printf in an array
# -t removes strips \n characters
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -printf "%f\n")

# this is the file to process for
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
FILE_NAME="${FILE%.*}"

# quantum espresso output directory
OUT_DIR=$OUTPUT_DIR/$FILE_NAME
echo "creating output directory ${OUT_DIR}"
mkdir "${OUT_DIR}"

OUTPUT_FILE=$OUT_DIR/$FILE_NAME.out

# set omp threads to the number of cpus per task.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Processing: $FILE"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "OMP threads: $OMP_NUM_THREADS"

mpiexec -n $SLURM_NTASKS pw.x -in ${INPUT_DIR}/$FILE > "${OUTPUT_FILE}"
