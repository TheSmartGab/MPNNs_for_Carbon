#!/bin/bash

# output and general management
# MODIFY at will
# output file (stdout)
#SBATCH -o /exa/projects/GNPGi/gpaw_C2_scan/logs/gpaw_c2_scan.%j.%N.out
# output error (stderr)
#SBATCH -e /exa/projects/GNPGi/gpaw_C2_scan/logs/gpaw_c2_scan.%j.%N.err
# working directory
#SBATCH -D /exa/projects/GNPGi/gpaw_C2_scan
#SBATCH --job-name=GPAW_C2_SCAN
#SBATCH --get-user-env # import user variables.

# resources
#SBATCH -p light
#SBATCH --time=02:00:00
#SBARCH --nodes=1
#SBATCH --cpus-per-task=1 # for each of the n parallel task, use this many cpus. I find this better than -ncpus (-c) for array parallelization
#SBATCH --mem-per-cpu=4000MB # for each of the cpus used, allocate this much memory
#SBATCH --account=GNPGi
#SBATCH --array=0-225%10 # if you have N files, pass 0-(N-1). %n runs n tasks in parallel. 10 is the maximum number of parallel jobs I can submit apparently. this is somewhat weird, with a bash for loop, I could allocate, say, 32 cores, and run in background 32 parallel instances of the same script.

# mail
#SBATCH --mail-type=ALL                                                     #SBATCH --mail-user=gabriele.colombo13@studenti.unimi.it

# run with  
# sbatch --export=ALL,INPUT_DIR=inputs_scan/,OUTPUT_DIR=outputs_scan/,ECUT=500,KPTS="1 1 1",FORMAT=extxyz,SCRIPT=/home/gabri/Thesis/PROJECT/src/DFT/GPAW/gpaw_run_single_point.py /home/gabri/Thesis/PROJECT/src/DFT/GPAW/slurm_array.sh
# or similar. in this program, some command line parameters are read:
# INPUT_DIR = directory with input files
# OUTPUT_DIR = directory to store output
# ECUT = energy cutoff for plane waves
# KPTS = k points for plane waves
# FORMAT = output format
# SCRIPT = script to run in paralel
# in the example are the paths in my local prohect directory for the original use of this script. substitute with paths in your project folder.

# activate conda envirorment with gpaw (and ase)
module load anaconda/2025-6
source /exa/software/Anaconda/2025-6/0/.bashanaconda
conda activate /exa/projects/GNPGi/programs/sources/conda_envs/gpaw

# Load file list
# look for files in input dir
# printf the file names with newlines at the end
# mapfile stores the output of printf in an array
# -t removes strips \n characters
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -printf "%f\n")

# this is the file to process for
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# set omp threads to the number of cpus per task.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Processing: $FILE"
echo "Using ECUT=$ECUT, KPTS=$KPTS, FORMAT=$FORMAT"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "OMP threads: $OMP_NUM_THREADS"

python3 "$SCRIPT" \
    "$INPUT_DIR/$FILE" \
    --ecut "$ECUT" \
    --kpoints $KPTS \
    --format "$FORMAT" \
    --output_name "$OUTPUT_DIR/$FILE"
