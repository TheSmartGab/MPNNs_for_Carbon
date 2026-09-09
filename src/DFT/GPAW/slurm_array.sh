#!/bin/bash
#SBATCH --job-name=GPAW_C2
#SBATCH --output=logs/gpaw_sp_%A_%a.out
#SBATCH --error=logs/gpaw_sp_%A_%a.err
#SBATCH --time=00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=5500
#SBATCH --array=0-15%2

# Load file list

mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -printf "%f\n")

FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

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
