#!/bin/bash

# here INPUT dir should already exist with appropriate input files, check the input dir in this directory as an example

echo "Usage: compute_pot.sh <target_dir> <model_file>
        target_dir: directory with INPUT dir with required input files
        model_file: path to the deploied model that will be used in lammps"

target_dir="$1"
target_dir=$(realpath "$target_dir")

current_dir=$(pwd)
INPUT_DIR=$(realpath ${target_dir}/INPUT)
OUTPUT_DIR=$(realpath ${target_dir}/OUTPUT)
WORKING_DIR=$(realpath ${target_dir}/WORKING)

mkdir -p $OUTPUT_DIR
mkdir -p $WORKING_DIR

lammps="lmp_PHONON"

model_file="$2"

# clear energy_vs_distance.txt
echo "" > "${OUTPUT_DIR}/energy_vs_distance.txt"
${lammps} -in "${INPUT_DIR}/"compute_pot.in -var MODEL_FILE "${model_file}" -var OUTPUT_FILE "${OUTPUT_DIR}/energy_vs_distance.txt"
mv log.lammps "${WORKING_DIR}/"

python3 ${target_dir}/plot.py  "${OUTPUT_DIR}/energy_vs_distance.txt" "${OUTPUT_DIR}/potential_plot.pdf" "${OUTPUT_DIR}/info.txt"
