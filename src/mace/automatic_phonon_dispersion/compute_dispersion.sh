#!/bin/bash

echo "Usage: compute_pot.sh <target_dir> <model_file> <input_filename> <n_replica_x> <n_replica_y> <n_replica_z>
        target_dir: directory with INPUT dir with required input files
        model_file: path to the deploied model that will be used in lammps
        input_filename: name of input file in INPUT dir
        n_replica_x: number of replicas in x direction for the supercell
        n_replica_y: number of replicas in y direction for the supercell
        n_replica_z: number of replicas in z direction for the supercell"

target_dir="$1"
target_dir=$(realpath "$target_dir")
current_dir=$(pwd)

model_file="$2"
input_filename="$3"

##########################################################
# directories to work with
INPUT_DIR="${target_dir}/INPUT" # input config and scripts
WORKING_DIR="${target_dir}/WORKING" # where temporary files are stored
OUTPUT_DIR="${target_dir}/OUTPUT" # where final results are stored

echo "================================================"
echo "Directories used:"
echo "${INPUT_DIR}"
echo "${WORKING_DIR}"
echo "${OUTPUT_DIR}"
echo "================================================"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${WORKING_DIR}"

##########################################################
# programs
lammps="lmp"
phonopy="phonopy"
bandplot="phonopy-bandplot"

##########################################################
# parameters
n_replica_x="$4"
n_replica_y="$5"
n_replica_z="$6"

##########################################################
# input files
# primitive cell file
input_config="${INPUT_DIR}/${input_filename}"
# relax input
relax_input="${INPUT_DIR}/lammps_relax.in"
# forces input
forces_input="${INPUT_DIR}/lammps_phono.in"
# plot config file 
config_file="${INPUT_DIR}/band.conf"

##########################################################
# working files
# forces file
forces_file="${WORKING_DIR}/force.*"

##########################################################
# RUN
##########################################################

echo '[INFO] Run start'

echo '[INFO] generating supercell with phonopy'
# generate supercell with phonopy
"${phonopy}" --lammps -c "${input_config}" -d --dim "${n_replica_x} ${n_replica_y} ${n_replica_z}"

# move phonopy generated files to working dir
mv supercell* "${WORKING_DIR}"
mv *.yaml "${WORKING_DIR}"

echo '[INFO] computing forces with lammps'
# compute forces with lammps
"${lammps}" -in ${forces_input} -var INPUT_DATA "${WORKING_DIR}/supercell-001" -var OUTPUT_DATA "${WORKING_DIR}/force.0" -var MODEL_FILE ${model_file}

# move to working dir to ease file handling with phonopy
cd "${WORKING_DIR}"

echo '[INFO] generating FORCE_SETS'
# generate FORCE_SETS file
# read forces computed by lammps and create FORCE_SETS
phonopy -f "force.0"

echo '[INFO] plotting results'
# plot dispersion
phonopy -p -s "${config_file}"
phonopy-bandplot band.yaml -o "phonopy_plot_legacy.pdf" --legacy -l --ylabel "frequency [THz]" -t "Phonon dispersion"
phonopy-bandplot band.yaml -o "phonopy_plot.pdf"
phonopy-bandplot band.yaml --gnuplot > band.data # save raw data for further plotting if needed

cd "${current_dir}"
mv "${WORKING_DIR}/"*.pdf ${OUTPUT_DIR}
mv ${WORKING_DIR}/band.data ${OUTPUT_DIR}
mv ${WORKING_DIR}/*.yaml ${OUTPUT_DIR}

mv log.lammps "${WORKING_DIR}"

cd $current_dir

echo "[INFO] Plotting group velocities"
# plot group velocities
python3 "${target_dir}"/plot_band_vel.py ${OUTPUT_DIR}/band.yaml
