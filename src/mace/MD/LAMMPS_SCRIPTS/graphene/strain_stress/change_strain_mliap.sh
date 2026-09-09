#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPSIN="${SCRIPT_DIR}/change_strain_mliap.in"

INPUTDIR=''
OUTPUTDIR=''
RUNCONF=''

print_usage() {
  printf "Usage: $0 --input_dir <dir> --output_dir <dir> --runconf <conf>\n"
}

OPTS=$(getopt -o i:o:r: --long input_dir:,output_dir:,runconf: -n 'script' -- "$@")
eval set -- "$OPTS"

while true; do
  case "$1" in
    -i|--input_dir) INPUTDIR="$2"; shift 2 ;;
    -o|--output_dir) OUTPUTDIR="$2"; shift 2 ;;
    -r|--runconf) RUNCONF="$2"; shift 2 ;;
    --) shift; break ;;
    *) print_usage; exit 1 ;;
  esac
done

mkdir -p "${OUTPUTDIR}"

# Load simulation parameters
source "${RUNCONF}"

FILES=("${INPUTDIR}"/*.lammps-data)

declare -i MIN_PRIME_INDEX=100
PRIME_INDEX=${MIN_PRIME_INDEX}

export OMP_NUM_THREADS=4
MAX_JOBS=4   # Max concurrent LAMMPS runs

# Function to run a single simulation
run_lammps() {
    local f="$1"
    local prime="$2"

    local file=$(basename "$f")
    local name="${file%.lammps-data}"

    local LOGFILE="${OUTPUTDIR}/${name}.log"
    local DUMPFILE_PATH="${OUTPUTDIR}/${DUMPDIR}/${name}.dump"

    mkdir -p "${OUTPUTDIR}/${DUMPDIR}"

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -log "${LOGFILE}" \
        -in "${LAMMPSIN}" \
        -var INPUT_CONFIG "$f" \
        -var MODEL "${MODEL}" \
        -var TEMP "${TEMP}" \
        -var SEED "${prime}" \
        -var NSTRAINS "${NSTRAINS}" \
        -var timestep "${timestep}" \
        -var tdamp "${tdamp}" \
        -var stepscurrent "${stepscurrent}" \
        -var stepsnext "${stepsnext}" \
        -var target_exx "${target_exx}" \
        -var target_eyy "${target_eyy}" \
        -var target_ezz "${target_ezz}" \
        -var target_gxy "${target_gxy}" \
        -var target_gxz "${target_gxz}" \
        -var target_gyz "${target_gyz}" \
        -var current_exx "${current_exx}" \
        -var current_eyy "${current_eyy}" \
        -var current_ezz "${current_ezz}" \
        -var current_gxy "${current_gxy}" \
        -var current_gxz "${current_gxz}" \
        -var current_gyz "${current_gyz}" \
        -var current_step "${current_step}" \
        -var current_strain "${current_strain}" \
        -var DUMPFILE "${DUMPFILE_PATH}"
}

# Semaphore function to limit parallel jobs
run_with_limit() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
    run_lammps "$1" "$2" &
}

# Main loop
for f in "${FILES[@]}"; do
    echo "==============================================================================="
    echo "Processing $f"

    prime=$(get_prime.php ${PRIME_INDEX})
    ((PRIME_INDEX++))

    echo "Using seed $prime"

    run_with_limit "$f" "$prime"
done

# Wait for all background jobs to finish
wait
echo "All simulations completed."