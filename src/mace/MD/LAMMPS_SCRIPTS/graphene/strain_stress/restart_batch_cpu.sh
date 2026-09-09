#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPSIN="${SCRIPT_DIR}/restart_strain_stress.in"

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

export OMP_NUM_THREADS=2
MAX_JOBS=2   # Max concurrent LAMMPS runs

# Function to run a single simulation
run_lammps() {
    local f="$1"
    local prime="$2"

    local file=$(basename "$f")
    local name="${file%.lammps-data}"

    local LOGFILE="${OUTPUTDIR}/${name}.log"
    local DUMPFILE_PATH="${OUTPUTDIR}/${DUMPDIR}/${name}.dump"

    mkdir -p "${OUTPUTDIR}/${DUMPDIR}"

    lmp_cpu \
        -log "${LOGFILE}" \
        -in "${LAMMPSIN}" \
        -var INPUT_CONFIG "$f" \
        -var MODEL "${MODEL}" \
        -var TEMP "${TEMP}" \
        -var SEED "${prime}" \
        -var NSTRAINS "${NSTRAINS}" \
        -var timestep "${timestep}" \
        -var tdamp "${tdamp}" \
        -var nsteps "${nsteps}" \
        -var max_exx "${max_exx}" \
        -var max_eyy "${max_eyy}" \
        -var max_ezz "${max_ezz}" \
        -var max_gxy "${max_gxy}" \
        -var max_gxz "${max_gxz}" \
        -var max_gyz "${max_gyz}" \
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
