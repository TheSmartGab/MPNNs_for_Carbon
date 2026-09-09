#!/bin/bash
# thermalize an initial configiuration to different temperatures.
# Note that this may not be ideal if your temperature range is very large.
# in that case, you would run serially thermalizations at different temperatures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPSIN="${SCRIPT_DIR}/thermalize_mliap.in"

RUNCONF='' # this file contains settings common to all runs, such as model number of steps, tdamp, timestep
TEMP_FILE='' # file containing temperatures to run
INPUTCONFIG='' # input configuration. should be a relaxed configuration
NAME="thermalization"
OUTNAME="thermalized"
DUMPDIR=DUMP

print_usage() {
  printf "Usage: $0 --input_dir <dir> --output_dir <dir> --runconf <conf>\n"
}

OPTS=$(getopt -o i:r:t: --long input_dir:,runconf:,tempfile: -n 'script' -- "$@")
eval set -- "$OPTS"

while true; do
  case "$1" in
    -i|--input_file) INPUTCONFIG="$2"; shift 2 ;;
    -r|--runconf) RUNCONF="$2"; shift 2 ;;
    -t|--tempfile) TEMP_FILE="$2"; shift 2 ;;
    --) shift; break ;;
    *) print_usage; exit 1 ;;
  esac
done

echo ========================================================
echo "INPUTCONFIG ${INPUTCONFIG}"
echo "RUNCONF ${RUNCONF}"
echo "TEMP_FILE ${TEMP_FILE}"
echo ========================================================


# Load simulation parameters
source "${RUNCONF}"

# read temperatures from file
TEMPS=($(cat "${TEMP_FILE}"))


declare -i MIN_PRIME_INDEX=100
PRIME_INDEX=${MIN_PRIME_INDEX}

export OMP_NUM_THREADS=4
MAX_JOBS=4   # Max concurrent LAMMPS runs

# Function to run a single simulation
run_lammps() {
    local T="$1"
    local prime="$2"

    local OUTPUTDIR="${T}K/THERMALIZATION"
    mkdir -p "${OUTPUTDIR}"

    local filenames="${NAME}_${T}K"

    local LOGFILE="${OUTPUTDIR}/${filenames}.log"
    local DUMPFILE_PATH="${OUTPUTDIR}/${DUMPDIR}/${filenames}.dump"
    local OUT_CONFIG="${OUTPUTDIR}/${OUTNAME}_${T}K.lammps-data"

    mkdir -p "${OUTPUTDIR}/${DUMPDIR}"

    echo ========================================================
    echo LOGFILE "${LOGFILE}"
    echo "${LAMMPSIN}"
    echo INPUT_CONFIG "${INPUTCONFIG}"
    echo MODEL "${MODEL}"
    echo TEMP "${T}"
    echo SEED "${prime}"
    echo timestep "${timestep}"
    echo tdamp "${tdamp}"
    echo nsteps "${nsteps}"
    echo DUMPFILE "${DUMPFILE_PATH}"
    echo OUT_CONFIG "${OUT_CONFIG}"
    echo ========================================================


    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -log "${LOGFILE}" \
        -in "${LAMMPSIN}" \
        -var INPUT_CONFIG "${INPUTCONFIG}" \
        -var MODEL "${MODEL}" \
        -var TEMP "${T}" \
        -var SEED "${prime}" \
        -var timestep "${timestep}" \
        -var tdamp "${tdamp}" \
        -var nsteps "${nsteps}" \
        -var DUMPFILE "${DUMPFILE_PATH}" \
        -var OUT_CONFIG "${OUT_CONFIG}"
}

# Semaphore function to limit parallel jobs
run_with_limit() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 10
    done
    run_lammps "$1" "$2" &
}

# Main loop
for T in "${TEMPS[@]}"; do
    echo "==============================================================================="
    echo "Processing $T"

    prime=$(get_prime.php ${PRIME_INDEX})
    ((PRIME_INDEX++))

    echo "Using seed $prime"

    run_with_limit "$T" "$prime"
done

# Wait for all background jobs to finish
wait
echo "All simulations completed."