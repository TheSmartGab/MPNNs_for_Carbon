#!/bin/bash
# thermalize an initial configiuration to different temperatures.
# Note that this may not be ideal if your temperature range is very large.
# in that case, you would run serially thermalizations at different temperatures
# this script was thought to restart from thermalized configurations 


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPSIN="${SCRIPT_DIR}/strain_stress_mliap.in"

RUNCONF='' # this file contains settings common to all runs, such as model number of steps, tdamp, timestep
FILESLIST="" # file with list of files to use as starting configuration
OUTPUTSFILE="" # file with outputdirs. WARNING: should be parallel to filelists
TEMPFILE="" # file with temperatures. this layout is somewhat redundant, since input and output directories could be inferred from just the temperature ... 
NAME="strainstress"
DUMPDIR="DUMP"

print_usage() {
  printf "Usage: $0 --filelist <file> --outputs <file> --tempfile <file> --runconf <runconf>\n"
}

OPTS=$(getopt -o f:o:r:t: --long filelist:,outputs:,runconf:,tempfile: -- "$@")
eval set -- "$OPTS"

while true; do
  case "$1" in
    -f|--filelist) FILESLIST="$2"; shift 2 ;;
    -o|--outputs) OUTPUTSFILE="$2"; shift 2;;
    -t|--tempfile) TEMPFILE="$2"; shift 2;;
    -r|--runconf) RUNCONF="$2"; shift 2 ;;
    --) shift; break ;;
    *) print_usage; exit 1 ;;
  esac
done

echo ========================================================
echo "FILELIST ${FILELIST}"
echo "OUTPUTSFILE ${OUTPUTSFILE}"
echo "TEMPFILE ${TEMPFILE}"
echo "RUNCONF ${RUNCONF}"
echo ========================================================

# Load simulation parameters
source "${RUNCONF}"

# read temperatures from file
FILES=($(cat "${FILESLIST}"))
OUTPUTS=($(cat "${OUTPUTSFILE}"))
TEMPERATURES=($(cat "${TEMPFILE}"))

# check nfiles == nouts
nfiles=${#FILES[@]}
nouts=${#OUTPUTS[@]}
ntemps=${#TEMPERATURES[@]}

if (( nfiles != nouts )); then 
    echo "[ERROR] nfiles $nfiles is not equal to nouts $nouts"
    echo "check FILELIST and OUTPUTSFILE, they should have the same number of entries"
    exit 1
fi 

if (( nfiles != ntemps )); then 
    echo "[ERROR] nfiles $nfiles is not equal to ntemps $ntemps"
    echo "check FILELIST and OUTPUTSFILE, they should have the same number of entries"
    exit 1
fi 

declare -i MIN_PRIME_INDEX=100
PRIME_INDEX=${MIN_PRIME_INDEX}

export OMP_NUM_THREADS=4
MAX_JOBS=4   # Max concurrent LAMMPS runs

# Function to run a single simulation
run_lammps() {
    local F="$1" # input file
    local O="$2" # output directory
    local T="$3" # temperature
    local prime="$4"

    local OUTPUTDIR="${O}/STRAINSTRESS"
    mkdir -p "${OUTPUTDIR}"

    local LOGFILE="${OUTPUTDIR}/${NAME}.log"
    local DUMPFILE_PATH="${OUTPUTDIR}/${DUMPDIR}/${NAME}.dump"

    mkdir -p "${OUTPUTDIR}/${DUMPDIR}"

    echo ========================================================
    echo LOGFILE "${LOGFILE}"
    echo "${LAMMPSIN}"
    echo INPUT_CONFIG "${F}"
    echo MODEL "${MODEL}"
    echo TEMP "${T}"
    echo SEED "${prime}"
    echo timestep "${timestep}"
    echo tdamp "${tdamp}"
    echo nsteps "${nsteps}"
    echo DUMPFILE "${DUMPFILE_PATH}"
    echo ========================================================

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -log "${LOGFILE}" \
        -in "${LAMMPSIN}" \
        -var INPUT_CONFIG "${F}" \
        -var MODEL "${MODEL}" \
        -var TEMP "${T}" \
        -var SEED "${prime}" \
        -var timestep "${timestep}" \
        -var tdamp "${tdamp}" \
        -var nsteps "${nsteps}" \
        -var DUMPFILE "${DUMPFILE_PATH}" \
        -var max_exx "${max_exx}" \
        -var max_eyy "${max_eyy}" \
        -var max_ezz "${max_ezz}" \
        -var max_gyz "${max_gyz}" \
        -var max_gxz "${max_gxz}" \
        -var max_gxy "${max_gxy}" \
        -var NSTRAINS "${NSTRAINS}"
}

# Semaphore function to limit parallel jobs
run_with_limit() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 10
    done
    run_lammps "$1" "$2" "$3" "$4" &
}

# Main loop
for i in $(seq 0 $(($nfiles-1))); do
    FILE="${FILES[$i]}"
    OUT="${OUTPUTS[$i]}"
    T="${TEMPERATURES[$i]}"
    echo "==============================================================================="
    echo "Processing $T"

    prime=$(get_prime.php ${PRIME_INDEX})
    ((PRIME_INDEX++))

    echo "Using seed $prime"

    run_with_limit "${FILE}" "${OUT}" "${T}" "${prime}"
done

# Wait for all background jobs to finish
wait
echo "All simulations completed."