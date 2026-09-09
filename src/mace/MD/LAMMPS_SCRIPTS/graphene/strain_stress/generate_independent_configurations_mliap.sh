#!/bin/bash
# generate N independent configurations until N valid ones are found

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPSIN="${SCRIPT_DIR}/change_strain_mliap.in"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
CHECKER="$PROJECT_ROOT/src/structure/broken.py"

declare -i N=0
INFILE=''
OUTDIR=''
RUNCONF=''

print_usage() {
    printf "Usage: $0 --nconfigs <n> --infile <file> --outdir <dir> --runconf <conf>\n"
}

OPTS=$(getopt -o n:i:o:r: --long nconfigs:,infile:,outdir:,runconf: -n 'script' -- "$@")
eval set -- "$OPTS"

while true; do
    case "$1" in
        -n|--nconfigs) N="$2"; shift 2 ;;
        -i|--infile) INFILE="$2"; shift 2;;
        -o|--outdir) OUTDIR="$2"; shift 2 ;;
        -r|--runconf) RUNCONF="$2"; shift 2 ;;
        --) shift; break ;;
        *) print_usage; exit 1 ;;
    esac
done

mkdir -p "${OUTDIR}"
source "${RUNCONF}"

declare -i COUNT=0
declare -i ATTEMPT=0
declare -i PRIME_INDEX=100

echo "Starting generation of $N valid configurations..."

while [ $COUNT -lt $N ]; do
    ATTEMPT=$((ATTEMPT + 1))
    
    prime=$(get_prime.php ${PRIME_INDEX})
    ((PRIME_INDEX++))

    # Use absolute paths to avoid any ambiguity
    NAME="${COUNT}"
    LOGFILE="$(realpath "${OUTDIR}")/${NAME}.log"
    DUMPFILE_PATH="$(realpath "${OUTDIR}")/${DUMPDIR}/${NAME}.dump"
    OUTFILE_DATA="$(realpath "${OUTDIR}")/${NAME}.lammps-data"

    echo ">>> Attempt $ATTEMPT | Seed: $prime"

    echo NAME $NAME
    echo LOGFILE $LOGFILE
    echo DUMPFILE_PATH $DUMPFILE_PATH
    echo OUTFILE_DATA $OUTFILE_DATA

    # Call lmp with clear variable assignments
    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -log "${LOGFILE}" \
        -in "${LAMMPSIN}" \
        -var INPUT_CONFIG "$INFILE" \
        -var MODEL "${MODEL}" \
        -var SEED "${prime}" \
        -var DUMPFILE "${DUMPFILE_PATH}" \
        -var OUTFILE "${OUTFILE_DATA}" \
        -var TEMP "${TEMP}" \
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
        -var current_strain "${current_strain}"

    sed -i 's/atomic\/kk/atomic/g' "${OUTFILE_DATA}"
    # Check if the file actually exists and isn't literally named '${OUTFILE}'
    if [ -f "${OUTFILE_DATA}" ] && python "$CHECKER" "${OUTFILE_DATA}"; then
        echo "Successfully generated $OUTFILE_DATA"
        COUNT=$((COUNT + 1))
    else
        echo "Failed or Broken. Cleaning up..."
        #rm -f "${LOGFILE}" "${DUMPFILE_PATH}" "${OUTFILE_DATA}"
        # Also cleanup the literal file if LAMMPS misbehaved
        #rm -f '${OUTFILE}' 
    fi
done

echo "==============================================================================="
echo "Completed! Generated $N valid configurations in $ATTEMPT total attempts."