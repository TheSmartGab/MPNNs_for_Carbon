#!/bin/bash

# This code aims to automate the restart process for the breaking rate studies of graphene
# it is convenient to periodiclaly check whether a structure is broken, and stop simulating it 
# I have not found a way of doing this in lammps
# dump2traj.py converts dump files to .traj ase binary files. usefull since I will use this for analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/change_strain_mliap.sh" # run script
DUMP2TRAJ="${SCRIPT_DIR}/../../../../../structure/dump2traj.py"
FINDBROKEN="${SCRIPT_DIR}/../../../../../structure/find_broken_traj.py"

#####################
# PARAMETERS
declare -i NMAX=$1

INPUT_TEMPLATE=$2
OUTPUT_TEMPLATE=$3

RUNCONF0=$4 # this is used to change strain
RUNCONF1=$5 # this keeps strain the same in subsequent runs
######################

######################
# other script variables
INPUTDIR=""
OUTPUTDIR=""
declare -i n=0 # overwritten in loop
declare -i nextn=$((n+1))
######################

INPUTDIR="${INPUT_TEMPLATE}${n}"
OUTPUTDIR="${OUTPUT_TEMPLATE}${n}"
mkdir -p "${OUTPUTDIR}"
nextn=$((n+1))
mkdir -p "${INPUT_TEMPLATE}${nextn}"


source "${RUN}" -i "${INPUTDIR}" -o "${OUTPUTDIR}" -r "${RUNCONF0}"
python "${DUMP2TRAJ}" "${OUTPUTDIR}/DUMP"
python "${FINDBROKEN}" -i "${OUTPUTDIR}/DUMP" -o "${INPUT_TEMPLATE}${nextn}"

for n in $(seq 1 $NMAX);
do
    # update variables
    INPUTDIR="${INPUT_TEMPLATE}${n}"
    OUTPUTDIR="${OUTPUT_TEMPLATE}${n}"
    mkdir -p "${OUTPUTDIR}"
    nextn=$((n+1))
    mkdir -p "${INPUT_TEMPLATE}${nextn}"

    source "${RUN}" -i "${INPUTDIR}" -o "${OUTPUTDIR}" -r "${RUNCONF1}"
    echo "calling python" "${DUMP2TRAJ}" "${OUTPUTDIR}/DUMP"
    python "${DUMP2TRAJ}" "${OUTPUTDIR}/DUMP"
    python "${FINDBROKEN}" -i "${OUTPUTDIR}/DUMP" -o "${INPUT_TEMPLATE}${nextn}"
done