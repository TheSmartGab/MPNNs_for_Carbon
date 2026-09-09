#!/bin/bash

SRC_DIR="/home/gabri/Thesis/PROJECT/src/2_atoms_potential"
SRC_DIRNAME="2_atoms_potential"
EXEC_NAME="compute_pot.sh"
COMPILER="/home/gabri/Thesis/PROJECT/src/compile/compile_and_deliver.sh"
DEPLOYIED_NAME="deployed_pair.nequip.pth"

dirs=($(echo checkpoint_v*))
N=${#dirs[@]}

DIR0=$(pwd)

for i in $(seq 1 $N); do
    index=$(($i-1))
    dir=$(realpath ${dirs[index]})  # corrected index
    model=$dir                      # fixed variable
    version_dir="v$index"

    if [ ! -d "$version_dir" ]; then
        echo "processing $version_dir"

        # compile and deliver
        cd "$dir"
        source "$COMPILER"
        cd "$DIR0"

        # create version directory and run script
        mkdir "$version_dir"
        cp -r "$SRC_DIR" "$version_dir"
        cd "$version_dir/$SRC_DIRNAME"
        source "$EXEC_NAME" . "${dir}/${DEPLOYIED_NAME}"

        cd "$DIR0"
    fi

done   # ← REQUIRED

echo "[DONE]"

