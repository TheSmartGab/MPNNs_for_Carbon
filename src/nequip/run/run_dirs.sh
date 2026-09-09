#!/bin/bash

DIR0=$(pwd)

target=$1 # root directory with some CONFIG_NAME files somewhere further down the path. 
target=$(realpath "$target")

CONFIG_NAME="config.yaml"

# safer array fill
mapfile -t configs < <(find "$target" -type f -name "$CONFIG_NAME")

# activate conda environment
source ~/SetConda.sh
conda activate nequip-allegro

# loop over files
for config in "${configs[@]}"; do
    dir="$(dirname "$config")"
    echo "==========================================="
    echo "processing $dir"
    echo "==========================================="

    cd "$dir" || exit 1
    echo using config file "$(realpath $CONFIG_NAME)"

    nequip-train -cn "$CONFIG_NAME"

    # return to working directory
    cd "$DIR0" || exit 1
done

cd "$DIR0"
echo "[DONE] runs_dirs.sh completed"

