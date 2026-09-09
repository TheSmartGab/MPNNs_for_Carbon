#!/bin/bash

# I added the script nequip-eval to my nequip installation
# this script is used to compare the models in a given directory against a reference to find the error as a function of interatomic distance

# expected tree of target directories:
# .
# ├── 2_atoms_potential
# │   ├── INPUT
# │   │   └── compute_pot.in
# │   ├── OUTPUT
# │   │   ├── energy_vs_distance.txt
# │   │   ├── info.txt
# │   │   └── potential_plot.pdf
# │   ├── WORKING
# │   │   └── log.lammps
# │   ├── compute_over_dir.sh
# │   ├── compute_pot.py
# │   ├── compute_pot.sh
# │   ├── force.pdf
# │   ├── info.out
# │   ├── model_penergy_forces.out
# │   ├── plot.py
# │   ├── plot_merge.py
# │   └── potential_energy.pdf
# ├── checkpoints
# │   ├── best-v1.ckpt
# │   ├── best.ckpt
# │   ├── deployed_pair.nequip.pth
# │   ├── last-v1.ckpt
# │   ├── last.ckpt
# │   └── packaged_model.nequip.zip
# ├── config.yaml
# ├── eval
# │   └── validation
# │       ├── energy_error_vs_distance.pdf
# │       ├── force_error_vs_distance.pdf
# │       └── validation_model.extxyz
# ├── logger
# │   └── version_0
# │       ├── hparams.yaml
# │       ├── learning_rate.pdf
# │       ├── metrics.csv
# │       ├── test_results.json
# │       └── training_metrics.pdf
# └── outputs
#     └── 2026-01-06
#         └── 10-02-24
#             └── train.log

DIR0=$(pwd)
DIFF_FUNCTION_DISTANCE=$(realpath "src/compare_outputs/diff_function_distance.py") # works from PROJECT dir
# this just compares 2 extxyz files, first you must evaluate the differences using nequip-eval

target=$1
ref_file=$2   # reference file. eg: validation file from the split used for training 
output_dir=$3 # output ir for nequip-eval. relative path from the 2_atoms_potential dir. eg: eval/validation/ if reference is the validation set
output_filename=$4 # output filename for nequip-eval. eg: validation_model.extxyz

models=($(find $target -type f -name "deployed_pair.nequip.pth")) 

for model in "${models[@]}"; do
    cd $DIR0
    echo ======================================
    echo processing model:
    echo $model
    echo ======================================

    model_dir=$(dirname $(dirname "${model}"))
    pot_dir="${model_dir}/2_atoms_potential"
    cd "${pot_dir}"

    # run eval script
    nequip-eval --model_path "$model" --configurations "$ref_file" --device cpu --format extxyz --output_file $output_dir/$output_filename

    python $DIFF_FUNCTION_DISTANCE $ref_file $output_dir/$output_filename $output_dir 

done
