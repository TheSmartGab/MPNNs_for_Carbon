echo "Directory must be organized as: 
target/
├── config_runs.yaml
├── hparams.yaml
├── v0
│   ├── checkpoints
│   │   ├── best.ckpt
│   │   └── last.ckpt
│   ├── config.yaml
│   ├── logger
│   │   └── version_0
│   │       ├── hparams.yaml
│   │       ├── learning_rate.pdf
│   │       ├── metrics.csv
│   │       ├── test_results.json
│   │       └── training_metrics.pdf
│   └── outputs
│       └── 2025-11-21
│           └── 15-39-29
│               └── train.log
├── v1
│   ├── checkpoints
│   │   ├── best.ckpt
│   │   └── last.ckpt
│   ├── config.yaml
│   ├── logger
│   │   └── version_0
│   │       ├── hparams.yaml
│   │       ├── learning_rate.pdf
│   │       ├── metrics.csv
│   │       ├── test_results.json
│   │       └── training_metrics.pdf
│   └── outputs
│       └── 2025-11-21
│           └── 16-17-40
│               └── train.log

i.e.: model dirs that have checkpoints subdir. model dirs will host 2_atoms_potential dir that will be copied there.
"

# execute from PROJECT directory for this to work
POT_SRC_DIR=$(realpath "$DIR0/src/2_atoms_potential")
COMPILER=$(realpath "$DIR0/src/compile/compile_and_deliver.sh")

echo =====================================
echo Using potential source dir:
echo $POT_SRC_DIR
echo COMPILER:
echo $COMPILER
echo =====================================

target=$1

target=$(realpath "${target}")

checpoints=($(find $target -type f -name "best.ckpt"))

DIR0=$(pwd)

for chpt in "${checpoints[@]}"; do
    echo ======================================
    echo processing checkpoint:
    echo $chpt
    echo ======================================

    chpt_dir=$(dirname "${chpt}")
    model_dir=$(dirname "${chpt_dir}")
    pot_dir="${model_dir}/2_atoms_potential"

    cp -r "${POT_SRC_DIR}" $pot_dir

    # compile the model
    cd $chpt_dir
    source $COMPILER

    # actually process the given didrectory
    cd $DIR0
    source $POT_SRC_DIR/compute_pot.sh "${pot_dir}" "${chpt_dir}/deployed_pair.nequip.pth"


    done

echo ["[DONE] compute_over_dir.sh"]