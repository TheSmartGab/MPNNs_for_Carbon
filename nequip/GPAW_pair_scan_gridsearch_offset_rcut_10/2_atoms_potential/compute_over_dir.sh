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

DIR0=$(pwd)
# execute from PROJECT directory for this to work
POT_SRC_DIR=$(realpath "$DIR0/src/nequip/2_atoms_potential")
COMPILER=$(realpath "$DIR0/src/nequip/compile/compile_and_deliver.sh")

echo =====================================
echo Using potential source dir:
echo $POT_SRC_DIR
echo COMPILER:
echo $COMPILER
echo =====================================

# set cona envirorment
source ~/SetConda.sh
conda activate nequip-allegro

target=$1
# elements
e1=$2
e2=$3
# reference dir to plot training, validation and test data
ref_split=$4
ref_extension=$5
ref_format=$6

target=$(realpath "${target}")

checpoints=($(find $target -type f -name "best.ckpt"))


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
    # since I started using relative paths in the configs
    # I actually have to compile from the same directory from which
    # the model was trained
    # the following is not robust, but works for the chosen directory
    # structure
    cd $chpt_dir
    cd ..
    mv $chpt_dir/best.ckpt . 
    source $COMPILER
    mv *.pth *.ckpt $chpt_dir

    # actually process the given directory
    cd $DIR0
    # source $POT_SRC_DIR/compute_pot.sh "${pot_dir}" "${chpt_dir}/deployed_pair.nequip.pth" # legacy using LAMMPS
    # a bit more confortable using ase with nequip calculator
    python "${pot_dir}/compute_pot.py" \
    --model_path "${chpt_dir}/deployed_pair.nequip.pth" \
    --elements $e1 $e2 \
    --device cpu \
    --output_dir "${pot_dir}" \
    --ref_split "${ref_split}" \
    --ref_extension ${ref_extension} \
    --ref_format ${ref_format}

    done

echo ["[DONE] compute_over_dir.sh"]
