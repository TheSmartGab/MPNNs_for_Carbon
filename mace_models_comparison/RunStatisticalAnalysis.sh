#!/bin/bash

# this script was used to evaluate to evaluate training an validation configurations with the different models over the given dirs.
# Then, error distribution and summary metrics were computed, and the results plotted.
# This script does not work out-of-the-box, since the datasets are needed, and they are not included in this PROJECT.
# This script can be easily adapted to fit your needs.


PROJECT_ROOT="$(git rev-parse --show-toplevel)"

dirs=("v5training")

for d in ${dirs[@]};
do
	mkdir -p $d/eval/train $d/eval/val

	# eval configurations
	echo running directory $d
	mace_eval_configs --configs $PROJECT_ROOT/DATASETS/C/GAP2020/config_types/VAL.extxyz --model $d/chosenMACE.model --output $d/eval/val/model_val.extxyz --device cuda --default_dtype float32 --batch_size 10 --compute_stress --return_descriptors --info_prefix ""
	mace_eval_configs --configs $PROJECT_ROOT/DATASETS/C/GAP2020/config_types/TRAIN.extxyz --model $d/chosenMACE.model --output $d/eval/train/model_train.extxyz --device cuda --default_dtype float32 --batch_size 10 --compute_stress --return_descriptors --info_prefix ""
	# run error analysis
	python $PROJECT_ROOT/src/compare_outputs/diff_distributions.py --ref $PROJECT_ROOT/DATASETS/C/GAP2020/config_types/TRAIN.extxyz $PROJECT_ROOT/DATASETS/C/GAP2020/config_types/VAL.extxyz --model $d/eval/train/model_train.extxyz $d/eval/val/model_val.extxyz --labels train validation --output_dir $d/eval/new_new_global --discard_comp --logscale --E0 1.02433787 --image_format pdf --config_type config_type

done
