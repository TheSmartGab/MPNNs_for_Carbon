#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

source ~/.bashrc

# --- BASE CONFIGURATIONS ---
BASE_PROJECT="$(git rev-parse --show-toplevel)"
EVAL_SCRIPT="$BASE_PROJECT/src/nequip/eval_configs.py"
DIFF_SCRIPT="$BASE_PROJECT/src/compare_outputs/diff_distributions.py"

# Reference Dataset Paths
TRAIN_REF="$BASE_PROJECT/DATASETS/C/nomad/GRAPHENE_MLIP/all_100frames/split/training/training.xyz"
VAL_REF="$BASE_PROJECT/DATASETS/C/nomad/GRAPHENE_MLIP/all_100frames/split/validation/validation.xyz"

echo "========================================================================="
echo " Starting MLIP Batch Evaluation (Deduplicated - Last Checkpoint Only)"
echo "========================================================================="

# Declare an associative array to store the best checkpoint per model directory
declare -A LATEST_CHECKPOINTS

# 1. Discover and filter checkpoints
# Sorting alphabetically ensures 'checkpoint_v2/best_v2.ckpt' comes after 'checkpoint/best.ckpt'
while read -r CKPT_PATH; do
    # Extract the top-level folder name (e.g., model_T3_b8f_l1_pT_f16_rd1_rw16)
    MODEL_DIR=$(echo "$CKPT_PATH" | cut -d'/' -f2)
    
    # Overwrite the entry. Because find/sort lists v1, v2 sequentially, 
    # the highest version variant will automatically be saved last.
    LATEST_CHECKPOINTS["$MODEL_DIR"]="$CKPT_PATH"
done < <(find . -type f -name "best*" | sort)

# 2. Iterate through the deduplicated list
for MODEL_DIR in "${!LATEST_CHECKPOINTS[@]}"; do
    CKPT_PATH="${LATEST_CHECKPOINTS[$MODEL_DIR]}"
    
    echo "------------------------------------------------------------------------"
    echo "Targeting Unique Model: $MODEL_DIR"
    echo "Using Checkpoint File:  $CKPT_PATH"
    
    # Resolve absolute destination directories for evaluations
    MODEL_ROOT="$BASE_PROJECT/nequip/GRAPHENE_MLIP/all_100frames/$MODEL_DIR"
    EVAL_TRAIN_DIR="$MODEL_ROOT/eval/train"
    EVAL_VAL_DIR="$MODEL_ROOT/eval/val"
    GLOBAL_OUT_DIR="$MODEL_ROOT/eval/global"
    
    # Create the output directories if they don't exist
    mkdir -p "$EVAL_TRAIN_DIR" "$EVAL_VAL_DIR" "$GLOBAL_OUT_DIR"
    
    # Define exact output files
    TRAIN_OUT="$EVAL_TRAIN_DIR/model_train.extxyz"
    VAL_OUT="$EVAL_VAL_DIR/model_val.extxyz"
    
    # 3. Run Evaluation for Training Dataset
    conda activate nequip
    echo " -> Evaluating training configurations..."
    python "$EVAL_SCRIPT" \
        -i "$TRAIN_REF" \
        -c "$CKPT_PATH" \
        -o "$TRAIN_OUT" \
        -d cpu
        
    # 4. Run Evaluation for Validation Dataset
    echo " -> Evaluating validation configurations..."
    python "$EVAL_SCRIPT" \
        -i "$VAL_REF" \
        -c "$CKPT_PATH" \
        -o "$VAL_OUT" \
        -d cpu
        
    # 5. Run Distribution Comparison & Plotting
    echo " -> Generating comparison distribution plots..."
    conda deactivate
    source $BASE_PROJECT/.venv/bin/activate
    python "$DIFF_SCRIPT" \
        --ref "$TRAIN_REF" "$VAL_REF" \
        --model "$TRAIN_OUT" "$VAL_OUT" \
        --labels train validation \
        --output_dir "$GLOBAL_OUT_DIR" \
        --discard_comp \
        --logscale \
        --E0 1.02433787 \
        --image_format pdf
    deactivate
    echo "Finished pipeline iteration successfully for $MODEL_DIR."
done

echo "========================================================================="
echo " Complete batch pipeline terminated successfully!"
echo "========================================================================="
