#!/bin/bash

# Define the array of training set sizes
NUMTRAIN=(50 100 200 400 800 1600 3200 4266)

run(){
    TRAINSIZE=$1
    TRAINFILE="/home/gabri/Thesis/PROJECT/DATASETS/C/GAP2020/config_types/NEW_SPLIT/train_${TRAINSIZE}.xyz"

    mkdir -p "$TRAINSIZE"
    cd "$TRAINSIZE" || exit

    # --- Start GPU Logging ---
    # Logs timestamp, memory used (MiB), and GPU utilization (%) every 1 second
    LOGFILE="gpu_usage_${TRAINSIZE}.log"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$LOGFILE" &
    GPU_LOG_PID=$! # Capture the PID of the logger
    # -------------------------

    echo "Starting MACE training for size: $TRAINSIZE"

    python -m mace.cli.run_train \
        --name="MACE_${TRAINSIZE}" \
        --atomic_numbers="[6]" \
        --train_file="${TRAINFILE}" \
        --valid_file="/home/gabri/Thesis/PROJECT/DATASETS/C/GAP2020/config_types/NEW_SPLIT/val.xyz" \
        --test_file="/home/gabri/Thesis/PROJECT/DATASETS/C/GAP2020/config_types/NEW_SPLIT/test.xyz" \
        --compute_stress=True \
        --energy_weight=1.0 \
        --forces_weight=100.0 \
        --stress_weight=10.0 \
        --E0s="{6: 1.02433787}" \
        --lr=0.001 \
        --scaling="rms_forces_scaling" \
        --batch_size=4 \
        --valid_batch_size=8 \
        --num_workers=2 \
        --max_num_epochs=1000 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --scheduler="ReduceLROnPlateau" \
        --lr_factor=0.5 \
        --scheduler_patience=20 \
        --patience=6 \
        --clip_grad=10 \
        --default_dtype="float32" \
        --device=cuda \
        --seed=1009 \
        --energy_key="energy" \
        --forces_key="forces" \
        --stress_key="stress" \
        --pair_repulsion \
        --loss='stress' \
        --num_interactions=2 \
        --radial_MLP="[64,64,64]" \
        --num_radial_basis=10 \
        --r_max=6 \
        --num_channels=128 \
        --edge_irreps="16x0e + 16x1o + 16x2e + 16x3o" \
        --hidden_irreps="128x0e + 128x1o" \
        --correlation=3 \
        --swa \
        --start_swa=980 \
	--swa_lr=0.001 \
        --swa_energy_weight=1000 \
        --swa_forces_weight=100 \
        --swa_stress_weight=100 \
        --eval_interval 5

    # --- Stop GPU Logging ---
    kill $GPU_LOG_PID
    echo "Training for $TRAINSIZE finished. GPU log saved to $TRAINSIZE/$LOGFILE"
    # ------------------------

    cd ..
}

for NT in "${NUMTRAIN[@]}"; do
    run "$NT"
done
