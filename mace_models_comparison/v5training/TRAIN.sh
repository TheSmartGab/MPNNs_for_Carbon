#!/bin/bash
# =====================================
# Script Name: TRAIN.sh (v5 MACE Training)
# Purpose: Train MACE model on carbon (GAP2020 dataset) with stress and energy loss configuration
# Usage: Run from KAGGLE_DOWNLOAD/v5training/ directory to initiate multi-GPU MACE training run
#
# Training Configuration:
# - Dataset: DATASETS/C/GAP2020/config_types/TRAIN.extxyz / VAL.extxyz (Carbon only, atomic_number=6)
# - Loss function: stress loss with energy_weight=1.0, forces_weight=10.0, stress_weight=100.0
# - Optimization: Adam with AMSGrad, ReduceLROnPlateau scheduler, 100 max epochs
# - SWA (Stochastic Weight Averaging): Starts at epoch 80, lr=0.001, forces_weight=100, energy_weight=1000
# - EMA (Exponential Moving Average): decay=0.99
# =====================================

CUDA_VISIBLE_DEVICES=0,1,2,3,4 mace_run_train \
  --name="MACE" \
  --atomic_numbers="[6]" \
  --train_file="../DATASETS/C/GAP2020/config_types/TRAIN.extxyz" \
  --valid_file="../DATASETS/C/GAP2020/config_types/VAL.extxyz" \
  --compute_stress=True \
  --energy_weight=1.0 \
  --forces_weight=10.0 \
  --stress_weight=100.0 \
  --E0s="{6: 1.02433787}"\
  --lr=0.005 \
  --scaling="rms_forces_scaling" \
  --batch_size=8 \
  --valid_batch_size=8 \
  --num_workers=2 \
  --max_num_epochs=100 \
  --swa \
  --start_swa 80 \
  --swa_lr 0.001 \
  --swa_forces_weight 100 \
  --swa_energy_weight 1000 \
  --swa_stress_weight 10 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --scheduler="ReduceLROnPlateau" \
  --lr_factor=0.8 \
  --scheduler_patience=5 \
  --patience=25 \
  --clip_grad=10 \
  --default_dtype="float32" \
  --device=cuda \
  --enable_cueq=True \
  --distributed \
  --seed=1009 \
  --energy_key="energy" \
  --forces_key="forces" \
  --stress_key="stress" \
  --pair_repulsion \
  --loss='stress' \
  --num_interactions=2 \
  --radial_MLP="[16, 16]" \
  --num_radial_basis=10 \
  --r_max=6 \
  --num_channels=16 \
  --edge_irreps="16x0e + 16x1o + 16x2e + 16x3o + 16x4e" \
  --hidden_irreps="16x0e + 16x1o + 16x2e" \
  --correlation=3 \
  --weight_decay=0
