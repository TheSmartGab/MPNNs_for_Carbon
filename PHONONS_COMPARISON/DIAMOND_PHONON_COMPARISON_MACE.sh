#!/bin/bash
# =====================================
# Script Name: DIAMOND_PHONON_COMPARISON.sh
# Purpose: Compare phonon dispersion calculations for diamond across multiple MACE models and DFT reference data
# Usage: Run the script to generate comparison plots of phonon band structures
#
# Models compared:
# - from_C2_v1 (GAP2020-trained on C2)
# - from_C2_cut (cut version)
# - mace-GAP2020_FROM_SCRATCH_smaller (various versions v2-v5)
# - DFT-PBEsol reference (Materials Project mp-66_phonon_bs_mp.json)
# Output: PHONONS_COMPARISON/17_06/diamond/band_comparison_plots
# =====================================

python3 src/phonons/plot_phonons.py \
	--sources \
		../mace_models_comparison/from_C2_v1/phonons_diamond/band.json \
		../mace_models_comparison/from_C2_cut/phonons_diamond/band.json \
		../mace_models_comparison/mace-GAP2020_FROM_SCRATCH_smaller/phonons_diamond/band.json \
		../mace_models_comparison/mace-GAP2020_FROM_SCRATCH_smaller2_l2/phonons_diamond/band.json \
		../mace_models_comparison/mace-GAP2020-FROM_SCRATCH_v3/phonons_diamond/band.json \
		../mace_models_comparison/mace-GAP2020-FROM_SCRATCH_v4/phonons_diamond/band.json \
		../mace_models_comparison/v5/phonons_diamond/band.json \
		../EXTERNAL_DATA/diamond_phonons/mp-66_phonon_bs_mp.json \
	--formats \
		ase \
		ase \
		ase \
		ase \
		ase \
		ase \
		ase \
		mp \
	--labels \
		v1 \
		v1-cut \
		v2 \
		v2-l2 \
		v3 \
		v4 \
		v5 \
		DFT-PBEsol \
	--output_dir PHONONS_COMPARISON/17_06/diamond \
	--kpath GxwkGluwlk
