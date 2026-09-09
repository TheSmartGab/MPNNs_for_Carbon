#!/bin/bash
# =====================================
# Script Name: GRAPHENE_PHONON_COMPARISON_NEQUIP.sh
# Purpose: Compare phonon dispersion calculations for graphene across multiple NequIP models and DFT reference data
# Usage: Run the script to generate comparison plots of graphene phonon band structures from NequIP L0, L1, L2 variants
#
# Models compared:
# - model_T3_b8f_l0_pT_f16_rd1_rw16 (Nequip with l=0)
# - model_T3_b8f_l1_pT_f16_rd1_rw16 (Nequip with l=1)
# - model_T3_b8f_l2_pT_f16_rd1_rw16 (NequIP with l=2)
# - DFT-PBEsol reference (Materials Project graphene_phonons data)
# Output: Nequip_PHONONS_COMPARISON/23_06/graphene/band_comparison_plots
# =====================================

python3 src/phonons/plot_phonons.py \
	--sources \
		nequip/GRAPHENE_MLIP/all_100frames/model_T3_b8f_l0_pT_f16_rd1_rw16/automatic_phonon_dispersion/OUTPUT/band.yaml \
		nequip/GRAPHENE_MLIP/all_100frames/model_T3_b8f_l1_pT_f16_rd1_rw16/automatic_phonon_dispersion/OUTPUT/band.yaml \
		nequip/GRAPHENE_MLIP/all_100frames/model_T3_b8f_l2_pT_f16_rd1_rw16/automatic_phonon_dispersion/OUTPUT/band.yaml \
		EXTERNAL_DATA/graphene_phonons/5383108b-180d-4eb0-a34f-b28ff9e430d7/band.yaml \
	--formats \
		phonopy \
		phonopy \
		phonopy \
		phonopy \
	--labels \
		l0 \
		l1 \
		l2 \
		DFT-PBEsol \
	--output_dir Nequip_PHONONS_COMPARISON/23_06/graphene \
	--kpath GMKG

