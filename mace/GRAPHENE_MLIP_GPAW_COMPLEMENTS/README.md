# Graphene MLIP GPAW Complements (`mace/GRAPHENE_MLIP_GPAW_COMPLEMENTS`)

This directory contains complementary DFT/GPAW reference calculations for the graphene MLIP models. These GPAW outputs provide high-accuracy reference energies, forces, and stresses used to train or validate the MACE graphene potentials.

## Contents:

GPAW reference calculation directories and outputs for:
- Graphene single-point energy validations
- Force comparison configurations against MLIP predictions
- Stress tensor reference calculations for 2D lattice validation

## Usage:

These GPAW complement files are used to:
1. Generate training data corrections or "complement" sets for MACE model fine-tuning
2. Validate MLIP-predicted forces and stresses against DFT references
3. Compute error metrics using `src/compare_outputs/` utilities (RMSE, MAE distributions)

Related graphene MLIP models are stored in `mace/GRAPHENE_MLIP/`, and NequIP counterparts are in `nequip/GRAPHENE_MLIP/`.