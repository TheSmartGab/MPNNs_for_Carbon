# Graphene MLIP Models (`mace/GRAPHENE_MLIP`)

This directory contains MACE machine learning interatomic potential (MLIP) models and configurations specifically trained for graphene and 2D carbon materials. These models are optimized to capture the unique bonding, flexural modes, and elastic properties of graphene.

## Contents:

MACE model checkpoints and training outputs for:
- Graphene-specific ML potentials
- 2D carbon lattice vibration predictions
- In-plane and out-of-plane (flexural) phonon mode accuracy

## Usage:

These graphene MLIP models are used for:
1. Simulating graphene molecular dynamics with near-DFT accuracy at reduced computational cost
2. Phonon dispersion calculations using `src/mace/phonons/compute_dispersion.py`
3. Comparing against DFT/GPAW reference phonon data in `EXTERNAL_DATA/graphene_phonons/`

Complementary GPAW calculation references and complements are stored in `mace/GRAPHENE_MLIP_GPAW_COMPLEMENTS/`. Related NequIP graphene models are in `nequip/GRAPHENE_MLIP/`.