# Thesis Project — ML Interatomic Potentials for Carbon Systems

This repository contains the computational work of a thesis on **machine-learning
interatomic potentials (MLIP)** for carbon systems — primarily **graphene** and
**diamond**. The central activity is training and benchmarking neural-network
potentials (**MACE**, **NequIP**, **Allegro**) against **DFT reference data**
produced with **Quantum ESPRESSO** and **GPAW**, then using the best models to study
structural, vibrational (phonon), and mechanical properties of graphene.

> **Orientation note:** this repo mixes *tracked* content (reproducible scripts, model
> weights, DFT input setups) with large *on-disk-only* data that is intentionally kept out
> of git (see [Git hygiene](#git-hygiene-and-conventions)). A fresh clone will be missing
> the bulk data directories — that is expected and documented below.

---

## The important stuff

The main results obtained in this thesis are 
-  The calculation of the stability boundary at 0 temperature. The main script is src/mace/phonons/phonon_stability_boundary.py; the rest of the sources in that directory have been used for plotting or exploring phonon calculations using mlips. It should be noted that I used the finite difference method implemented in ASE for phonon calculations, and that this should be improved by computing the potential energy Hessian via backpropagation. Currently, the script expects a MACE model, but it may be easily adapted to accept a general one; the actual stability boundary search is model-independent, as the model is only used to compute force coefficients.
-  The (Bayesian) inference of the graphene breaking rate at finite temperature. To this end, the LAMMPS and bash scripts used to automate the LAMMPS run are stored in src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress. The tools for the structural analysis and Bayesian inference are stored in src/structure, and they are model-independent. The analysis has been run in mace_models_comparison/from_C2_v1/LAMMPS_STUDIES/strain_stress, in particular, in 5_5/zigzag for different temperatures and strains. The LAMMPS scripts used take parameters from command-line input. To configure run configurations, .conf files are stored in the corresponding directories; they are actually bash scripts declaring and initialising variables that are then passed to the LAMMPS scripts by the automation.sh scripts are in the src directory. The actual .conf files I used for the simulations are stored in the corresponding directories.

Many of the Python scripts and notebooks reported here are helpers, plotters, and tests. They are not intended to be general in their purpose, but they may help understand the obtained results and the
problems encountered along the way in a bottom-up approach; thus, they are retained in this repository for illustrative purposes.

All the scripts have been written using ArgumentParser from argparse

---

## Repository layout

| Directory | What it is |
|---|---|
| `mace_models_comparison/` | **Primary MACE models** (thesis ch. 3) + training provenance (`MODELS_INFO.txt`) |
| `src/` | **Main analysis code**, installed as the Python package `thesisproject` |
| `DFT_checks/` | DFT reference computations — Quantum ESPRESSO input setups & structures |
| `DATASET_SIZE_STUDY/` | MACE scaling study: model quality vs. training-set size (ch. 4) |
| `nequip/` | NequIP training runs + pair-scan / 2-body potential comparisons |
| `mace/` | MACE training dirs + pretrained/foundation models |
| `EXTERNAL_DATA/` | External phonon data (Materials Project) + a third-party MACE model |
| `mace_basis_function_exploration/` | Exploration of MACE's radial basis-function construction |
| `PHONONS_COMPARISON/`, `Nequip_PHONONS_COMPARISON/` | Generated phonon-comparison outputs (from the root `.sh` scripts) |
| `primes/` | PHP prime generator used to keep MD runs independent |
| `WORKFLOWS/` | Graphviz diagram of the overall thesis workflow |
| `GRAPHENE_PHONON_COMPARISON.sh`, `DIAMOND_PHONON_COMPARISON.sh`, `GRAPHENE_PHONON_COMPARISON_NEQUIP.sh` | Root entry-point scripts for phonon band-structure comparison |
| `pyproject.toml`, `uv.lock` | Python packaging (`thesisproject`) and locked dependencies |

---

## Main directories of interest

### 1. Where the MACE models live

**Hero model — `mace_models_comparison/from_C2_v1/`.** This is the GAP2020-trained-on-C2
MACE model that serves as the main comparison baseline (thesis ch. 3) and is carried into
ch. 4 to investigate the structural properties of graphene. Its siblings in
`mace_models_comparison/` are the other final models compared against it:

- `from_C2_cut/` — a "cut" variant of the same model.
- `mace-GAP2020_FROM_SCRATCH_smaller/`, `…_smaller2_l2/` — smaller from-scratch models (v2 / v2-l2).
- `mace-GAP2020-FROM_SCRATCH_v3/`, `…_v4/` — from-scratch versions 3 and 4.
- `v5training/` — the v5 training run.

Full training commands (`python -m mace.cli.run_train …`) and provenance are recorded in
`mace_models_comparison/MODELS_INFO.txt`. Note that **fine-tuning was abandoned** ("E0s are
wrong"), so the models above were trained from scratch or from the C2 base.

**Dataset-size study — `DATASET_SIZE_STUDY/NEW_SPLIT/mace/from_C2_v1/*/`.** One directory per
training-set size, each holding the final model(s) for that size; this is where the
model-quality-vs-data-quantity results (ch. 4) come from.

**Other final models:** `mace/GAP2020/all/C2_only/{smaller,v0}/` and
`mace/GRAPHENE_MLIP_GPAW_COMPLEMENTS/…` hold additional exported finals.

**Pretrained / foundation models (not the research models):** `mace/mace-off`,
`mace/mace-omat-0-medium`, `mace/mace-mh0`, `mace/mace-mh1`, `mace/mace-mp-0`, and
`mace/mace-mpa-0-medium` are vendor/foundation MACE checkpoints used as starting points or
references. Their bulk weights are gitignored; only the small tracked metadata remains.

> **Model-weight policy:** only **final exported weights** are committed. Per-epoch
> checkpoints (`*_epoch-N.pt`) and pretrained/foundation weights are deliberately excluded.
> See [Git hygiene](#git-hygiene-and-conventions).

### 2. Where the main analysis code lives — `src/`

The analysis code is a normal Python package, `thesisproject`, using a **src layout**
(`pyproject.toml` → `[tool.setuptools] package-dir = {"" = "src"}`). Subpackages:

| Subpackage | Purpose |
|---|---|
| `src/phonons/` | Phonon band-structure comparison; `plot_phonons.py` is the engine behind the root `.sh` scripts |
| `src/mace/` | MACE metrics, MD setup, and LAMMPS scripting |
| `src/structure/` | Structural analysis — rate inference, restart inspection |
| `src/datasets/` | Dataset preparation + external data requests (`requests/`) |
| `src/dimers/` | 2-body / Tersoff potential work in graphene |
| `src/DFT/` | DFT-related utilities |
| `src/model_inspection/` | Model introspection (SVDV, hidden descriptors) |
| `src/nequip/` | NequIP-specific analysis |
| `src/compare_outputs/` | Cross-model output comparison |
| `src/tools/` | Shared helpers |

**Console entry points** (defined in `pyproject.toml`, available on the venv `PATH`):
`diff_distributions`, `dump2traj`, `compare_rates`, and `mace_compare_sizes`.

The root `.sh` scripts are thin wrappers that call `src/phonons/plot_phonons.py` with a set of
model band-structure files plus a DFT reference, writing comparison plots into
`PHONONS_COMPARISON/17_06/{graphene,diamond}` and (for NequIP)
`Nequip_PHONONS_COMPARISON/23_06/graphene`.

---

## Data sources & provenance

| Source | Where used | Reference |
|---|---|---|
| Kaggle **"periodic-gap2020"** dataset (`TRAIN.extxyz` / `VAL.extxyz`) | Main training data for the MACE models | Kaggle input path referenced in `mace_models_comparison/MODELS_INFO.txt` and `src/` notebooks |
| **Materials Project** phonons — mp-66 (diamond), mp-1040425 (graphene) | DFT-PBEsol phonon references | `EXTERNAL_DATA/diamond_phonons/`, `EXTERNAL_DATA/graphene_phonons/` |
| **IVOR** MACE model | Third-party comparison model | `doi:10.1021/acs.inorgchem.5c01115` — see `EXTERNAL_DATA/IVOR/` |
| **NOMAD** upload & **AFLOW** API | Data requests for additional configurations | request logs under `src/datasets/requests/` |

---

## Git hygiene and conventions

These rules are load-bearing for this repo and should be preserved:

- **Never commit bulk data.** Before any commit, confirm no bulk files are staged:
  `git status --short`, and
  `git ls-files | grep -cE '\.(npy|npz|json|csv|done|log|extxyz|traj|lammpstrj|dump)'` must stay
  `0`. `.git` should be hundreds of MB, not GBs.
- **Final-weights-only model policy.** Only final exported model weights are tracked. To commit
  a new final model, add a `!<path>` negation line to the KEPT section of `.gitignore`; do **not**
  remove a bulk-extension ignore to "fix" a missing model. Per-epoch checkpoints and
  pretrained/foundation weights stay out.
- **`.gitignore` is designed around negations.** The 97 kept final weights are tracked via `!`
  lines that override the blanket `*.model` / `*.pt` / `*.pth` / `*.ckpt` ignores.
- **Path/portability:** never hardcode the absolute repo path in scripts. Resolve the root at
  runtime with `git rev-parse --show-toplevel`.

---

## Environment

- Python package `thesisproject` (setuptools, src layout), Python ≥ 3.9. Dependencies are locked
  in `uv.lock`; a virtual environment is expected at the repo root (`.venv/`).
- Core deps: numpy, scipy, ase, matplotlib, pandas, lammps_logfile, emcee, numba, tqdm,
  seaborn, argcomplete. Optional `gpu` extra adds `torch` for MACE/NequIP on GPU.
- Some analysis scripts expect the `.venv` at the repo root and expect the gitignored data
  directories (`DATASETS/`, `gpaw_testing/`) to be present locally.
