set terminal pdfcairo size 12cm,8cm enhanced font "Helvetica,16"
set output "Hypers2BodyPotential.pdf"

ROOT = system("git rev-parse --show-toplevel")
# NOTE: gnuplot's system() already returns the command output with the trailing
# newline stripped, so ROOT is a clean path ready for (ROOT . "/..."). Do NOT
# re-add a substr()-based strip here: 2-arg substr() and the regex forms both
# error out on gnuplot 6.0.x ("stack underflow" / "non-integer passed to boolean").

set xlabel "Distance [Å]"
set ylabel "Potential energy [eV]"
set grid

# This forces the legend to fill vertically, maxing out at 5 rows per column
set key bottom right inside columns 2 vertical maxrows 5

# set title "2-atom potential energy curves"

# ---- files ----
FILES = (ROOT . "/mace/GAP2020/all/C2_only/v0/2_atoms_potential/model_penergy.out") . " \
./from_C2_v1/2_atoms_potential/model_penergy.out \
./from_C2_cut/2_atoms_potential/model_penergy.out \
./mace-GAP2020_FROM_SCRATCH_smaller/2_atoms_potential/model_penergy.out \
./mace-GAP2020_FROM_SCRATCH_smaller2_l2/2_atoms_potential/model_penergy.out \
./mace-GAP2020-FROM_SCRATCH_v3/checkpoints/2_body_potential/model_penergy.out \
./mace-GAP2020-FROM_SCRATCH_v4/checkpoints/2_body_potential/model_penergy.out \
./mace-GAP2020-FROM_SCRATCH_v5/checkpoints/2_body_potential/model_penergy.out"

# ---- labels (same order!) ----
LABELS = "\
v1-C2 \
v1 \
v1-cut \
v2 \
v2-l2 \
v3 \
v4 \
v5"

# ---- plotting back in original sequence ----
plot [0.8:6] \
    (ROOT . "/DATASETS/C/GAP2020/config_types/Dimer/split/training/penergy.dat") u 1:2 w p lw 2 ps 1 title "training", \
    (ROOT . "/DATASETS/C/GAP2020/config_types/Dimer/split/validation/penergy.dat") u 1:2 w p lw 2 ps 1 title "validation", \
    for [i=1:words(FILES)] word(FILES,i) u 1:2 w l lw 2 title word(LABELS,i)

# Explicitly close the output file
set output
