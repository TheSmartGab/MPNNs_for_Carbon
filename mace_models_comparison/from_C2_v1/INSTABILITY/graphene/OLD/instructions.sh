#!/bin/bash
#
python /home/gabri/Thesis/PROJECT/src/mace/phonons/maximum_stress.py -lp 2.4598289662305346 --model ../../MACE.model --device cuda --outdir TESTING/10_10_1_opt --optimize --supercell 10 10 1

python /home/gabri/Thesis/PROJECT/src/mace/phonons/maximum_stress.py -lp 2.4598289662305346 --model ../../MACE.model --device cuda --outdir TESTING/10_10_1 --supercell 10 10 1
