python /home/gabri/Thesis/PROJECT/src/mace/phonons/maximum_stress.py --model ../../../MACE.model --device cuda --optimize --kcell --objective lateral -o kcell_lateral -lp 2.4598288225767555
python /home/gabri/Thesis/PROJECT/src/mace/phonons/maximum_stress.py --model ../../../MACE.model --device cuda --optimize --kcell --objective frobenius -o kcell_frobenius -lp 2.4598288225767555
python /home/gabri/Thesis/PROJECT/src/mace/phonons/maximum_stress.py --model ../../../MACE.model --device cuda --optimize --objective lateral -o primitive_lateral -lp 2.4598288225767555
python /home/gabri/Thesis/PROJECT/src/mace/phonons/maximum_stress.py --model ../../../MACE.model --device cuda --optimize --objective frobenius -o primitive_frobenius -lp 2.4598288225767555
