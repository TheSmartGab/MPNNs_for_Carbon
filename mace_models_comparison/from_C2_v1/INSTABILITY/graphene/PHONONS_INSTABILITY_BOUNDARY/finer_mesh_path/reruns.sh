PROJECT_ROOT="$(git rev-parse --show-toplevel)"

rs=($(seq 0 15))

for r in ${rs[@]};
do
	
	python $PROJECT_ROOT/src/mace/phonons/rerun_phonon_stability_boundary.py --model_path $PROJECT_ROOT/KAGGLE_DOWNLOAD/from_C2_v1/MACE.model --device cuda --fmax 1e-5 --threshold 0.0015 --min_strain 0 --max_strain 0.3 --outdir ./R$r --rot $r --final_boundary_file "R$r/"final_boundary.csv

done


