#!/bin/bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

ROTS=($(seq 0 15))

compute="$PROJECT_ROOT/src/mace/phonons/phonon_stability_boundary.py"
plot="$PROJECT_ROOT/src/mace/phonons/plot_phonon_stability_boundary.py"
merge_plot="$PROJECT_ROOT/src/mace/phonons/plot_phonon_stability_boundary_aggregate.py"

for r in ${ROTS[@]}; 
do
	rm -rf "R${r}"
	python $compute --model_path $PROJECT_ROOT/from_C2_v1/MACE.model --device cuda --fmax 1e-5 --threshold 0.0015 --nstrains 91 --min_strain 0.10 --max_strain 0.30 --outdir "R${r}" --rot $r  --convergence_threshold 1e-8 --delta 1e-4 --supercell 9 9 1
	python $plot -f "R${r}/final_boundary.csv"
done

python plot_phonon_stability_boundary_aggregate.py --root_dir . --pattern final_boundary.csv -o boundaries_combined
