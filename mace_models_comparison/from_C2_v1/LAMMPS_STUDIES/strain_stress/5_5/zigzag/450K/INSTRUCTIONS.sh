PROJECT_ROOT="$(git rev-parse --show-toplevel)"

source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/generate_independent_configurations_mliap.sh -n 100 -i STRAINSTRESS/190000.lammps-data -o RESTARTS_0.19 -r prolong_0.19.conf

source $PROJECT_ROOT/src/mace/MD/LAMMPS_SCRIPTS/graphene/strain_stress/generate_independent_configurations_mliap.sh -n 100 -i STRAINSTRESS/200000.lammps-data -o RESTARTS_0.20 -r prolong_0.20.conf
