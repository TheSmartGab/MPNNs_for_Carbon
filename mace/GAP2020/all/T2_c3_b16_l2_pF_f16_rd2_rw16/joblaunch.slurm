#!/bin/bash
#SBATCH -o /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/mace/GAP2020/all/T2_c3_b16_l2_pF_f16_rd2_rw16/myjob.%j.%N.out
#SBATCH -e /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/mace/GAP2020/all/T2_c3_b16_l2_pF_f16_rd2_rw16/myjob.%j.%N.err
#SBATCH -D /home/users/gabriele.colombo13.stud/GNPGi/PROJECT/mace/GAP2020/all/T2_c3_b16_l2_pF_f16_rd2_rw16
#SBATCH -J MACE_train
#SBATCH --get-user-env
#SBATCH -p light
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000MB
#SBATCH --time=12:00:00
#SBATCH --account=GNPGi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriele.colombo13@studenti.unimi.it 

module load anaconda/2025-6
source /exa/software/Anaconda/2025-6/0/.bashanaconda
conda activate /exa/projects/GNPGi/programs/sources/conda_envs/mace

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun mace_run_train --config config.yaml
