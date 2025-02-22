#!/bin/bash
#SBATCH --job-name=p_est-15
#SBATCH --output="work.out" 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 6-23:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bx2038@nyu.edu


module purge
module load python/3.7.10-gcc-8.5.0
module load anaconda/22 
module load miniconda3/22.11.1

source activate Env1

python Power_simulation.py