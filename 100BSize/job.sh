#!/bin/bash
#SBATCH -J 100job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 32:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load python/3.6.3
conda activate NN
cd $HOME/diamond/main
echo "JOBID="$0
python3 ./main_rotation_AE.py 100-1 Max 100 3 '..AE.AE_12' 'AE' 8
