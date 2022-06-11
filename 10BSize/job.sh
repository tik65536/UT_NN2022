#!/bin/bash
#SBATCH -J 10job1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 32:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.6.3
conda activate NN
cd $HOME/diamond/main
echo "JOBID="$0
python3 ./main_rotation_AE.py 10-1 Max 10 3 '..AE.AE_12' 'AE' 4
