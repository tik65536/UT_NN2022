#!/bin/bash
#SBATCH -J 10job42
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 32:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

module load python/3.6.3
conda activate NN
cd $HOME/diamond/main_local_10Bsize
echo "SLURM_JOBID="$0
python3 ./main_rotation_AE_test_noConv2Dhook.py 10-42 Max 10 3 '..AE.AE_128' 'AE' 2 5
