#!/bin/bash
#SBATCH -J 100job9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 32:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load python/3.6.3
conda activate NN
cd $HOME/diamond/main
echo "SLURM_JOBID="$0
python3 ./main_rotation_AE_test.py 100-9 Max 100 3 '..AE.vgg16AE_12' 'Vgg16AE' 8
