#!/bin/bash
#SBATCH -J job11
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 32:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.6.3
conda activate NN
cd $HOME/diamond/main
echo "SLURM_JOBID="$0
python3 ./main_rotation_VAE_test.py 11 Max 50 3 '..VAE.vgg16VAE_12' 'Vgg16VAE' 4
