#!/bin/bash
#SBATCH --job-name=cpu_slurm
#SBATCH --output=/home/khickey/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation/sslgan_experiment_out
#SBATCH --time=2:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

RUNPATH=/home/khickey/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation
cd $RUNPATH
source env_ssl_gan/bin/activate
HYDRA_FULL_ERROR=1
PYTHONIOENCODING=utf8 srun python3 train.py experiment=sslgan/physionet 