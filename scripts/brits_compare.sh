#!/bin/bash
#SBATCH --job-name=brits_compare
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=brits_compare_%j.out  # Save the output to a file with the job ID

RUNPATH=/home/khickey/test_impute
cd $RUNPATH
source env/bin/activate


id="5iommtt7"
srun wandb agent hickeykevin/multitask_missing/$id

exit