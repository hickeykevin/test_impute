#!/bin/bash
#SBATCH --job-name=wandb_agent_job
#SBATCH --array=0-31
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --mem=8GB

RUNPATH=/home/khickey/test_impute
cd $RUNPATH
source env/bin/activate

# Replace {agent_id} with your actual agent ID
agent_id="pybfhwgr"

srun wandb agent hickeykevin/multitask_missing/$agent_id
# # Run wandb agent in parallel with a wait period between each call 
# for i in $(seq 1 $SLURM_NTASKS); 
#     do srun --exclusive --cpu-bind=cores --distribution=cyclic wandb agent hickeykevin/multitask_missing/$agent_id & 
#     sleep 10 # Wait for 10 seconds before starting the next agent
# done

# Wait for all background processes to finish
wait