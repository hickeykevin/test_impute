#!/bin/bash
#SBATCH --job-name=table3
#SBATCH --time=22:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

RUNPATH=/home/khickey/test_impute
cd $RUNPATH
source env/bin/activate
HYDRA_FULL_ERROR=1

srun python3 train.py --multirun \
 task=debug \
 trainer=gpu \
 trainer.max_epochs=50 \
 data=daicwoz \
 data.type_missing='Random' \
 data.ricardo=True \
 data.question='advice_yourself','anything_regret','argued_someone','controlling_temper','diagnosed_depression','diagnosed_p_t_s_d','doing_today' \
 data.open_face='all' \
 data.ratio_missing=0.00,0.05,0.10,0.15,0.20,0.25,0.30 \
 data.num_workers=1 \
 data.batch_size=32 \
 model=csdi \
 callbacks=[default,imputation_metrics] \
 callbacks.model_checkpoint.monitor='val/f1' \
 callbacks.clf_metrics.boot_val=False \
 callbacks.imputation_metrics.boot_val=False \
 logger=csv \
 'hydra.sweep.subdir="${data.question}/${data.ratio_missing}/${seed}"' \
 test=True \
 seed=0,1,2,3,4,5,6,7,8,9

exit