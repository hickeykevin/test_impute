#!/bin/bash
#SBATCH --job-name=table3
#SBATCH --time=22:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1

RUNPATH=/home/khickey/test_impute
cd $RUNPATH
source env/bin/activate
HYDRA_FULL_ERROR=1

srun python3 train.py --multirun \
 task=debug \
 trainer=cpu \
 trainer.max_epochs=50 \
 data=daicwoz \
 data.type_missing='Random' \
 data.ricardo=True \
 data.question='dream_job','easy_sleep','feeling_lately','friend_describe','last_happy','proud_life','study_school','travel_lot' \
 data.open_face='all' \
 data.ratio_missing=0.00 \
 data.num_workers=1 \
 data.batch_size=32 \
 model=gru \
 callbacks=default \
 callbacks.model_checkpoint.monitor='val/f1' \
 callbacks.clf_metrics.boot_val=False \
 logger=csv \
 'hydra.sweep.subdir="${data.question}/${data.ratio_missing}/${seed}"' \
 test=False \
 seed=0,1,2,3,4,5,6,7,8,9

exit