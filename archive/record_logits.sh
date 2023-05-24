#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=cpu-preemptable
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=2-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/pred_dist-j-%j.out
#SBATCH --array=0-323

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/


python analyze_preds_dist.py \
          --config-path scripts/distillation \
          --config-name flips.yaml \
                      ++data.num_workers=9 \
                      ++wandb.project="1-2_class_prediction_dist_study" \
                      ++model_id="$SLURM_ARRAY_TASK_ID" \
                      ++tag='Preds Dist Test'

