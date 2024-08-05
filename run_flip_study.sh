#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=2-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/flips-j-%j.out
#SBATCH --array=425,90,380,36,341,130

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/val_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $val_datapath $SCRATCH
val_datapath=$SCRATCH/val_500_0.50_90.ffcv
ls $SCRATCH

python main_flips.py \
          --config-path config \
          --config-name prediction_flips/flips.yaml \
                      ++wandb.project="1-1_class_prediction_flips_study" \
                      ++seed="$SLURM_ARRAY_TASK_ID" \
                      ++data.val_path=$val_datapath \
                      ++mode='Sims Test'