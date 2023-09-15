#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/cars-pretrain-j-%j.out
#SBATCH --array=14

models=( 41 5 26 131 40 130 214 2 160 234 302 77 12 310 239 )


echo "${models[$SLURM_ARRAY_TASK_ID]}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

#train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/train_500_0.50_90.ffcv
#val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/val_500_0.50_90.ffcv
#train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/train_500_0.50_90.ffcv
#val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/val_500_0.50_90.ffcv
#rsync -avhW --no-compress --progress $train_datapath $SCRATCH
#train_datapath=$SCRATCH/train_500_0.50_90.ffcv
#rsync -avhW --no-compress --progress $val_datapath $SCRATCH
#val_datapath=$SCRATCH/val_500_0.50_90.ffcv
#ls $SCRATCH

python main_pretrain.py \
        --config-path scripts/distillation \
        --config-name pretrain.yaml \
            ++name='pretrain_cars' \
            ++ffcv_augmentation='default' \
            ++seed=1234 \
            ++data.dataset="cars" \
            ++data.num_classes=196 \
            ++data.num_workers=10 \
            ++optimizer.batch_size='auto' \
            ++optimizer.lr=0.01 \
            ++wandb.project='Cars_Pretrain' \
            ++model_id="${models[$SLURM_ARRAY_TASK_ID]}" \
            ++search_id=None \
            ++max_epochs=50 \
            ++checkpoint.enabled=False \
            ++freeze=True \
            ++mode='Cars_Finetune_Test'


