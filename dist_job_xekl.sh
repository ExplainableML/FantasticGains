#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=2-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/xekl_dist-j-%j.out
#SBATCH --array=0-31

# teachers: 77,66,136,157,94,43
# diverse teachers: 124,214,291,101,232,79,145,151,26,277,77,109,182,299,36,130,2,292,211,234
# KL Missing
#students=( 41 )
#teachers=( 234 )
# XEKL Missing
students=( 41 )
teachers=( 151 )
echo "${students[$SLURM_ARRAY_TASK_ID]}"
echo "${teachers[$SLURM_ARRAY_TASK_ID]}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

#train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/train_500_0.50_90.ffcv
#val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/val_500_0.50_90.ffcv
train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/train_500_0.50_90.ffcv
val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/val_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $train_datapath $SCRATCH
train_datapath=$SCRATCH/train_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $val_datapath $SCRATCH
val_datapath=$SCRATCH/val_500_0.50_90.ffcv
ls $SCRATCH

python main_distillation.py \
        --config-path scripts/distillation \
        --config-name xekl_dist.yaml \
            ++name='xekl-distillation-imagenet' \
            ++ffcv_augmentation='default' \
            ++devices="[0]" \
            ++strategy='dp' \
            ++seed=123 \
            ++data.dataset="imagenet" \
            ++data.num_workers=4 \
            ++precache=True \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++optimizer.batch_size='auto' \
            ++wandb.project='2_distill_between_experts' \
            ++ffcv_dtype='float16' \
            ++scheduler.warmup=0 \
            ++scheduler.name=None \
            ++checkpoint.enabled=True \
            ++loss.name='xekl'\
            ++loss.alpha=1 \
            ++loss.kd_T=1 \
            ++loss.k=1000 \
            ++loss.gamma=0 \
            ++loss.strat='NA' \
            ++optimizer.lr=0.0001 \
            ++on_flip.pos='distill' \
            ++on_flip.neg='distill' \
            ++on_flip.neut='distill' \
            ++student_id=0 \
            ++teacher_id=0 \
            ++search_id="$SLURM_ARRAY_TASK_ID" \
            ++max_epochs=20 \
            ++tag='KL_Dist'


