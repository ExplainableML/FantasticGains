#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/mt_dist-j-%j.out
#SBATCH --array=0-8

students=( 41  5   26  131 40  130 214 2   160 )
teachers=( 239 239 239 239 239 239 239 239 239 )

echo "${students[$SLURM_ARRAY_TASK_ID]}"
echo "${teachers[$SLURM_ARRAY_TASK_ID]}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/train_500_0.50_90.ffcv
val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/val_500_0.50_90.ffcv
#train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/train_500_0.50_90.ffcv
#val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/val_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $train_datapath $SCRATCH
train_datapath=$SCRATCH/train_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $val_datapath $SCRATCH
val_datapath=$SCRATCH/val_500_0.50_90.ffcv
ls $SCRATCH

python main_distillation.py \
        --config-path scripts/distillation \
        --config-name kl_dp_dist.yaml \
            ++name='xekl-distillation-imagenet' \
            ++ffcv_augmentation='default' \
            ++devices="[0,1,2,3,4,5,6]" \
            ++seed=123 \
            ++data.dataset="imagenet_subset" \
            ++data.num_classes=1000 \
            ++data.num_workers=9 \
            ++precache=True \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++data.data_path="/mnt/qb/akata/aoq877/" \
            ++loss.k=1000 \
            ++optimizer.batch_size='auto' \
            ++wandb.project='2_distill_between_experts' \
            ++student_id="${students[$SLURM_ARRAY_TASK_ID]}" \
            ++teacher_id="${teachers[$SLURM_ARRAY_TASK_ID]}" \
            ++search_id=None \
            ++checkpoint.enabled=False \
            ++freeze=False \
            ++teacher_pretrain='infograph' \
            ++mode='KL+MT_Dist'


