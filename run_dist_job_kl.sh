#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/kl_dist-j-%j.out
#SBATCH --array=0-32

students=( 41  5   26  131 40  130 214 2   160 41  5   26  131 40  130 214 2   160 41  5   26  131 40  130 214 2   160 41  5   26  131 40  130 214 2   160 )
teachers=( 239 239 239 239 239 239 239 239 239 234 234 234 234 234 234 234 234 234 302 302 302 302 302 302 302 302 302 77  77  77  77  77  77  77  77  77  )

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
        --config-path config \
        --config-name distillation/kl_dist.yaml \
            ++devices="[0,1]" \
            ++data.dataset="imagenet_subset" \
            ++data.num_classes=1000 \
            ++data.num_workers=9 \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++data.data_path="/mnt/qb/akata/aoq877/" \
            ++loss.k=1000 \
            ++optimizer.batch_size='auto' \
            ++wandb.project='b_ensemble-baseline' \
            ++student_id="${students[$SLURM_ARRAY_TASK_ID]}" \
            ++teacher_id="${teachers[$SLURM_ARRAY_TASK_ID]}" \
            ++checkpoint.enabled=False \
            ++search_id=None \
            ++freeze=False \
            ++teacher_pretrain='imagenet' \
            ++mode='ensemble'


