#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=2-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/mcl_dist-j-%j.out
#SBATCH --array=2

students=( 250 186 53 1  203 113 274 9  242 258 22 185 88 261 36 30  219 234 119 129 160 134 272 32 235 29  )
teachers=( 24  88  17 12 143 2   33  82 77  70  12 139 95 43  39 161 19  89  30  242 157 232 76  9  214 176 )

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
        --config-name xekl_mcl_dist.yaml \
            ++devices="[0, 1]" \
            ++seed=123 \
            ++data.dataset="imagenet_subset" \
            ++data.num_workers=9 \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++wandb.project='2_distill_between_experts' \
            ++student_id="${students[$SLURM_ARRAY_TASK_ID]}" \
            ++teacher_id="${teachers[$SLURM_ARRAY_TASK_ID]}" \
            ++search_id=None \
            ++max_epochs=20 \
            ++mode='XEKL+MCL_Dist'


