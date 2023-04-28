#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=2-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/cd_dist-j-%j.out
#SBATCH --array=0-46

# teachers: 77,66,136,157,94,43
# diverse teachers: 124,214,291,101,232,79,145,151,26,277,77,109,182,299,36,130,2,292,211,234
# students=( 250 186 53 1 203 113 274 9 242 258 22 185 88 261 36 30 219 234 119 129 160 134 272 32 235 29 )
# teachers=( 24 88 17 12 143 2 33 82 77 70 12 139 95 43 39 161 19 89 30 242 157 232 76 9 214 176 )
# missing CD
#students=( 171 171 171 171 171 171 171 171 171 171 171 132 132 132 132 132 132 132 9   9   9   9   160 160 160 160 160 160)
#teachers=( 211 234 36  182 77  151 145 232 101 291 124 211 234 36  182 232 101 291 234 151 101 124 211 234 151 145 232 291)
# missing CRD
students=( 171 )
teachers=( 101 )
#students=( 88 242 )
#teachers=( 95 77 )
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
        --config-name crd_dist.yaml \
            ++name='cd-distillation-imagenet' \
            ++ffcv_augmentation='default' \
            ++devices="[0]" \
            ++seed=123 \
            ++data.dataset="imagenet_subset" \
            ++data.num_workers=9 \
            ++precache=True \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++optimizer.batch_size='auto' \
            ++wandb.project='2_distill_between_experts' \
            ++ffcv_dtype='float16' \
            ++scheduler.warmup=0 \
            ++scheduler.name=None \
            ++checkpoint.enabled=True \
            ++student_id="${students[$SLURM_ARRAY_TASK_ID]}" \
            ++teacher_id="${teachers[$SLURM_ARRAY_TASK_ID]}" \
            ++search_id=None \
            ++max_epochs=20 \
            ++tag='CRD_Dist'


