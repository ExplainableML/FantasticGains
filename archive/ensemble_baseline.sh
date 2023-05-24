#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --time=1-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/ensemble-j-%j.out
#SBATCH --array=124,214,291,101,232,79,145,151,26,277,77,109,182,299,36,130,2,292,211,234

# teachers: 77,66,136,157,94,43
# diverse teachers: 124,214,291,101,232,79,145,151,26,277,77,109,182,299,36,130,2,292,211,234
students=( 250 186 53 1 203 113 274 9 242 258 22 185 88 261 36 30 219 234 119 129 160 134 272 32 235 29 )
teachers=( 24 88 17 12 143 2 33 82 77 70 12 139 95 43 39 161 19 89 30 242 157 232 76 9 214 176 )
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
        --config-name xekl_dist.yaml \
            ++name='ensemble-imagenet' \
            ++ffcv_augmentation='default' \
            ++devices="[0,1,2,3]" \
            ++seed=123 \
            ++data.dataset="imagenet_subset" \
            ++data.num_workers=10 \
            ++precache=True \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++optimizer.batch_size='auto' \
            ++wandb.project='b_ensemble-baseline' \
            ++ffcv_dtype='float16' \
            ++scheduler.warmup=0 \
            ++scheduler.name=None \
            ++checkpoint.enabled=False \
            ++loss.name='ensemble_baseline'\
            ++loss.alpha=0 \
            ++loss.kd_T=0 \
            ++loss.k=0 \
            ++loss.gamma=0 \
            ++loss.strat='simple' \
            ++optimizer.lr=0 \
            ++on_flip.pos='NA' \
            ++on_flip.neg='NA' \
            ++on_flip.neut='NA' \
            ++student_id=9 \
            ++teacher_id="$SLURM_ARRAY_TASK_ID" \
            ++search_id=None \
            ++max_epochs=0 \
            ++tag='st-test'


