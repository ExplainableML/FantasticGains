#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --time=1-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/dist-j-%j.out
#SBATCH --array=0-300

# teachers: 77,66,136,157,94,43
# diverse teachers: 124,214,291,101,232,79,145,151,26,277,77,109,182,299,36,130,2,292,211,234
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/train_500_0.50_90.ffcv
val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/val_500_0.50_90.ffcv
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
            ++devices="[0,1,2,3]" \
            ++seed=123 \
            ++search_id="$SLURM_ARRAY_TASK_ID" \
            ++data.dataset="imagenet_subset" \
            ++data.num_workers=10 \
            ++precache=True \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++optimizer.batch_size='auto' \
            ++optimizer.lr=0.001 \
            ++wandb.project='xekl-randomsearch' \
            ++ffcv_dtype='float16' \
            ++scheduler.warmup=0 \
            ++scheduler.name=None \
            ++checkpoint.enabled=True \
            ++loss.name='xekl'\
            ++loss.alpha=0.68 \
            ++loss.kd_T=1 \
            ++loss.k=100 \
            ++loss.tau=0.9999 \
            ++loss.N=2 \
            ++on_flip.pos='distill' \
            ++on_flip.neg='nothing' \
            ++on_flip.neut='distill' \
            ++student_id=1 \
            ++teacher_id=1 \
            ++max_epochs=20 \
            ++tag='random-search'


