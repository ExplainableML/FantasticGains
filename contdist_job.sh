#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --time=1-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/contdist-j-%j.out
#SBATCH --array=132,160,261,171,242,267

# students: 132,160,261,171,242,267

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/train_500_0.50_90.ffcv
val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/val_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $train_datapath $SCRATCH
train_datapath=$SCRATCH/train_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $val_datapath $SCRATCH
val_datapath=$SCRATCH/val_500_0.50_90.ffcv
ls $SCRATCH

python main_contdist.py \
        --config-path scripts/distillation \
        --config-name cont_dist.yaml \
            ++ffcv_augmentation='default' \
            ++devices="[0,1,2,3]" \
            ++data.dataset="imagenet_subset" \
            ++data.num_workers=10 \
            ++precache=True \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++optimizer.batch_size='auto' \
            ++optimizer.lr=0.01 \
            ++wandb.project='continual-distillation' \
            ++ffcv_dtype='float16' \
            ++checkpoint.enabled=True \
            ++checkpoint.dir="trained_models/contdist-short-dist-long" \
            ++contdist.student_id="$SLURM_ARRAY_TASK_ID" \
            ++contdist.n_teachers=19 \
            ++contdist.t_seed=1234 \
            ++max_epochs=5


