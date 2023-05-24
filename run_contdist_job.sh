#!/bin/bash
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/contdist-j-%j.out
#SBATCH --array=41

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/akata/aoq877/torch_models/

#train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/train_500_0.50_90.ffcv
#val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012_subset100k/val_500_0.50_90.ffcv
train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/train_500_0.50_90.ffcv
val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/val_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $train_datapath $SCRATCH
train_datapath=$SCRATCH/train_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $val_datapath $SCRATCH
val_datapath=$SCRATCH/val_500_0.50_90.ffcv
ls $SCRATCH

python main_contdist.py \
        --config-path scripts/distillation \
        --config-name kl_dp_dist.yaml \
            ++devices="[0,1,2,3,4,5,6,7]" \
            ++data.dataset="imagenet" \
            ++data.num_workers=9 \
            ++data.train_path="$train_datapath" \
            ++data.val_path="$val_datapath" \
            ++wandb.project='3_continual_distillation' \
            ++checkpoint.dir="/mnt/qb/akata/aoq877/trained_models/contdist" \
            ++contdist.student_id="$SLURM_ARRAY_TASK_ID" \
            ++contdist.approach='simple' \
            ++contdist.n_teachers=3 \
            ++contdist.t_seed=1234 \
            ++contdist.curriculum='asc' \
            ++contdist.sequential=True \
            ++max_epochs=20 \
            ++search_id=None \
            ++mode='Cont_MT'


