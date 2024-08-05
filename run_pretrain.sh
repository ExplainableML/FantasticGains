#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/pretrain-j-%j.out
#SBATCH --array=5

models=( 41 5 26 131 40 130 214 2 160 234 302 77 12 310 214 2 160 12 77 239 )

echo "${models[$SLURM_ARRAY_TASK_ID]}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_HOME=/mnt/qb/work/akata/aoq877/torch_models/

python main_pretrain.py \
        --config-path scripts \
        --config-name pretrain/cub_pretrain.yaml \
            ++model_id="${models[$SLURM_ARRAY_TASK_ID]}"



