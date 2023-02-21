#!/bin/bash
#SBATCH --cpus-per-task=36
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/work/akata/aoq877/repsssl/LOGS/flip-s-%j.out

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#train_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/train_500_0.50_90.ffcv
val_datapath=/mnt/qb/akata/shared/ffcv_imagenet2012/val_500_0.50_90.ffcv
#rsync -avhW --no-compress --progress $train_datapath $SCRATCH
#train_datapath=$SCRATCH/train_500_0.50_90.ffcv
rsync -avhW --no-compress --progress $val_datapath $SCRATCH
val_datapath=$SCRATCH/val_500_0.50_90.ffcv
ls $SCRATCH

for s in $(seq 33 450); do
  python flip_study.py --config-path scripts/distillation --config-name flips.yaml \
                        ++name="flips-study-imagenet" \
                        ++ffcv_augmentation="default" \
                        ++devices=[0,1,2,3] ++seed=$s \
                        ++data.num_workers=36 \
                        ++data.format="ffcv" \
                        ++strategy="dp" \
                        ++precache=True \
                        ++data.train_path=$train_datapath \
                        ++data.val_path=$val_datapath \
                        ++optimizer.batch_size=64 \
                        ++wandb.project="flips-study-imagenet" \
                        ++ffcv_dtype="float16" \
                        ++topn=5
  if [[ $(($s % 10)) -eq 0 ]]
  then
    echo "clearing cash at seed $s"
    rm -r /home/akata/aoq877/.cache/torch/hub
  fi
done
