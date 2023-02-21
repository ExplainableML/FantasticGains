#!/bin/bash

#for s in $(seq 30); do
#for s in 9 19 21 18 100 34 38 106 120; do
for t in 0.995 0.997 0.999 0.9995 0.9997; do
  for lr in 0.01 0.001 0.0001 0.00001; do #  0.00001
    sbatch dist_job.sh -lr $lr -t $t
  done
done
#done