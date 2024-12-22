#!/bin/bash

ITER=0900000
cfg_scale=4
num_steps=100
t=1
mkdir -p image_samples
SAMPLE_DIR=image_samples/${ITER}_${cfg_scale}_${num_steps}_${t}
rm -r ${SAMPLE_DIR}

python sample.py ODE \
  --ema --cfg_scale ${cfg_scale} --precision tf32 --num_gpus 1 \
  --num-fid-samples 100 --sample-dir ${SAMPLE_DIR} --ema \
  --per-proc-batch-size 32 --num_sampling_steps ${num_steps} --sampling-method euler \
  --time_shifting_factor ${t} --prediction data \
  --ckpt /home/pgao/zl/zl/ckpt/next-dit/${ITER} \
  --ode_imp --save_traj