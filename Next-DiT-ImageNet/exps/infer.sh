#!/usr/bin/env sh

CUDA_VISIBLE_DEVICES=4 python infer.py \
    --ckpt /home/pgao/zl/zl/ckpt/next-dit/0900000 \
    --data_path /data0/data/imagenet/train \
    --image_save_path results/next_dit_90k \
    --batch_size 16 \
    --max_steps 10