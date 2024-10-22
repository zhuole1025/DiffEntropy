#!/usr/bin/env sh

export HF_TOKEN="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"
# export HF_HOME="/data/huggingface"

# Lumina-Next supports any resolution (up to 2K)
low_res="256"
high_res="1024,2048"
t=1
txt_cfg=0.0
img_cfg=2.0
seed=25
steps=30
solver=euler
train_steps=0005000
model_dir=/home/ubuntu/zl/DiffEntropy/flux/results/1024,2048,4096_0.5,0.5,0.0_wo_drop/checkpoints/${train_steps}
cap_dir=/home/ubuntu/zl/DiffEntropy/validation_data.json
out_dir=samples/wo_cond_${train_steps}_txtcfg${txt_cfg}_imgcfg${img_cfg}_steps${steps}_seed${seed}
root_dir=/home/ubuntu/goosedata/images

CUDA_VISIBLE_DEVICES=0 python -u sample.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--root_path ${root_dir} \
--seed ${seed} \
--high_res_list ${high_res} \
--low_res_list ${low_res} \
--time_shifting_factor ${t} \
--txt_cfg_scale ${txt_cfg} \
--img_cfg_scale ${img_cfg} \
--batch_size 1 \
# --drop_cond \
# --attn_token_select \
