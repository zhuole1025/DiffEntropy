#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/data/huggingface"

# Lumina-Next supports any resolution (up to 2K)
low_res="256,512"
high_res="1024,2048"
t=1
txt_cfg=1.0
img_cfg=2.0
seed=25
steps=30
solver=euler
train_steps=0004000
model_dir=/data/DiffEntropy/flux/results/1024,2048,4096_0.5,0.5,0.0_zero_init_bias_norm/checkpoints/${train_steps}
cap_dir=validation_data.json
out_dir=samples/fix_backbone_zero_init_${train_steps}_txtcfg${txt_cfg}_imgcfg${img_cfg}_steps${steps}_seed${seed}
root_dir=/goosedata/images

CUDA_VISIBLE_DEVICES=7 python -u sample.py --ckpt ${model_dir} \
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
--zero_init \
# --ema \
# --drop_cond \
# --attn_token_select \
