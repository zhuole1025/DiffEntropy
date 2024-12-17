#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/zl/huggingface"

# Lumina-Next supports any resolution (up to 2K)
res="1024:1024x1024"
t=1
txt_cfg=1.0
seed=25
steps=50
solver=euler
train_steps=0002000
# model_dir=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/zl/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/flux1-dev.safetensors
model_dir=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/zl/DiffEntropy/flux/results/1024_lr_1e-4_bsz_32_1_10_cfg1/checkpoints/${train_steps}
cap_dir=validation_data.json
out_dir=samples/cfg1_gt_shift_${train_steps}_txtcfg${txt_cfg}_imgcfg${img_cfg}_steps${steps}_seed${seed}
# out_dir=samples/wo_huber_txtcfg4_shift_steps${steps}_seed${seed}
root_dir=/goosedata/images

CUDA_VISIBLE_DEVICES=7 python -u sample.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--txt_cfg_scale ${txt_cfg} \
--batch_size 1 \
# --zero_init \
# --ema \
# --drop_cond \
# --attn_token_select \
