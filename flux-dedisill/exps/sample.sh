#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/data/huggingface"

# Lumina-Next supports any resolution (up to 2K)
res="1024:1024x1024"
t=1
txt_cfg=8.0
seed=25
steps=30
solver=euler
train_steps=0002000
model_dir=/data/DiffEntropy/flux/results/1024_1.0_256,128,64_0.4,0.4,0.2_controlnet_2_2_backbone_19_38_snr_lognorm_0.5_1_cnet_snr_uniform_cfg_1.0_wo_noise_wo_shift_lr_1e-5/checkpoints/${train_steps}
cap_dir=validation_data.json
out_dir=samples/huber_lognorm_gt_N_1_10_${train_steps}_txtcfg${txt_cfg}_steps${steps}_seed${seed}
root_dir=/goosedata/images

CUDA_VISIBLE_DEVICES=3 python -u sample.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--txt_cfg_scale ${txt_cfg} \
--batch_size 1 \

