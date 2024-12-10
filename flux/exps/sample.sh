#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/ceph/data-bk/huggingface"

# Lumina-Next supports any resolution (up to 2K)
low_res="256"
high_res="1024"
t=1
txt_cfg=1.0
img_cfg=2.0
seed=25
steps=30
solver=euler
train_steps=0016000
double_gate=1.0
single_gate=1.0
model_dir=/data/zl/DiffEntropy/flux/results/1024_1.0_256,128,64_0.4,0.4,0.2_controlnet_2_4_backbone_19_38_snr_lognorm_0.5_1_cnet_snr_uniform_cfg_1.0_wo_noise_wo_shift_lr_1e-5_cap_redux_tiled/checkpoints/${train_steps}
cap_dir=validation_data.json
out_dir=samples/v3_with_single_control_redux_tiled_null_prompt_${train_steps}_gate_${double_gate}_${single_gate}
root_dir=/goosedata/images
img_embedder_path='/data/huggingface/hub/models--black-forest-labs--FLUX.1-Redux-dev/snapshots/1282f955f706b5240161278f2ef261d2a29ad649/flux1-redux-dev.safetensors'

CUDA_VISIBLE_DEVICES=4 python -u sample_controlnet.py --ckpt ${model_dir} \
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
--double_depth 2 \
--single_depth 4 \
--backbone_depth 19 \
--backbone_depth_single 38 \
--double_gate ${double_gate} \
--single_gate ${single_gate} \
--img_embedder_path ${img_embedder_path} \
# --zero_init \
# --ema \
# --drop_cond \
# --attn_token_select \
