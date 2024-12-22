#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/ceph/data-bk/huggingface"

# Lumina-Next supports any resolution (up to 2K)
low_res="256"
high_res="1024"
t=1
txt_cfg=1.0
img_cfg=1.0
controlnet_cfg=1.0
backbone_cfg=2.0
seed=25
steps=30
solver=euler
train_steps=0016000
controlnet_snr=none
double_gate=1.0
single_gate=1.0
model_dir=/ceph/data-bk/zl/DiffEntropy/flux/results/1024_1.0_256_1.0_depth_2_4_19_38_snr_lognorm_none_cfg_1.0_1.0_bsz_8_wo_shift_lr_1e-5_cap_redux_only_tiled_multi_degradation_wo_noise_wo_usm_fix/checkpoints/${train_steps}
cap_dir=validation_data.json
out_dir=samples/v3_with_single_control_redux_tiled_multi_degradation_train_wo_noise_wo_usm_fix_${train_steps}_gate_${double_gate}_${single_gate}_cfg_${backbone_cfg}_${controlnet_cfg}_image_prompt_lognorm
root_dir=/ceph/data-bk/images
img_embedder_path='/ceph/data-bk/zl/ckpts/flux1-redux-dev.safetensors'

CUDA_VISIBLE_DEVICES=5 python -u sample_controlnet.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--data_path ${cap_dir} \
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
--controlnet_cfg ${controlnet_cfg} \
--backbone_cfg ${backbone_cfg} \
--cond_type image \
--proportion_empty_prompts 1.0
# --controlnet_snr ${controlnet_snr} \
# --zero_init \
# --ema \
# --drop_cond \
# --attn_token_select \