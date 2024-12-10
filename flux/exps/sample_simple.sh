#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/ceph/data-bk/huggingface"

t=1
seed=25
steps=30
solver=euler
train_steps=0016000
double_gate=1.0
single_gate=1.0
model_dir=/data/zl/DiffEntropy/flux/results/1024_1.0_256,128,64_0.4,0.4,0.2_controlnet_2_2_backbone_19_38_snr_lognorm_0.5_1_cnet_snr_uniform_cfg_1.0_wo_noise_wo_shift_lr_1e-5_cap_redux/checkpoints/${train_steps}
out_dir=samples/test_
img_embedder_path='/data/huggingface/hub/models--black-forest-labs--FLUX.1-Redux-dev/snapshots/1282f955f706b5240161278f2ef261d2a29ad649/flux1-redux-dev.safetensors'
prompt="Outdoor portrait of two women during sunset, centered around a young woman with light brown hair and a subtle smile, gazing confidently at the camera. She wears a sleeveless top with a circular emblem on the chest. Positioned slightly to the right of the frame, her hair catches the soft warm light. Behind her, slightly out of focus, is another woman with curly hair, wearing a patterned sleeveless outfit, and looking towards the left. The blurred background suggests a bridge structure with softly blurred lines, hinting at a serene waterfront setting. Dreamy lighting, warm pastel tones, shallow depth of field, soft focus on background, golden hour ambiance, intimate and relaxed atmosphere, fashion photography."
img_path=/data/zl/DiffEntropy/flux/samples/redux_t_0.5/cond_images/euler_30_0_256_1024_low.jpg
height=1024
width=1024
downsample_factor=1
denoising_strength=0.8


CUDA_VISIBLE_DEVICES=0 python -u sample_simple.py --ckpt ${model_dir} \
--prompt ${prompt} \
--img_path ${img_path} \
--height ${height} \
--width ${width} \
--downsample_factor ${downsample_factor} \
--denoising_strength ${denoising_strength} \
--image_save_path ${out_dir} \
--solver ${solver} \
--num_sampling_steps ${steps} \
--seed ${seed} \
--time_shifting_factor ${t} \
--double_gate ${double_gate} \
--single_gate ${single_gate} \
--img_embedder_path ${img_embedder_path} \
