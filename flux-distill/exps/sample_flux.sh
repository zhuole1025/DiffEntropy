#!/usr/bin/env sh

export HF_TOKEN="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"
export HF_HOME="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/zl/huggingface"

# Lumina-Next supports any resolution (up to 2K)
res="1024:1024x1024"
t=1
txt_cfg='4.0'
seed=25
steps=30
solver=euler
model_dir=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/zl/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/flux1-dev.safetensors
cap_dir=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/qinqi/lumina_next_t2i_256/t2i_1024/captions.txt
out_dir=samples/baseline_cap1_shift_steps${steps}_seed${seed}
root_dir=/goosedata/images

CUDA_VISIBLE_DEVICES=1 python -u sample_flux.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--guidance_list ${txt_cfg} \
--batch_size 1 \

