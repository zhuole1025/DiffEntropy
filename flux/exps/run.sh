#!/bin/bash

export WANDB_API_KEY="75de1215548653cdc8084ae0d1450f2d84fd9a20"
export HF_TOKEN="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"
export HF_HOME="/ceph/data-bk/huggingface"

train_data_root='configs/data/2M.yaml'
batch_size=256
micro_batch_size=8
lr=1e-5
precision=bf16
low_res_list=256
low_res_probs=1.0
high_res_list=1024
high_res_probs=1.0
snr_type=uniform
controlnet_snr=none
backbone_cfg=1.0
controlnet_cfg=1.0
double_depth=2
single_depth=4
backbone_depth=19
backbone_depth_single=38
img_embedder_path='/data/huggingface/hub/models--black-forest-labs--FLUX.1-Redux-dev/snapshots/1282f955f706b5240161278f2ef261d2a29ad649/flux1-redux-dev.safetensors'

exp_name=${high_res_list}_${high_res_probs}_${low_res_list}_${low_res_probs}_depth_${double_depth}_${single_depth}_${backbone_depth}_${backbone_depth_single}_snr_${snr_type}_${controlnet_snr}_cfg_${backbone_cfg}_${controlnet_cfg}_bsz_${batch_size}_wo_shift_lr_${lr}_cap_redux_only_tiled_multi_degradation_wo_noise_wo_usm
# exp_name=learnable_gate_lr_${lr}
# init_from=/data/zl/DiffEntropy/flux/results/1024_1.0_256_1.0_depth_2_4_19_38_snr_uniform_none_cfg_1.0_1.0_wo_shift_lr_1e-5_cap_redux_tiled_multi_degradation_wo_noise_wo_usm/checkpoints/0040000
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 --nnodes=1 --master_port 29311 train_controlnet.py \
    --master_port 18181 \
    --global_bs ${batch_size} \
    --micro_bs ${micro_batch_size} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 500 --log_every 10 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 12 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --high_res_list ${high_res_list} \
    --high_res_probs ${high_res_probs} \
    --low_res_list ${low_res_list} \
    --low_res_probs ${low_res_probs} \
    --use_wandb \
    --double_depth ${double_depth} \
    --single_depth ${single_depth} \
    --backbone_depth ${backbone_depth} \
    --backbone_depth_single ${backbone_depth_single} \
    --img_embedder_path ${img_embedder_path} \
    --backbone_cfg ${backbone_cfg} \
    --controlnet_cfg ${controlnet_cfg} \
    --load_clip \
    --caption_dropout_prob 1.0 \
    --cond_type image \
    --checkpointing \
    # --learnable_gate \
    # --init_from ${init_from} \
    # --controlnet_snr ${controlnet_snr} \
    # --compute_controlnet_loss \
    
