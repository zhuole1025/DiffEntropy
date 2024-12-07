#!/bin/bash

export WANDB_API_KEY="75de1215548653cdc8084ae0d1450f2d84fd9a20"
export HF_TOKEN="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"
export HF_HOME="/data/huggingface"

train_data_root='configs/data/2M.yaml'
batch_size=8
micro_batch_size=1
lr=1e-5
precision=bf16
low_res_list=256,128,64
low_res_probs=0.4,0.4,0.2
high_res_list=1024
high_res_probs=1.0
snr_type=lognorm
controlnet_depth=2
backbone_depth=19

exp_name=${high_res_list}_${high_res_probs}_${low_res_list}_${low_res_probs}_controlnet_${controlnet_depth}_backbone_${backbone_depth}_snr_${snr_type}_0.5_1_cnet_snr_uniform_cfg_1.0_wo_noise_wo_shift
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
    --ckpt_every 4000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 8 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --high_res_list ${high_res_list} \
    --high_res_probs ${high_res_probs} \
    --low_res_list ${low_res_list} \
    --low_res_probs ${low_res_probs} \
    --use_wandb \
    --controlnet_depth ${controlnet_depth} \
    --backbone_depth ${backbone_depth} \
    # --checkpointing \
    
