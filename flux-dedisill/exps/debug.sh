#!/bin/bash

export WANDB_API_KEY="75de1215548653cdc8084ae0d1450f2d84fd9a20"
export HF_TOKEN="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"
export HF_HOME="/ceph/data-bk/huggingface"

train_data_root='configs/data/2M.yaml'
batch_size=4
micro_batch_size=1
lr=1e-5
precision=bf16
low_res_list=1024
low_res_probs=1.0
high_res_list=1024
high_res_probs=1.0
snr_type=lognorm

exp_name=test
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --nnodes=1 --master_port 29338 train.py \
    --master_port 18181 \
    --global_bs ${batch_size} \
    --micro_bs ${micro_batch_size} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 1000 --log_every 1 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 1 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --high_res_list ${high_res_list} \
    --high_res_probs ${high_res_probs} \
    --low_res_list ${low_res_list} \
    --low_res_probs ${low_res_probs} \
    --loss_type mse \
    --debug \
    --load_t5 \
    --load_clip \
    # --zero_init \
    # --use_wandb
    
