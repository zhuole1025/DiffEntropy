#!/bin/bash

export WANDB_API_KEY="75de1215548653cdc8084ae0d1450f2d84fd9a20"
export HF_TOKEN="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"
export HF_HOME="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/gaopeng/public/zl/huggingface"

train_data_root='configs/data/2M.yaml'
batch_size=32
micro_batch_size=1
lr=1e-4
precision=bf16
low_res_list=1024
low_res_probs=1.0
high_res_list=1024
high_res_probs=1.0
snr_type=lognorm

exp_name=${high_res_list}_lr_${lr}_bsz_${batch_size}_1_10_cfg1_sg_cond_lora
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 --nnodes=1 --master_port 29311 train_lora.py \
    --master_port 18181 \
    --global_bs ${batch_size} \
    --micro_bs ${micro_batch_size} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --lr ${lr} --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 2000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 8 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --high_res_list ${high_res_list} \
    --high_res_probs ${high_res_probs} \
    --low_res_list ${low_res_list} \
    --low_res_probs ${low_res_probs} \
    --load_t5 \
    --load_clip \
    --full_model \
    --checkpointing \
    --caption_dropout_prob 0.1 \
    --full_model \