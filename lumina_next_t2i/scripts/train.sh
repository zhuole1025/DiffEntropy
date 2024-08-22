#!/usr/bin/env sh

export HF_HOME="/data4/zl/.cache/huggingface"

train_data_root='configs/data/ft.yaml'

model=NextDiT_2B_GQA_patch2
batch_size=1
lr=1e-4
precision=bf16
image_size=1024
cfg_scale=4.0
vae=sdxl
init_from=checkpoints/lumina-next-sft

exp_name=test_${model}_bs${batch_size}_lr${lr}_${precision}_${image_size}px_vae${vae}
mkdir -p results/"$exp_name"

unset NCCL_IB_HCA
# export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=7 MASTER_ADDR=localhost torchrun --nproc-per-node=1 --master_port 18183 train.py \
    --master_port 18181 \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir /home/pgao/zl/zl/ckpt/${exp_name} \
    --micro_batch_size 1 \
    --global_batch_size ${batch_size} \
    --lr ${lr} \
    --grad_clip 2.0 \
    --data_parallel fsdp \
    --max_steps 30000 \
    --ckpt_every 1000 --log_every 10 \
    --precision ${precision} \
    --grad_precision fp32 \
    --qk_norm \
    --image_size ${image_size} \
    --global_seed 20240620 \
    --vae ${vae} \
    --num_workers 8 \
    --init_from $init_from \
    # 2>&1 | tee -a results/"$exp_name"/output.log
# 