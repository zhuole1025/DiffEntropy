#!/usr/bin/env sh

train_data_root='configs/data/sc14e6.yaml'

model=NextDiT_2B_GQA_patch2
batch_size=512
lr=$1
precision=bf16
image_size=256
snr_type=uniform

exp_name=8_sc14e6_${model}_bs${batch_size}_lr${lr}_${precision}_${image_size}px_snr${snr_type}
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

python -u train.py \
    --master_port 18181 \
    --model ${model} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --micro_batch_size 16 \
    --global_batch_size ${batch_size} --lr ${lr} --grad_clip 2.0 \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 10000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --image_size ${image_size} \
    --global_seed 20240825 \
    --num_workers 8 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    2>&1 | tee -a results/"$exp_name"/output.log