#!/usr/bin/env sh

#SBATCH -p Gveval-S1
#SBATCH --gres=gpu:8
#SBATCH -n 8
#SBATCH --ntasks-per-node 8
#SBATCH --output slurm_output/%j.out
#SBATCH --error slurm_output/%j.err
#SBATCH --quotatype reserved
#SBATCH --job-name distill
#SBATCH --requeue
#SBATCH --open-mode=append

source ~/.bashrc
conda activate ldyacc2.3
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

export WANDB_MODE="offline"
export WANDB_API_KEY="75de1215548653cdc8084ae0d1450f2d84fd9a20"
export HF_TOKEN="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"
export HF_HUB_OFFLINE=1

train_data_root=configs/data/zl_all_256.yaml
batch_size=32
micro_batch_size=1
generator_lr=5e-7
guidance_lr=5e-7
precision=bf16
high_res_list=1024
high_res_probs=1.0
snr_type=uniform
num_discriminator_heads=8
real_guidance_scale=4.0
fake_guidance_scale=1.0
generator_guidance_scale=4.0
dfake_gen_update_ratio=5
gen_cls_loss_weight=5e-3
guidance_cls_loss_weight=1e-2

exp_name=test
mkdir -p results/"$exp_name"

# unset NCCL_IB_HCA
#export TOKENIZERS_PARALLELISM=false

# srun -p Gveval-S1 --gres=gpu:1 --cpus-per-task 8 --ntasks-per-node=1 --job-name lumina \
srun python train.py \
    --master_port 18181 \
    --global_bs ${batch_size} \
    --micro_bs ${micro_batch_size} \
    --data_path ${train_data_root} \
    --results_dir results/${exp_name} \
    --generator_lr ${generator_lr} \
    --guidance_lr ${guidance_lr} \
    --grad_clip 10.0 \
    --data_parallel fsdp \
    --max_steps 1000000 \
    --ckpt_every 1000 --log_every 1 \
    --precision ${precision} --grad_precision fp32 \
    --global_seed 20240826 \
    --num_workers 8 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --high_res_list ${high_res_list} \
    --high_res_probs ${high_res_probs} \
    --num_discriminator_heads ${num_discriminator_heads} \
    --real_guidance_scale ${real_guidance_scale} \
    --fake_guidance_scale ${fake_guidance_scale} \
    --generator_guidance_scale ${generator_guidance_scale} \
    --dfake_gen_update_ratio ${dfake_gen_update_ratio} \
    --gen_cls_loss_weight ${gen_cls_loss_weight} \
    --guidance_cls_loss_weight ${guidance_cls_loss_weight} \
    --checkpointing \
    --full_model \
    --load_t5 \
    --load_clip \
    --use_wandb \
    
