#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import math
from tqdm import tqdm
import torch
import sys
import torch.distributed as dist
from PIL import Image
from diffusers.models import AutoencoderKL
import models
import argparse
import numpy as np
import socket
import os
import fairscale.nn.model_parallel.initialize as fs_init
import json
from transport import create_transport, Sampler
import multiprocessing as mp


def count_images(file_path):
    # 初始化计数器
    png_count = 0
    max_idx = 0
    if os.path.exists(file_path):
        # 遍历文件夹中的文件
        for file in os.listdir(file_path):
            # 检查文件是否以'.png'结尾
            if file.endswith('.png'):
                cur_idx = int(file.split('.')[0])
                max_idx = max(cur_idx, max_idx)
                png_count += 1
    return png_count, max_idx


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def none_or_str(value):
    if value == 'None':
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise", "data"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

def main(args, rank, master_port, mode):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(1)
    device = "cuda"
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))

    if dist.get_rank() == 0:
        print("Model arguments used for inference:",
              json.dumps(train_args.__dict__, indent=2))

    # Load model:
    latent_size = train_args.image_size // 8
    model = models.__dict__[train_args.model](
        input_size=latent_size,
        num_classes=train_args.num_classes,
        qk_norm=train_args.qk_norm,
    )

    torch_dtype = {
        "fp32": torch.float, "tf32": torch.float,
        "bf16": torch.bfloat16, "fp16": torch.float16,
    }[args.precision]
    model.to(torch_dtype).cuda()
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # assert train_args.model_parallel_size == args.num_gpus
    if not args.ema:
        ckpt_path = os.path.join(args.ckpt, f"consolidated.00-of-01.pth")
    else:
        ckpt_path = os.path.join(args.ckpt, f"consolidated_ema.00-of-01.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(ckpt_path)
    model.load_state_dict(ckpt, strict=True)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{train_args.vae}" if train_args.local_diffusers_model_root is None else
        os.path.join(train_args.local_diffusers_model_root,
                     f"stabilityai/sd-vae-ft-{train_args.vae}")
        ).to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = train_args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{train_args.image_size}-vae-{train_args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        elif args.ode_imp:
            sample_fn = sampler.sample_ode_imp(
                num_steps=args.num_sampling_steps,
                reverse=args.reverse,
                time_shifting_factor=args.time_shifting_factor,
                start_time=0,
                end_time=0,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    already_sampled, start_img_idx = count_images(sample_folder_dir)
    rnd_path = f"{sample_folder_dir}/random_state.pth"
    if os.path.exists(rnd_path):
        print(f'resume rng state from: {rnd_path}')
        rng_state = torch.load(rnd_path)
        torch.random.set_rng_state(rng_state)

    print(f'already sampled: {already_sampled}')
    print(f'already start_img_idx: {start_img_idx}')
    total_samples = int(math.ceil((args.num_fid_samples - already_sampled) / global_batch_size) * global_batch_size) + global_batch_size
    if total_samples < 0:
        if rank == 0:
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            print("Done.")
        dist.barrier()
        exit(0)
    if rank == 0:
        print(f'already sampled: {already_sampled}')
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=torch_dtype)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        model_fn = model.forward_with_cfg

        all_samples = sample_fn(z, model_fn, **model_kwargs)
        samples = all_samples[-1].to(device)
        
        if args.save_traj:
            os.makedirs(f"{sample_folder_dir}/trajs", exist_ok=True)
            for t in range(all_samples.shape[0]):
                samples_t = all_samples[t].to(device)
                if using_cfg:
                    samples_t, _ = samples_t.chunk(2, dim=0)
                samples_t = vae.decode(samples_t.float() / 0.18215).sample
                samples_t = torch.clamp(127.5 * samples_t + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for i, sample in enumerate(samples_t):
                    index = i * dist.get_world_size() + rank + total + start_img_idx
                    Image.fromarray(sample).save(f"{sample_folder_dir}/trajs/{index:06d}_t{t:03d}.png")
                
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples.float() / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total + start_img_idx
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        torch.save(torch.random.get_rng_state(), rnd_path)
        total += global_batch_size
        dist.barrier()

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--num-fid-samples", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1000)



    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--class_labels", type=int, nargs="+",
        help="Class labels to generate the images for.",
        default=[207, 360, 387, 974, 88, 979, 417, 279],
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"],
        default="tf32",
    )
    parser.add_argument(
        "--local_diffusers_model_root", type=str,
        help="Specify the root directory if diffusers models are to be loaded "
             "from the local filesystem (instead of being automatically "
             "downloaded from the Internet). Useful in environments without "
             "Internet access."
    )
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument(
        "--time_shifting_factor", type=float, default=1.0,
    )
    parser.add_argument(
        "--ode_imp", action="store_true",
    )
    parser.add_argument(
        "--save_traj", action="store_true",
    )
    
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument("--no_ema", action="store_false", dest="ema", help="Do not use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--sample-dir", type=str,
        help="If specified, overrides the default image save path "
             "(sample{_ema}.png in the model checkpoint directory)."
    )
    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    mp.set_start_method("spawn")
    # main(args, master_port)
    procs = []
    for i in range(args.num_gpus):
        p = mp.Process(target=main, args=(args, i, master_port, mode))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
