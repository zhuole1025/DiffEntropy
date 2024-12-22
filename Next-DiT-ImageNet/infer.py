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
from torch.utils.data import DataLoader
from PIL import Image
from diffusers.models import AutoencoderKL
import models
import argparse
import functools
import numpy as np
import socket
import os
import fairscale.nn.model_parallel.initialize as fs_init
import json
from transport import create_transport
from torchvision import transforms
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def none_or_str(value):
    if value == 'None':
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--snr-type", type=str, default="lognorm", choices=["uniform", "lognorm"])
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(args.num_gpus)
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
    ckpt = torch.load(os.path.join(
        args.ckpt,
        f"consolidated{'_ema' if args.ema else ''}."
        f"{rank:02d}-of-{args.num_gpus:02d}.pth",
    ), map_location="cpu")
    model.load_state_dict(ckpt, strict=True)

    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-ema"
        if args.local_diffusers_model_root is None else
        os.path.join(args.local_diffusers_model_root,
                     f"stabilityai/sd-vae-ft-{train_args.vae}")
    ).cuda()
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        args.snr_type,
    )
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(
            functools.partial(center_crop_arr, image_size=train_args.image_size)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                             inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )
    
    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving results at {sample_folder_dir}")
    dist.barrier()

    all_t = torch.linspace(0, 1, 51)
    for step, (x, y) in enumerate(tqdm(loader)):
        if step >= args.max_steps:
            break
        x = x.to("cuda")
        y = y.to("cuda")
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x0 = torch.rand_like(x1)
        x_pred_list = [x0.detach().cpu().numpy(), x1.detach().cpu().numpy()]
        for idx, t in tqdm(zip(range(len(all_t) - 1), all_t[:-1])):
            with torch.no_grad():
                xt = t * x1 + (1 - t) * x0
                timestep = torch.ones((x1.shape[0],), device="cuda") * t
                model_kwargs = dict(y=y, t=timestep)
                x_pred = model(xt, **model_kwargs)
                
                x_pred_list.append(x_pred.detach().cpu().numpy())
        all_x_pred = np.stack(x_pred_list)
        
        if rank == 0:
            np.save(os.path.join(sample_folder_dir, f"all_x_pred_{step}.npy"), all_x_pred)
            
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

    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument("--no_ema", action="store_false", dest="ema", help="Do not use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path", type=str,
        help="If specified, overrides the default image save path "
             "(sample{_ema}.png in the model checkpoint directory)."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parse_transport_args(parser)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
