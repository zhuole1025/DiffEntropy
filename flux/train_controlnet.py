# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import math
import random
import socket
from time import time
import warnings
from safetensors import safe_open
from safetensors.torch import load_file
import wandb

from PIL import Image
import numpy as np
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, circular_lowpass_kernel, random_mixed_kernels

from data import ItemProcessor, MyDataset
from flux.sampling import prepare
from flux.util import load_ae, load_clip, load_flow_model, load_t5, load_controlnet
from flux.modules.image_embedders import ReduxImageEncoder
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from util.misc import SmoothedValue

#############################################################################
#                            Data item Processor                            #
#############################################################################


def invert_transform(x):
    x = x * 0.5 + 0.5
    x = torch.clamp(x, 0, 1)
    x = transforms.ToPILImage()(x)
    return x


class T2IItemProcessor(ItemProcessor):
    def __init__(self, high_res_list, high_res_probs, low_res_list, low_res_probs, downsample_factor=16):
        # prepare hyper-parameters for resizing
        if len(high_res_list) != 1:
            raise ValueError("Currently only support single resolution for high-res images")
        self.high_res_list = high_res_list
        self.high_res_probs = high_res_probs
        self.low_res_list = low_res_list
        self.low_res_probs = low_res_probs
        self.crop_size_dict = {}
        for high_res in high_res_list:
            scale = high_res // min(low_res_list)
            max_num_tokens = round((high_res / downsample_factor) ** 2)
            self.crop_size_dict[high_res] = generate_crop_size_list(max_num_tokens, downsample_factor, step_size=scale)
        
        # prepare hyper-parameters for degradation
        self.sinc_prob = 0.1
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        
        self.sinc_prob2 = 0.1
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        
        self.final_sinc_prob = 0.8
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1.0
        
        self.resize_prob = [0.2, 0.7, 0.1] # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.jpeg_range = [30, 95]
        
        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.3, 0.4, 0.3]
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        self.jpeg_range2 = [30, 95]
        
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp()  # do usm sharpening
    
    @torch.no_grad()
    def get_condition(self, image, ds_factor):
        # image = self.usm_sharpener(image)
        ori_h, ori_w = image.shape[-2], image.shape[-1]
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel1 = torch.FloatTensor(kernel).to(image.device)
        kernel2 = torch.FloatTensor(kernel2).to(image.device)
        sinc_kernel = sinc_kernel.to(image.device)
        
        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(image, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.gray_noise_prob
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / ds_factor * scale), int(ori_w / ds_factor * scale)), mode=mode)
        # add noise
        gray_noise_prob = self.gray_noise_prob2
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // ds_factor, ori_w // ds_factor), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // ds_factor, ori_w // ds_factor), mode=mode)
            out = filter2D(out, sinc_kernel)

        # clamp and round
        cond_image = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        cond_image = cond_image.contiguous()
        
        return image, cond_image

    def process_item(self, data_item, training_mode=False):
        text_emb = None
        if "path" in data_item:
            image_path = data_item["path"]
            image = Image.open(image_path).convert("RGB")
            text = data_item["gpt_4_caption"]
            # if "text_embeddings_path" in data_item:
                # text_emb = load_file(data_item["text_embeddings_path"], device="cpu")
        else:
            raise ValueError(f"Unrecognized item: {data_item}")
        
        if random.random() < 0.5:
            is_tiled = True
        else:
            is_tiled = False
        
        high_res = random.choices(self.high_res_list, self.high_res_probs)[0]
        low_res = random.choices(self.low_res_list, self.low_res_probs)[0]
        ds_factor = high_res // low_res
        image = to_rgb_if_rgba(image)
        image = var_center_crop(image, crop_size_list=self.crop_size_dict[high_res], random_top_k=1, is_tiled=is_tiled)
        image = transforms.ToTensor()(image)
        image, cond_image = self.get_condition(image[None, ...], ds_factor)
        image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
        cond_image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(cond_image)

        return image, cond_image, text, text_emb


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    cond_image = [x[1] for x in samples]
    caps = [x[2] for x in samples]
    text_emb = [x[3] for x in samples]
    if all(x is None for x in text_emb):
        text_emb = None
    return image, cond_image, caps, text_emb


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module, auto_wrap_policy) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    device_str = f"cuda:{device}"
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        # Create wandb logger
        if args.use_wandb:
            wandb.init(
                project="FLUX",
                name=args.results_dir.split("/")[-1],
                config=args.__dict__,  # Use args.__dict__ to pass all arguments
                dir=args.results_dir,  # Set the directory for wandb files
                job_type="training",
                reinit=True,  # Allows multiple runs in the same process
            )
    else:
        logger = create_logger(None)

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    if args.load_t5:
        t5 = load_t5(max_length=512)
        t5 = setup_lm_fsdp_sync(
            t5,
            functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in list(t5.hf_module.encoder.block),
            ),
        )
        logger.info("T5 loaded")
    else:
        t5 = None

    if args.load_clip:
        clip = load_clip()
        clip = setup_lm_fsdp_sync(
            clip,
            functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in list(clip.hf_module.text_model.encoder.layers),
            ),
        )
        logger.info(f"CLIP loaded")
    else:
        clip = None
        
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    ae.requires_grad_(False)
    logger.info(f"VAE loaded")
    
    if args.img_embedder_path is not None:
        img_embedder = ReduxImageEncoder(device=device_str, redux_path=args.img_embedder_path)
        img_embedder.requires_grad_(False)
        logger.info(f"Image embedder loaded")
    else:
        img_embedder = None
    
    model = load_flow_model("flux-dev", device=device_str, dtype=torch.bfloat16, attn_token_select=args.attn_token_select, mlp_token_select=args.mlp_token_select, zero_init=args.zero_init, learnable_gate=args.learnable_gate)
    # for block in model.double_blocks:
        # block.init_cond_weights()
    model_params = []
    controlnet_params = []
    for name, param in model.named_parameters():
        if args.full_model:
            param.requires_grad = True
            model_params.append(param)
        # elif 'norm' in name or 'bias' in name:
            # param.requires_grad = True
            # model_params.append(param)
        elif 'controlnet' in name:
            param.requires_grad = True
            controlnet_params.append(param)
        else:
            param.requires_grad = False
    
    controlnet = load_controlnet("flux-dev", device=device_str, dtype=torch.bfloat16, transformer=model, double_depth=args.double_depth, single_depth=args.single_depth, backbone_depth=args.backbone_depth, backbone_depth_single=args.backbone_depth_single, compute_loss=args.compute_controlnet_loss)
    if args.learnable_gate:
        controlnet.requires_grad_(False)
    else:
        controlnet.train()
        controlnet_params = controlnet_params + [p for p in controlnet.parameters() if p.requires_grad]
    
    total_params = model.parameter_count()
    size_in_gb = total_params * 4 / 1e9
    logger.info(f"Model Size: {size_in_gb:.2f} GB, Total Parameters: {total_params / 1e9:.2f} B, Trainable Parameters: {sum(p.numel() for p in model_params) / 1e9:.3f} B, ControlNet Trainable Parameters: {sum(p.numel() for p in controlnet_params) / 1e9:.3f} B")

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            controlnet.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_controlnet.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)
            model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            
            controlnet.load_state_dict(
                torch.load(
                    os.path.join(
                        args.init_from,
                        f"consolidated_controlnet.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.get_checkpointing_wrap_module_list())
        checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
        checkpointing_list_controlnet = list(controlnet.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
        checkpointing_list_ema = []
        checkpointing_list_controlnet = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)
    controlnet = setup_fsdp_sync(controlnet, args)
    
    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )
        apply_activation_checkpointing(
            controlnet,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_controlnet,
        )

    logger.info(f"model:\n{model}\n")
    logger.info(f"controlnet:\n{controlnet}\n")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    if len(model_params) > 0 and len(controlnet_params) > 0:
        opt = torch.optim.AdamW([
            {'params': model_params, 'lr': args.lr * 0.1},
            {'params': controlnet_params, 'lr': args.lr}
        ], weight_decay=args.wd)
    elif len(model_params) > 0:
        opt = torch.optim.AdamW(model_params, lr=args.lr, weight_decay=args.wd)
    elif len(controlnet_params) > 0:
        opt = torch.optim.AdamW(controlnet_params, lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError("No trainable parameters found in either model or controlnet")

    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr  # todo learning rate and weight decay
            param_group["weight_decay"] = args.wd  # todo learning rate and weight decay

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0
    
    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
        snr_type=args.snr_type,
        do_shift=args.do_shift,
        token_target_ratio=args.token_target_ratio,
        token_loss_weight=args.token_loss_weight,
    )  # default: velocity;

    # Setup data:
    logger.info(f"Creating data")
    global_bsz = args.global_bsz
    local_bsz = global_bsz // dp_world_size  # todo caution for sequence parallel
    micro_bsz = args.micro_bsz
    num_samples = global_bsz * args.max_steps
    assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
    logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")
    patch_size = 16
    logger.info(f"patch size: {patch_size}")
    dataset = MyDataset(
        args.data_path,
        train_res=None,
        item_processor=T2IItemProcessor(args.high_res_list, args.high_res_probs, args.low_res_list, args.low_res_probs),
        cache_on_disk=args.cache_data_on_disk,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
    sampler = get_train_sampler(
        dataset,
        dp_rank,
        dp_world_size,
        global_bsz,
        args.max_steps,
        resume_step,
        args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_bsz,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
        # prefetch_factor=3,
    )

    data_pack = {
        "loader": loader,
        "loader_iter": iter(loader),
        "global_bsz": global_bsz,
        "local_bsz": local_bsz,
        "micro_bsz": micro_bsz,
        "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
        "transport": transport,
    }

    # Prepare models for training:
    model.train()

    # Variables for monitoring/logging purposes:
    logger.info(f"Training for {args.max_steps:,} steps...")
    start_time = time()
    for step in range(resume_step, args.max_steps):
        x, x_cond, caps, text_emb = next(data_pack["loader_iter"])
        x = [img.to(device, non_blocking=True) for img in x]
        x_cond = [img.to(device, non_blocking=True) for img in x_cond]
        with torch.no_grad():
            if img_embedder is not None:
                raw_x_cond = [invert_transform(img.squeeze(0)) for img in x_cond]
            else:
                raw_x_cond = None
            x_cond = [F.interpolate(x_cond[i], size=(x[i].shape[-2], x[i].shape[-1]), mode="bilinear", align_corners=False) for i in range(len(x_cond))]
            x_cond = [(ae.encode(img.to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x_cond]
            if False:
                x = [(ae.tiled_encode(img.to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x]
            else:
                x = [(ae.encode(img.to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor for img in x]
            
        with torch.no_grad():
            inp = prepare(t5=t5, clip=clip, img=x, img_cond=x_cond, prompt=caps, proportion_empty_prompts=args.caption_dropout_prob, proportion_empty_images=args.image_dropout_prob, text_emb=text_emb, img_embedder=img_embedder, raw_img_cond=raw_x_cond, cond_type=args.cond_type)

        loss_item = 0.0
        controlnet_loss_item = 0.0
        opt.zero_grad()
        
        # Number of bins, for loss recording
        n_loss_bins = 20
        # Create bins for t
        loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
        # Initialize occurrence and sum tensors
        bin_occurrence = torch.zeros(n_loss_bins, device="cuda")
        bin_sum_loss = torch.zeros(n_loss_bins, device="cuda")
            
        for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
            mb_st = mb_idx * data_pack["micro_bsz"]
            mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
            last_mb = mb_ed == data_pack["local_bsz"]

            x_mb = inp["img"][mb_st:mb_ed]
            model_kwargs = dict(
                img_ids=inp["img_ids"][mb_st:mb_ed],
                txt=inp["txt"][mb_st:mb_ed],
                txt_ids=inp["txt_ids"][mb_st:mb_ed],
                txt_mask=inp["txt_mask"][mb_st:mb_ed],
                y=inp["vec"][mb_st:mb_ed],
                img_mask=inp["img_mask"][mb_st:mb_ed],
                guidance=torch.full((x_mb.shape[0],), args.backbone_cfg, device=x_mb.device, dtype=x_mb.dtype),
            )
            
            controlnet_kwargs = dict(
                img_ids=inp["img_ids"][mb_st:mb_ed],
                controlnet_cond=inp["img_cond"][mb_st:mb_ed],
                txt=inp["txt"][mb_st:mb_ed],
                txt_ids=inp["txt_ids"][mb_st:mb_ed],
                y=inp["vec"][mb_st:mb_ed],
                txt_mask=inp["txt_mask"][mb_st:mb_ed],
                img_mask=inp["img_mask"][mb_st:mb_ed],
                guidance=torch.full((x_mb.shape[0],), args.controlnet_cfg, device=x_mb.device, dtype=x_mb.dtype),
                controlnet_snr=args.controlnet_snr,
            )
            
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = data_pack["transport"].training_losses(model, x_mb, model_kwargs, controlnet=controlnet, controlnet_kwargs=controlnet_kwargs)
            loss = loss_dict["loss"].sum() / data_pack["local_bsz"]
            loss_item += loss.item()
            if args.compute_controlnet_loss:
                controlnet_loss = loss_dict["controlnet_loss"].sum() / data_pack["local_bsz"]
                controlnet_loss_item += controlnet_loss.item()
            else:
                controlnet_loss_item += 0.0
            
            with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()
                
            # for bin-wise loss recording
            # Digitize t values to find which bin they belong to
            bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
            detached_loss = loss_dict["loss"].detach()
            
            # Iterate through each bin index to update occurrence and sum
            for i in range(n_loss_bins):
                mask = bin_indices == i  # Mask for elements in the i-th bin
                bin_occurrence[i] = bin_occurrence[i] + mask.sum()  # Count occurrences in the i-th bin
                bin_sum_loss[i] = bin_sum_loss[i] + detached_loss[mask].sum()  # Sum loss values in the i-th bin
        
        if len(model_params) > 0:
            grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)
        else:
            grad_norm = 0.0
        if len(controlnet_params) > 0:
            controlnet_grad_norm = controlnet.clip_grad_norm_(max_norm=args.grad_clip)
        else:
            controlnet_grad_norm = 0.0
            
        dist.all_reduce(bin_occurrence)
        dist.all_reduce(bin_sum_loss)

        if args.use_wandb and rank == 0:
            log_dict = {
                "train/loss": loss_item,
                "train/controlnet_loss": controlnet_loss_item,
                "train/grad_norm": grad_norm,
                "train/controlnet_grad_norm": controlnet_grad_norm,
                "train/lr": opt.param_groups[0]["lr"],
            }
            for i in range(n_loss_bins):
                if bin_occurrence[i] > 0:
                    bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                    log_dict[f"train/loss-bin{i+1}-{n_loss_bins}"] = bin_avg_loss
            wandb.log(log_dict, step=step)

        opt.step()
        end_time = time()

        # Log loss values:
        metrics = data_pack["metrics"]
        metrics["loss"].update(loss_item)
        metrics["controlnet_loss"].update(controlnet_loss_item)
        metrics["grad_norm"].update(grad_norm)
        metrics["controlnet_grad_norm"].update(controlnet_grad_norm)
        metrics["Secs/Step"].update(end_time - start_time)
        metrics["Imgs/Sec"].update(data_pack["global_bsz"] / (end_time - start_time))
        for i in range(n_loss_bins):
            if bin_occurrence[i] > 0:
                bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                metrics[f"bin{i + 1:02}-{n_loss_bins}"].update(bin_avg_loss, int(bin_occurrence[i].item()))
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            logger.info(
                f"(step{step + 1:07d}) "
                + f"lr{opt.param_groups[0]['lr']:.6f} "
                + " ".join([f"{key}:{str(metrics[key])}" for key in sorted(metrics.keys())])
            )

        start_time = time()

        update_ema(model_ema, model)

        # Save DiT checkpoint:
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            if len(model_params) > 0:
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if fs_init.get_data_parallel_rank() == 0:
                        consolidated_fn = (
                            "consolidated."
                            f"{fs_init.get_model_parallel_rank():02d}-of-"
                            f"{fs_init.get_model_parallel_world_size():02d}"
                            ".pth"
                        )
                        torch.save(
                            consolidated_model_state_dict,
                            os.path.join(checkpoint_path, consolidated_fn),
                        )
                dist.barrier()
                del consolidated_model_state_dict
                logger.info(f"Saved consolidated to {checkpoint_path}.")

                with FSDP.state_dict_type(
                    model_ema,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_ema_state_dict = model_ema.state_dict()
                    if fs_init.get_data_parallel_rank() == 0:
                        consolidated_ema_fn = (
                            "consolidated_ema."
                            f"{fs_init.get_model_parallel_rank():02d}-of-"
                            f"{fs_init.get_model_parallel_world_size():02d}"
                            ".pth"
                        )
                        torch.save(
                            consolidated_ema_state_dict,
                            os.path.join(checkpoint_path, consolidated_ema_fn),
                        )
                dist.barrier()
                del consolidated_ema_state_dict
                logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

                with FSDP.state_dict_type(
                    model,
                    StateDictType.LOCAL_STATE_DICT,
                ):
                    opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                    torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
                dist.barrier()
                logger.info(f"Saved optimizer to {checkpoint_path}.")
            
            with FSDP.state_dict_type(
                controlnet,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = controlnet.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated_controlnet."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved controlnet consolidated to {checkpoint_path}.")
            
            with FSDP.state_dict_type(
                controlnet,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer_controlnet.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved controlnet optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_bsz", type=int, default=256)
    parser.add_argument("--micro_bsz", type=int, default=1)
    parser.add_argument("--load_t5", action="store_true")
    parser.add_argument("--load_clip", action="store_true")
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--image_dropout_prob", type=float, default=0.0)
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--do_shift", default=False)
    parser.add_argument(
        "--no_shift",
        action="store_false",
        dest="do_shift",
        help="Do dynamic time shifting",
    )
    parser.add_argument(
        "--low_res_list",
        type=str,
        default="256,512,1024",
        help="Comma-separated list of low resolution for training."
    )
    parser.add_argument(
        "--high_res_list",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated list of high resolution for training."
    )
    parser.add_argument(
        "--high_res_probs",
        type=str,
        default="0.2,0.7,0.1",
        help="Comma-separated list of probabilities for sampling high resolution images."
    )
    parser.add_argument(
        "--low_res_probs",
        type=str,
        default="0.2,0.7,0.1",
        help="Comma-separated list of probabilities for sampling low resolution images."
    )
    parser.add_argument("--cond_type", type=str, default="image")
    parser.add_argument("--learnable_gate", action="store_true")
    parser.add_argument("--backbone_cfg", type=float, default=1.0)
    parser.add_argument("--controlnet_cfg", type=float, default=1.0)
    parser.add_argument("--compute_controlnet_loss", action="store_true")
    parser.add_argument("--controlnet_snr", type=str, default=None)
    parser.add_argument("--img_embedder_path", type=str, default=None)
    parser.add_argument("--double_depth", type=int, default=2)
    parser.add_argument("--single_depth", type=int, default=0)
    parser.add_argument("--backbone_depth", type=int, default=19)
    parser.add_argument("--backbone_depth_single", type=int, default=0)
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--token_target_ratio", type=float, default=0.5)
    parser.add_argument("--token_loss_weight", type=float, default=1.0)
    parser.add_argument("--attn_token_select", action="store_true")
    parser.add_argument("--mlp_token_select", action="store_true")
    parser.add_argument("--zero_init", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.low_res_list = [int(res) for res in args.low_res_list.split(",")]
    args.high_res_list = [int(res) for res in args.high_res_list.split(",")]
    args.high_res_probs = [float(prob) for prob in args.high_res_probs.split(",")]
    args.low_res_probs = [float(prob) for prob in args.low_res_probs.split(",")]
    main(args)
