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
import random
import socket
from time import time
import warnings
from safetensors import safe_open
from safetensors.torch import load_file as load_sft
from huggingface_hub import hf_hub_download
import wandb

from PIL import Image
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

from data import ItemProcessor, MyDataset, AspectRatioBatchSampler, read_general
from flux.sampling import prepare
from flux.model import Flux, FluxParams, FluxLoraWrapper, FluxUnifiedWrapper
from flux.util import load_ae, load_clip, load_flow_model, load_t5, configs, print_load_warning
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop, ASPECT_RATIO_1024, ASPECT_RATIO_2048, ASPECT_RATIO_256, ASPECT_RATIO_512
from parallel import distributed_init, get_intra_node_process_group
from util.misc import SmoothedValue

#############################################################################
#                            Data item Processor                            #
#############################################################################


class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform

    def process_item(self, data_item, training_mode=False):
        text_emb = None
        if "conversations" in data_item:
            assert "image" in data_item and len(data_item["conversations"]) == 2
            image = Image.open(read_general(data_item["image"])).convert("RGB")
            text = data_item["conversations"][1]["value"]
        elif "path" in data_item:
            image_path = data_item["path"]
            image = Image.open(image_path).convert("RGB")
            text = data_item["gpt_4_caption"]
            if "text_embeddings_path" in data_item:
                with safe_open(data_item["text_embeddings_path"], framework="pt", device="cpu") as f:
                    text_emb = {k: f.get_tensor(k) for k in f.keys()}
        elif "image_path" in data_item:
            url = data_item["image_path"]
            image = Image.open(read_general(url)).convert("RGB")
            text = data_item["prompt"]
        elif "image_url" in data_item:
            url = data_item["image_url"]
            url = url.replace(
                "/mnt/petrelfs/share_data/gaopeng/image_text_data",
                "/mnt/hwfile/alpha_vl/gaopeng/share_data/image_text_data",
            )
            url = url.replace("/mnt/petrelfs/share_data/huxiangfei", "/mnt/hwfile/alpha_vl/huxiangfei")
            image = Image.open(read_general(url)).convert("RGB")
            caption_keys = [
                "sharegpt4v_long_cap",
                "cogvlm_long",
                "blip2_short_cap",
                "llava13b_long_cap",
                "spatial_caption",
                "coca_caption",
                "user_prompt",
                "tags_prompt",
                "gpt4v_concise_elements",
                "gpt4v_regions_detailed_description",
                "gpt4v_detailed_description",
                "gpt4v_concise_description",
                'cogvlm_long_user_prompt_conditioned',
                'internvl_V1.5_user_prompt_conditioned',
                'cogvlm2_long_user_prompt_conditioned',
                'florence2_cap',
            ]
            candidate_caps = [data_item[x] for x in caption_keys if x in data_item and data_item[x]]
            text = random.choice(candidate_caps) if len(candidate_caps) else ""
        elif "image" in data_item and "caption" in data_item:
            image = Image.open(read_general(data_item["image"].replace("hzh:s3://", "cluster_s_hdd_gp:s3://"))).convert(
                "RGB"
            )
            text = data_item["caption"]
        else:
            raise ValueError(f"Unrecognized item: {data_item}")

        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            if mode not in self.special_format_set:
                self.special_format_set.add(mode)
            if mode == "RGBA":
                image = to_rgb_if_rgba(image)
            else:
                image = image.convert("RGB")
        image = self.image_transform(image)

        return image, text, text_emb


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples]
    text_emb = [x[2] for x in samples]
    return image, caps, text_emb


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


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, rank: int) -> FSDP:
    
    if rank == 0:
        param_init_fn = None
    else:
        param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)
        
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
        param_init_fn=param_init_fn,
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


def accumulate_log_dict(accumulated, new_dict):
    """Recursively accumulate values from new_dict into accumulated dict."""
    if accumulated is None:
        return new_dict
        
    for k, v in new_dict.items():
        if isinstance(v, torch.Tensor):
            accumulated[k] = torch.cat([accumulated[k], v], dim=0)
        elif isinstance(v, list):
            accumulated[k].extend(v)  # Using extend instead of += for better performance
        elif isinstance(v, dict):
            accumulated[k] = accumulate_log_dict(accumulated[k], v)
        else:
            accumulated[k] = v  # For other types, just update
    return accumulated


def encode_in_chunks(ae, x, chunk_size=1):
    """Encode images in smaller chunks to avoid OOM."""
    num_chunks = (x.shape[0] + chunk_size - 1) // chunk_size
    latents = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, x.shape[0])
        chunk = x[start_idx:end_idx]
        
        with torch.no_grad():
            chunk_latent = ae.encode(chunk.to(ae.dtype)).latent_dist.sample()
            chunk_latent = (chunk_latent - ae.config.shift_factor) * ae.config.scaling_factor
            latents.append(chunk_latent)
            
    return torch.cat(latents, dim=0)


def prepare_images_for_saving(images_tensor, height, width, grid_size=4, range_type="neg1pos1"):
    if range_type != "uint8":
        images_tensor = (images_tensor * 0.5 + 0.5).clamp(0, 1) * 255

    images = images_tensor[:grid_size*grid_size].permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
    grid = images.reshape(grid_size, grid_size, height, width, 3)
    grid = np.swapaxes(grid, 1, 2).reshape(grid_size*height, grid_size*width, 3)
    return grid


def draw_valued_array(data, output_dir, grid_size=4):
    fig = plt.figure(figsize=(20,20))

    data = data[:grid_size*grid_size].reshape(grid_size, grid_size)
    cax = plt.matshow(data, cmap='viridis')  # Change cmap to your desired color map
    plt.colorbar(cax)

    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='black')

    plt.savefig(os.path.join(output_dir, "cache.jpg"))
    plt.close('all')

    # read the image 
    image = imageio.imread(os.path.join(output_dir, "cache.jpg"))
    return image


def gather_images(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gathers a 4D or 2D tensor from all ranks and concatenates along dim=0.
    """
    world_size = dist.get_world_size()
    # create placeholder tensors on each rank
    tensors_gather = [torch.empty_like(tensor) for _ in range(world_size)]
    # gather each rank's tensor into tensors_gather
    dist.all_gather(tensors_gather, tensor)
    # concatenate them along dim=0
    return torch.cat(tensors_gather, dim=0)


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
            run = wandb.init(
                project="FLUX-distill",
                name=args.results_dir.split("/")[-1],
                config=args.__dict__,  # Use args.__dict__ to pass all arguments
                dir=args.results_dir,  # Set the directory for wandb files
                job_type="training",
                reinit=True,  # Allows multiple runs in the same process
            )
            wandb_folder = run.dir
            os.makedirs(wandb_folder, exist_ok=True)
    else:
        logger = create_logger(None)

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))
    
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    ae.requires_grad_(False)
    logger.info("VAE loaded")
    
    config = configs["flux-dev"]
    ckpt_path = hf_hub_download(config.repo_id, config.repo_flow)
    if rank == 0:
        model = FluxUnifiedWrapper(params=config.params, num_discriminator_heads=args.num_discriminator_heads, snr_type=args.snr_type, do_shift=args.do_shift, grid_size=args.grid_size, device=device_str, offload_to_cpu=True, vae=ae, num_denoising_step=args.num_denoising_step, min_step_percent=args.min_step_percent, max_step_percent=args.max_step_percent, real_guidance_scale=args.real_guidance_scale, fake_guidance_scale=args.fake_guidance_scale, generator_guidance_scale=args.generator_guidance_scale, generator_loss_type=args.generator_loss_type)
        sd = load_sft(ckpt_path, device=str(device_str))
        model.load_state_dict(sd)
        del sd
    else:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = FluxUnifiedWrapper(params=config.params, num_discriminator_heads=args.num_discriminator_heads, snr_type=args.snr_type, do_shift=args.do_shift, grid_size=args.grid_size, device=device_str, offload_to_cpu=False, vae=ae, num_denoising_step=args.num_denoising_step, min_step_percent=args.min_step_percent, max_step_percent=args.max_step_percent, real_guidance_scale=args.real_guidance_scale, fake_guidance_scale=args.fake_guidance_scale, generator_guidance_scale=args.generator_guidance_scale, generator_loss_type=args.generator_loss_type)
    torch.cuda.empty_cache()
    logger.info("DiT loaded")

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

    generator_params = model.generator.parameters()
    guidance_params = model.fake_model.parameters()
    model.real_model.requires_grad_(False)
    model.fake_model.requires_grad_(True)
    model.generator.requires_grad_(True)
    # guidance_params = []
    # for name, param in model.guidance.named_parameters():
    #     if 'lora' in name:
    #         param.requires_grad = True
    #         guidance_params.append(param)
    #     elif 'discriminator' in name:
    #         param.requires_grad = True
    #         guidance_params.append(param)
    #     else:
    #         param.requires_grad = False
    total_params = model.generator.parameter_count()
    size_in_gb = total_params * 4 / 1e9
    logger.info(f"Model Size: {size_in_gb:.2f} GB, Student Parameters: {total_params / 1e9:.2f} B, Guidance Parameters: {sum(p.numel() for p in guidance_params) / 1e9:.2f} B")
    
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
    # model_ema = deepcopy(model)
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

            # logger.info(f"Resuming ema weights from: {args.resume}")
            # model_ema.load_state_dict(
            #     torch.load(
            #         os.path.join(
            #             args.resume,
            #             f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
            #         ),
            #         map_location="cpu",
            #     ),
            #     strict=True,
            # )
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

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            # missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            # assert set(missing_keys) == set(missing_keys_ema)
            # assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    dist.barrier()
    
    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.generator.get_checkpointing_wrap_module_list())
        checkpointing_guidance_list = list(model.fake_model.get_checkpointing_wrap_module_list())
        # checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
        # checkpointing_list_ema = []
    
    model.generator = setup_fsdp_sync(model.generator, args, rank)
    model.fake_model = setup_fsdp_sync(model.fake_model, args, rank)
    model.real_model = setup_fsdp_sync(model.real_model, args, rank)
    # model_ema = setup_fsdp_sync(model_ema, args)
    
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
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_guidance_list,
        )
        # apply_activation_checkpointing(
        #     model_ema,
        #     checkpoint_wrapper_fn=non_reentrant_wrapper,
        #     check_fn=lambda submodule: submodule in checkpointing_list_ema,
        # )

    logger.info(f"model:\n{model}\n")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    opt_generator = torch.optim.AdamW([
        {'params': generator_params, 'lr': args.generator_lr, 'weight_decay': args.wd}
    ])
    opt_guidance = torch.optim.AdamW([
        {'params': guidance_params, 'lr': args.guidance_lr, 'weight_decay': args.wd}
    ])
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer_generator.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt_generator.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer_generator.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        opt_guidance.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer_guidance.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt_generator.param_groups:
            param_group["lr"] = args.generator_lr
            param_group["weight_decay"] = args.wd
        for param_group in opt_guidance.param_groups:
            param_group["lr"] = args.guidance_lr 
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0
    
    # Setup data:
    logger.info(f"Creating data")
    data_collection = {}
    global_bsz = args.global_bsz
    local_bsz = global_bsz // dp_world_size  # todo caution for sequence parallel
    micro_bsz = args.micro_bsz
    num_samples = global_bsz * args.max_steps
    assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
    logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")
    patch_size = 16
    logger.info(f"patch size: {patch_size}")
    for train_res in args.high_res_list:
        image_transform = transforms.Compose(
            [
                transforms.Lambda(functools.partial(var_center_crop, crop_size_dict=eval(f'ASPECT_RATIO_{train_res}'))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset = MyDataset(
            args.data_path,
            train_res=None,
            item_processor=T2IItemProcessor(image_transform),
            cache_on_disk=args.cache_data_on_disk,
        )
        logger.info(f"Dataset for {train_res} contains {len(dataset):,} images ({args.data_path})")
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
        batch_sampler = AspectRatioBatchSampler(
            sampler,
            dataset,
            local_bsz,
            aspect_ratios=eval(f'ASPECT_RATIO_{train_res}'),
            drop_last=True,
        )
        denoising_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
        )
        real_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
        )
        data_collection[train_res] = {
            "denoising_loader_iter": iter(denoising_loader),
            "real_loader_iter": iter(real_loader),
            "global_bsz": global_bsz,
            "local_bsz": local_bsz,
            "micro_bsz": micro_bsz,
            "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
        }

    # Variables for monitoring/logging purposes:
    logger.info(f"Training for {args.max_steps:,} steps...")

    start_time = time()
    for step in range(resume_step, args.max_steps):
        model.train()
        COMPUTE_GENERATOR_GRADIENT = step % args.dfake_gen_update_ratio == 0
        high_res = random.choices(args.high_res_list, weights=args.high_res_probs)[0]
        high_res = torch.tensor(high_res, device=device)
        torch.distributed.broadcast(high_res, src=0)
        high_res = high_res.item()
        data_pack = data_collection[high_res]
        visual = step % args.visual_every == 0
        
        x_denoise, caps_denoise, text_emb_denoise = next(data_pack["denoising_loader_iter"])
        x_real, caps_real, text_emb_real = next(data_pack["real_loader_iter"])
        
        try:
            x_denoise = torch.stack([img.to(device, non_blocking=True) for img in x_denoise])
            x_real = torch.stack([img.to(device, non_blocking=True) for img in x_real])
            if rank == 0:
                print(step, x_denoise.shape, x_real.shape)
                continue
        except RuntimeError:
            print(f"RuntimeError at step {step}")
            for img in x_denoise:
                print(img.shape)
            for img in x_real:
                print(img.shape)
            continue
        bsz = len(caps_denoise)
        
        with torch.no_grad():
            x_denoise = encode_in_chunks(ae, x_denoise)
            x_real = encode_in_chunks(ae, x_real)
            
            inp_denoise = prepare(t5=t5, clip=clip, img=x_denoise, prompt=caps_denoise, proportion_empty_prompts=args.caption_dropout_prob, text_emb=text_emb_denoise)
            
            inpt_real = prepare(t5=t5, clip=clip, img=x_real, prompt=caps_real, proportion_empty_prompts=args.caption_dropout_prob, text_emb=text_emb_real)
            
        generator_loss_item = 0.0
        loss_dm_item = 0.0
        gen_cls_loss_item = 0.0
        accumulated_generator_log_dict = None
        for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
            mb_st = mb_idx * data_pack["micro_bsz"]
            mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
            last_mb = mb_ed == data_pack["local_bsz"]
            
            denoise_kwargs = dict(
                img=inp_denoise["img"][mb_st:mb_ed],
                img_ids=inp_denoise["img_ids"][mb_st:mb_ed],
                txt=inp_denoise["txt"][mb_st:mb_ed],
                txt_ids=inp_denoise["txt_ids"][mb_st:mb_ed],
                txt_mask=inp_denoise["txt_mask"][mb_st:mb_ed],
                y=inp_denoise["vec"][mb_st:mb_ed],
                img_mask=inp_denoise["img_mask"][mb_st:mb_ed],
                height=inp_denoise["height"],
                width=inp_denoise["width"],
            )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                generator_loss_dict, generator_log_dict = model(
                    model_kwargs=denoise_kwargs,
                    generator_turn=True,
                    guidance_turn=False,
                    compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                    visual=visual,
                )
            
            accumulated_generator_log_dict = accumulate_log_dict(accumulated_generator_log_dict, generator_log_dict)
            
            if COMPUTE_GENERATOR_GRADIENT:
                loss_dm = generator_loss_dict["loss_dm"].sum() / data_pack["local_bsz"]
                gen_cls_loss = generator_loss_dict["gen_cls_loss"].sum() / data_pack["local_bsz"]
                generator_loss = loss_dm * args.dm_loss_weight + gen_cls_loss * args.gen_cls_loss_weight
                generator_loss_item += generator_loss.item()
                loss_dm_item += loss_dm.item()
                gen_cls_loss_item += gen_cls_loss.item()
                
                with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                    generator_loss.backward()
        
        if COMPUTE_GENERATOR_GRADIENT:
            generator_grad_norm = model.generator.clip_grad_norm_(max_norm=args.grad_clip)
            opt_generator.step()
            opt_generator.zero_grad()
            opt_guidance.zero_grad()
        
        guidance_loss_item = 0.0
        loss_fake_item = 0.0
        guidance_cls_loss_item = 0.0
        accumulated_guidance_log_dict = None
        for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
            mb_st = mb_idx * data_pack["micro_bsz"]
            mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
            last_mb = mb_ed == data_pack["local_bsz"]
            
            denoise_kwargs["img"] = accumulated_generator_log_dict["guidance_data_dict"]["img"][mb_st:mb_ed]
            
            extra_kwargs = dict(
                img=inpt_real["img"][mb_st:mb_ed],
                img_ids=inpt_real["img_ids"][mb_st:mb_ed],
                txt=inpt_real["txt"][mb_st:mb_ed],
                txt_ids=inpt_real["txt_ids"][mb_st:mb_ed],
                txt_mask=inpt_real["txt_mask"][mb_st:mb_ed],
                y=inpt_real["vec"][mb_st:mb_ed],
                img_mask=inpt_real["img_mask"][mb_st:mb_ed],
                height=inpt_real["height"],
                width=inpt_real["width"],
            )
            
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                guidance_loss_dict, guidance_log_dict = model(
                    model_kwargs=denoise_kwargs, 
                    extra_kwargs=extra_kwargs,
                    generator_turn=False,
                    guidance_turn=True,
                    compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                    visual=visual,
                )
                
            accumulated_guidance_log_dict = accumulate_log_dict(accumulated_guidance_log_dict, guidance_log_dict)
            
            loss_fake = guidance_loss_dict["loss_fake"].sum() / data_pack["local_bsz"]
            guidance_cls_loss = guidance_loss_dict["guidance_cls_loss"].sum() / data_pack["local_bsz"]
            guidance_loss = loss_fake * args.guidance_loss_weight + guidance_cls_loss * args.guidance_cls_loss_weight
            guidance_loss_item += guidance_loss.item()
            loss_fake_item += loss_fake.item()
            guidance_cls_loss_item += guidance_cls_loss.item()
            
            with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                guidance_loss.backward()

        guidance_grad_norm = model.fake_model.clip_grad_norm_(max_norm=args.grad_clip)
        opt_guidance.step()
        opt_generator.zero_grad()
        opt_guidance.zero_grad()
        
        end_time = time()
        iter_time = end_time - start_time
        
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**accumulated_generator_log_dict, **accumulated_guidance_log_dict}
        
        generated_image_mean = log_dict["guidance_data_dict"]["img"].mean()
        generated_image_std = log_dict["guidance_data_dict"]["img"].std()
        torch.distributed.all_reduce(generated_image_mean, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(generated_image_std, op=torch.distributed.ReduceOp.AVG)
        
        if COMPUTE_GENERATOR_GRADIENT:
            dmtrain_pred_real_image_mean = log_dict['dmtrain_pred_real_image'].mean()
            dmtrain_pred_real_image_std = log_dict['dmtrain_pred_real_image'].std() 
            torch.distributed.all_reduce(dmtrain_pred_real_image_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(dmtrain_pred_real_image_std, op=torch.distributed.ReduceOp.AVG)
            
            dmtrain_pred_fake_image_mean = log_dict['dmtrain_pred_fake_image'].mean()
            dmtrain_pred_fake_image_std = log_dict['dmtrain_pred_fake_image'].std() 
            torch.distributed.all_reduce(dmtrain_pred_fake_image_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(dmtrain_pred_fake_image_std, op=torch.distributed.ReduceOp.AVG)
        
        original_image_mean = x_denoise.mean()
        original_image_std = x_denoise.std()
        torch.distributed.all_reduce(original_image_mean, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(original_image_std, op=torch.distributed.ReduceOp.AVG)
        
        if visual:
            log_dict["dmtrain_pred_real_image_decoded"] = gather_images(log_dict["dmtrain_pred_real_image_decoded"])
            log_dict["dmtrain_pred_fake_image_decoded"] = gather_images(log_dict["dmtrain_pred_fake_image_decoded"])
            log_dict["generated_image"] = gather_images(log_dict["generated_image"])
            log_dict["original_clean_image"] = gather_images(log_dict["original_clean_image"])
            log_dict["denoising_timestep"] = gather_images(log_dict["denoising_timestep"])
        
        if args.use_wandb and rank == 0:
            wandb_log_dict = {
                "iter_time": iter_time,
                "loss_fake": loss_fake_item,
                "guidance_cls_loss": guidance_cls_loss_item,
                "guidance_loss": guidance_loss_item,
                "guidance_grad_norm": guidance_grad_norm,
                "guidance_lr": opt_guidance.param_groups[0]["lr"],
                "generated_image_mean": generated_image_mean,
                "generated_image_std": generated_image_std,
            }
            
            if COMPUTE_GENERATOR_GRADIENT:
                wandb_log_dict.update({
                    "generator_lr": opt_generator.param_groups[0]["lr"],
                    "loss_dm": loss_dm_item,
                    "gen_cls_loss": gen_cls_loss_item,
                    "generator_loss": generator_loss_item,
                    "generator_grad_norm": generator_grad_norm,
                    "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean,
                    "dmtrain_pred_real_image_std": dmtrain_pred_real_image_std,
                    "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean,
                    "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std,
                })
            
            wandb_log_dict.update({
                "original_image_mean": original_image_mean,
                "original_image_std": original_image_std,
            })
            
            if visual:
                with torch.no_grad():
                    (
                        dmtrain_pred_real_image, dmtrain_pred_fake_image, 
                        dmtrain_gradient_norm
                    ) = (
                        log_dict['dmtrain_pred_real_image_decoded'], log_dict['dmtrain_pred_fake_image_decoded'], 
                        log_dict['dmtrain_gradient_norm']
                    )
                    
                    height = dmtrain_pred_real_image.shape[2]
                    width = dmtrain_pred_real_image.shape[3]
                    dmtrain_pred_real_image_grid = prepare_images_for_saving(dmtrain_pred_real_image, height=height, width=width, grid_size=args.grid_size)
                    dmtrain_pred_fake_image_grid = prepare_images_for_saving(dmtrain_pred_fake_image, height=height, width=width, grid_size=args.grid_size)

                    difference_scale_grid = draw_valued_array(
                        (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(dim=[1, 2, 3]).cpu().numpy(), 
                        output_dir=wandb_folder, grid_size=args.grid_size
                    )

                    difference = (dmtrain_pred_real_image - dmtrain_pred_fake_image)
                    difference = (difference - difference.min()) / (difference.max() - difference.min())
                    difference = (difference - 0.5) / 0.5
                    difference = prepare_images_for_saving(difference, height=height, width=width, grid_size=args.grid_size)

                    wandb_log_dict.update({
                        "dmtrain_pred_real_image": wandb.Image(dmtrain_pred_real_image_grid),
                        "dmtrain_pred_fake_image": wandb.Image(dmtrain_pred_fake_image_grid),
                        "difference": wandb.Image(difference),
                        "difference_norm_grid": wandb.Image(difference_scale_grid),
                    })

                    generated_image = log_dict['generated_image']
                    height = generated_image.shape[2]
                    width = generated_image.shape[3]
                    generated_image_grid = prepare_images_for_saving(generated_image, height=height, width=width, grid_size=args.grid_size)

                    generated_image_mean = generated_image.mean()
                    generated_image_std = generated_image.std()

                    wandb_log_dict.update({
                        "generated_image": wandb.Image(generated_image_grid),
                    })

                    origianl_clean_image = log_dict["original_clean_image"]
                    origianl_clean_image_grid = prepare_images_for_saving(origianl_clean_image, height=height, width=width, grid_size=args.grid_size)

                    denoising_timestep = log_dict["denoising_timestep"]
                    denoising_timestep_grid = draw_valued_array(denoising_timestep.cpu().numpy(), output_dir=wandb_folder, grid_size=args.grid_size)

                    wandb_log_dict.update(
                        {
                            "original_clean_image": wandb.Image(origianl_clean_image_grid),
                            "denoising_timestep": wandb.Image(denoising_timestep_grid)
                        }
                    )


                    pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                    pred_realism_on_real = log_dict["pred_realism_on_real"]

                    hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                    hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                    wandb_log_dict.update(
                        {
                            "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                            "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real),
                            # "real_image": wandb.Image(real_image_grid)
                        }
                    )


            wandb.log(wandb_log_dict, step=step)
                
                
        # Log loss values:
        metrics = data_pack["metrics"]
        if COMPUTE_GENERATOR_GRADIENT:
            metrics["loss_dm"].update(loss_dm_item)
            metrics["gen_cls_loss"].update(gen_cls_loss_item)
            metrics["generator_loss"].update(generator_loss_item)
            metrics["generator_grad_norm"].update(generator_grad_norm)
        metrics["loss_fake"].update(loss_fake_item)
        metrics["guidance_cls_loss"].update(guidance_cls_loss_item)
        metrics["guidance_loss"].update(guidance_loss_item)
        metrics["guidance_grad_norm"].update(guidance_grad_norm)
        metrics["Secs/Step"].update(iter_time)
        metrics["Imgs/Sec"].update(data_pack["global_bsz"] / iter_time)
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            logger.info(
                f"Res{high_res}: (step{step + 1:07d}) "
                + " ".join([f"{key}:{str(val)}" for key, val in metrics.items()])
            )

        start_time = time()

        # update_ema(model_ema, model)

        # Save DiT checkpoint:
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.generator.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "generator_consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
                
                consolidated_model_state_dict = model.fake_model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "fake_model_consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(consolidated_model_state_dict, os.path.join(checkpoint_path, consolidated_fn))
                    
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            # with FSDP.state_dict_type(
            #     model_ema,
            #     StateDictType.FULL_STATE_DICT,
            #     FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            # ):
            #     consolidated_ema_state_dict = model_ema.state_dict()
            #     if fs_init.get_data_parallel_rank() == 0:
            #         consolidated_ema_fn = (
            #             "consolidated_ema."
            #             f"{fs_init.get_model_parallel_rank():02d}-of-"
            #             f"{fs_init.get_model_parallel_world_size():02d}"
            #             ".pth"
            #         )
            #         torch.save(
            #             consolidated_ema_state_dict,
            #             os.path.join(checkpoint_path, consolidated_ema_fn),
            #         )
            # dist.barrier()
            # del consolidated_ema_state_dict
            # logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"generator_optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt_generator.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
                
                opt_state_fn = f"fake_model_optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt_guidance.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

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
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
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
    parser.add_argument("--generator_lr", type=float, default=1e-4, help="Learning rate of generator.")
    parser.add_argument("--guidance_lr", type=float, default=1e-4, help="Learning rate of guidance model.")
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
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument("--do_shift", default=True)
    parser.add_argument(
        "--no_shift",
        action="store_false",
        dest="do_shift",
        help="Do dynamic time shifting",
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
    parser.add_argument("--generator_loss_type", type=str, default="dmd")
    parser.add_argument("--generator_guidance_scale", type=float, default=1.0)
    parser.add_argument("--real_guidance_scale", type=float, default=1.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    parser.add_argument("--min_step_percent", type=float, default=0.02)
    parser.add_argument("--max_step_percent", type=float, default=0.98)
    parser.add_argument("--visual_every", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--guidance_loss_weight", type=float, default=1.0)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)
    parser.add_argument("--gen_cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_discriminator_heads", type=int, default=0)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    args.high_res_list = [int(res) for res in args.high_res_list.split(",")]
    args.high_res_probs = [float(prob) for prob in args.high_res_probs.split(",")]
    main(args)
