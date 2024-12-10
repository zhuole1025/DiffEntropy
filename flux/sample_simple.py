import argparse
import functools
import json
import os
import random
import time
from tqdm import tqdm

from einops import rearrange, repeat
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchdiffeq import odeint

from flux.controlnet import ControlNetFlux
from flux.model import Flux, FluxParams
from flux.sampling import prepare
from flux.util import configs, load_clip, load_t5, load_flow_model
from flux.modules.image_embedders import ReduxImageEncoder
from transport.utils import get_lin_function, time_shift
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop, apply_histogram_matching, apply_statistical_color_matching


def invert_transform(x):
    x = x * 0.5 + 0.5
    x = torch.clamp(x, 0, 1)
    x = transforms.ToPILImage()(x)
    return x


def none_or_str(value):
    if value == "None":
        return None
    return value


def sample_ode(
    x,
    model,
    model_kwargs,
    controlnet=None,
    controlnet_kwargs=None,
    sampling_method="euler",
    num_steps=50,
    do_shift=True,
    time_shifting_factor=None,
    denoising_strength=1.0
):

    device = x.device
    if controlnet is not None:
        x_cond = controlnet_kwargs.pop("controlnet_cond")
    def _fn(t, x):
        t = torch.ones(x.size(0)).to(device) * t
        if controlnet is not None:
            # t_cond = torch.ones(x_cond.size(0)).to(device) * t * 0.5 + 0.5
            t_cond = torch.ones(x_cond.size(0)).to(device) * 0.5
            noise = torch.randn_like(x_cond)
            xt_cond = x_cond * t_cond.view(-1, 1, 1) + noise * (1 - t_cond).view(-1, 1, 1)
            controlnet_out_dict = controlnet(xt_cond, timesteps=1 - t_cond, bb_timesteps=1 - t, **controlnet_kwargs)
            model_kwargs["double_controls"] = controlnet_out_dict["double_block_feats"]
            model_kwargs["single_controls"] = controlnet_out_dict["single_block_feats"]
        model_output = -model(x, 1 - t, **model_kwargs)
        return model_output
    
    t = torch.linspace(1 - denoising_strength, 1, num_steps).to(device)
    if do_shift:
        mu = get_lin_function(y1=0.5, y2=1.15)(x.shape[1])
        t = time_shift(mu, 1.0, t)
    samples = odeint(_fn, x, t, method=sampling_method)

    return samples
    

def generate_samples(
    prompt,
    init_img,
    model,
    controlnet,
    ae,
    t5,
    clip,
    img_embedder=None,
    height=1024,
    width=1024,
    downsample_factor=1,
    double_gate=1.0,
    single_gate=1.0,
    guidance_scale=4.0,
    guidance_scale_controlnet=1.0,
    denoising_strength=1.0,
    solver="euler",
    num_sampling_steps=50,
    do_shift=True,
    time_shifting_factor=1.0,
    device="cuda",
    dtype=torch.bfloat16,
):
    """Generate samples using the model pipeline.
    
    Args:
        prompt (str): Input prompt
        x_cond (PIL.Image.Image): Conditioning image (low resolution)
        model (nn.Module): Main model
        controlnet (nn.Module): ControlNet model
        ae (AutoencoderKL): VAE model
        t5 (nn.Module): T5 text encoder
        clip (nn.Module): CLIP model
        img_embedder (nn.Module, optional): Image embedder model, e.g. ReduxImageEncoder
        height (int): Height of the generated image
        width (int): Width of the generated image
        downsample_factor (int): Downsample factor for the conditioning image
        double_gate (float): Double gate value
        single_gate (float): Single gate value
        guidance_scale (float): Guidance scale for diffusion model
        guidance_scale_controlnet (float): Guidance scale for controlnet
        denoising_strength (float): Start denoising strength, 1.0 means start from pure noise
        solver (str): ODE solver method
        num_sampling_steps (int): Number of sampling steps
        do_shift (bool): Whether to apply time shifting
        time_shifting_factor (float): Time shifting factor
    
    Returns:
        tuple: Generated samples and conditioning images
    """
    if downsample_factor > 1:
        init_h, init_w = init_img.size
        down_h, down_w = init_h // downsample_factor, init_w // downsample_factor
        downsample_transform = transforms.Compose([
            transforms.Resize((down_h, down_w), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Resize((init_h, init_w), interpolation=transforms.InterpolationMode.LANCZOS)
        ])
        init_img = downsample_transform(init_img)
        
    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
        transforms.Lambda(functools.partial(var_center_crop, crop_size_list=[(height, width)], random_top_k=1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                            inplace=True),
    ])
    x_cond = image_transform(init_img).to(device)
    
    if img_embedder is not None:
        raw_x_cond = invert_transform(x_cond)
        raw_x_cond = [raw_x_cond]
    else:
        raw_x_cond = None
        
    with torch.no_grad():
        x_cond = (ae.encode(x_cond[None].to(ae.dtype)).latent_dist.sample()[0] - ae.config.shift_factor) * ae.config.scaling_factor
    x_cond = x_cond[None]
    
    noise = torch.randn([1, 16, height // 8, width // 8], device=device, dtype=dtype)
    start_time = 1 - denoising_strength
    if do_shift:
        mu = get_lin_function(y1=0.5, y2=1.15)((height // 16) * (width // 16))
        t = time_shift(mu, 1.0, start_time)
    x = noise * (1 - t) + x_cond * t
    
    with torch.no_grad():
        inp = prepare(
            t5=t5,
            clip=clip,
            img=x,
            img_cond=x_cond,
            prompt=[prompt],
            proportion_empty_prompts=0.0,
            proportion_empty_images=0.0,
            raw_img_cond=raw_x_cond,
            img_embedder=img_embedder,
            is_training=False
        )
    
    model_kwargs = dict(
        txt=inp["txt"],
        txt_ids=inp["txt_ids"],
        txt_mask=inp["txt_mask"],
        y=inp["vec"],
        img_ids=inp["img_ids"],
        img_mask=inp["img_mask"],
        guidance=torch.full((x.shape[0],), guidance_scale, device=x.device, dtype=x.dtype),
        double_gate=double_gate,
        single_gate=single_gate
    )
    
    controlnet_kwargs = dict(
        img_ids=inp["img_ids"],
        controlnet_cond=inp["img_cond"],
        txt=inp["txt"],
        txt_ids=inp["txt_ids"],
        y=inp["vec"],
        txt_mask=inp["txt_mask"],
        img_mask=inp["img_mask"],
        guidance=torch.full((x.shape[0],), guidance_scale_controlnet, device=x.device, dtype=x.dtype),
    )
    
    with torch.autocast(device, dtype):
        samples = sample_ode(
            x=inp["img"],
            model=model.forward_with_cfg,
            model_kwargs=model_kwargs,
            controlnet=controlnet,
            controlnet_kwargs=controlnet_kwargs,
            sampling_method=solver,
            num_steps=num_sampling_steps,
            do_shift=do_shift,
            time_shifting_factor=time_shifting_factor,
            denoising_strength=denoising_strength,
        )[-1]
    
    # Process samples
    samples = samples[:1]
    samples = rearrange(samples, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=height//16, w=width//16)
    samples = ae.decode(samples / ae.config.scaling_factor + ae.config.shift_factor)[0]
    samples = (samples + 1.0) / 2.0
    samples.clamp_(0.0, 1.0)
    
    # Process conditioning images
    x_cond = inp["img_cond"][:1]
    x_cond = rearrange(x_cond, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=height//16, w=width//16)
    x_cond = ae.decode(x_cond / ae.config.scaling_factor + ae.config.shift_factor)[0]
    x_cond = (x_cond + 1.0) / 2.0
    x_cond.clamp_(0.0, 1.0)
    
    return samples, x_cond


def main(args, rank=0):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    torch.cuda.set_device(rank)
    device_str = f"cuda:{rank}"

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    print("Init controlnet")
    params = configs["flux-dev"].params
    with torch.device(device_str):
        controlnet = ControlNetFlux(
            params, 
            double_depth=train_args.double_depth, 
            single_depth=train_args.single_depth, 
            backbone_depth=train_args.backbone_depth, 
            backbone_depth_single=train_args.backbone_depth_single,
            compute_loss=False,
        ).to(dtype)
    controlnet.eval()
    
    print("Init model")
    params.attn_token_select = False
    params.mlp_token_select = False
    params.zero_init = False
    with torch.device(device_str):
        model = Flux(params).to(dtype)
    model.eval()
        
    print("Init vae")
    ae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device_str)
    ae.requires_grad_(False)
    
    print("Init text encoder")
    t5 = load_t5(device_str, max_length=512)
    clip = load_clip(device_str)
    
    if args.img_embedder_path is not None:
        img_embedder = ReduxImageEncoder(device=device_str, redux_path=args.img_embedder_path)
        img_embedder.requires_grad_(False)
        print(f"Image embedder loaded")
    else:
        img_embedder = None
        
    ckpt = torch.load(
        os.path.join(
            args.ckpt,
            f"consolidated.00-of-01.pth",
        )
    )
    model.load_state_dict(ckpt, strict=True)
        
    ckpt = torch.load(
        os.path.join(
            args.ckpt,
            f"consolidated_controlnet.00-of-01.pth",
        )
    )
    controlnet.load_state_dict(ckpt, strict=True)
        
    sample_folder_dir = args.image_save_path
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    prompt = args.prompt
    image = Image.open(args.img_path)
    image = image.convert("RGB")
                                
    if int(args.seed) != 0:
        torch.random.manual_seed(int(args.seed))

    samples, x_conds = generate_samples(
        init_img=image,
        prompt=prompt,
        model=model,
        controlnet=controlnet,
        ae=ae,
        t5=t5,
        clip=clip,
        img_embedder=img_embedder,
        height=args.height,
        width=args.width,
        denoising_strength=args.denoising_strength,
        downsample_factor=args.downsample_factor,
        double_gate=args.double_gate,
        single_gate=args.single_gate,
        solver=args.solver,
        num_sampling_steps=args.num_sampling_steps,
        do_shift=args.do_shift,
        time_shifting_factor=args.time_shifting_factor,
    )
    sample, x_cond = samples[0], x_conds[0]
    img = to_pil_image(sample.float())
    # img = apply_statistical_color_matching(img, image)
    # convert img from numpy to PIL Image
    # img = Image.fromarray(img.astype('uint8'), 'RGB')
    save_path = f"{args.image_save_path}/{args.solver}_{args.num_sampling_steps}_{args.denoising_strength}.jpg"
    img.save(save_path, format='JPEG', quality=95)         
    low_img = to_pil_image(x_cond.float())
    low_save_path = f"{args.image_save_path}/{args.solver}_{args.num_sampling_steps}_{args.denoising_strength}_low.jpg"
    low_img.save(low_save_path, format='JPEG', quality=95)
                        
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument("--do_shift", default=True)
    parser.add_argument("--double_gate", type=float, default=1.0)
    parser.add_argument("--single_gate", type=float, default=1.0)
    parser.add_argument("--img_embedder_path", type=str, default=None)
    args = parser.parse_known_args()[0]
    
    main(args)