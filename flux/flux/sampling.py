import math
import random
from PIL import Image
from typing import Callable

from einops import rearrange, repeat
import torch
from torch import Tensor
import torch.nn.functional as F

from .model import Flux
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import ReduxImageEncoder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, img_cond: Tensor, prompt: str | list[str], proportion_empty_prompts: float = 0.1, proportion_empty_images: float = 0.0, is_train: bool = True, text_emb: list[dict[str, Tensor]] = None, img_embedder: ReduxImageEncoder = None, raw_img_cond: list[Image.Image] = None, is_training: bool = True) -> dict[str, Tensor]:
    if isinstance(img, torch.Tensor):
        bs, c, h, w = img.shape
        _, _, h_cond, w_cond = img_cond.shape
        down_factor = h // h_cond
        
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
        
        img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img_cond.shape[0] == 1 and bs > 1:
            img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        
        img_cond_ids = torch.zeros(h_cond // 2, w_cond // 2, 3)
        img_cond_ids[..., 0] = -1
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + (torch.arange(h_cond // 2)[:, None] * down_factor + down_factor / 2 - 0.5)
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + (torch.arange(w_cond // 2)[None, :] * down_factor + down_factor / 2 - 0.5)
        
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)
        img_mask = torch.ones(bs, img.shape[1], device=img.device, dtype=torch.int32)
        img_cond_mask = torch.ones(bs, img_cond.shape[1], device=img_cond.device, dtype=torch.int32) 
        # TODO: check this
        
    else:
        bs = len(img)
        max_len = max([i.shape[-2] * i.shape[-1] for i in img]) // 4
        max_len_cond = max([i.shape[-2] * i.shape[-1] for i in img_cond]) // 4
        # pad img to same length for batch processing
        img_mask = torch.zeros(bs, max_len, device=img[0].device, dtype=torch.int32)
        img_cond_mask = torch.zeros(bs, max_len_cond, device=img_cond[0].device, dtype=torch.int32)
        padded_img = []
        padded_img_ids = []
        padded_img_cond = []
        padded_img_cond_ids = []
        for i in range(bs):
            img_i = img[i].squeeze(0)
            img_cond_i = img_cond[i].squeeze(0)
            c, h, w = img_i.shape
            _, h_cond, w_cond = img_cond_i.shape
            down_factor = h // h_cond
            
            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            
            img_cond_ids = torch.zeros(h_cond // 2, w_cond // 2, 3)
            img_cond_ids[..., 0] = -1
            img_cond_ids[..., 1] = img_cond_ids[..., 1] + (torch.arange(h_cond // 2)[:, None] * down_factor + down_factor / 2 - 0.5)
            img_cond_ids[..., 2] = img_cond_ids[..., 2] + (torch.arange(w_cond // 2)[None, :] * down_factor + down_factor / 2 - 0.5)
            
            flat_img_ids = rearrange(img_ids, "h w c -> (h w) c")
            flat_img_cond_ids = rearrange(img_cond_ids, "h w c -> (h w) c")
            flat_img = rearrange(img_i, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            flat_img_cond = rearrange(img_cond_i, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            
            padded_img.append(F.pad(flat_img, (0, 0, 0, max_len - flat_img.shape[0])))
            padded_img_ids.append(F.pad(flat_img_ids, (0, 0, 0, max_len - flat_img_ids.shape[0])))
            padded_img_cond.append(F.pad(flat_img_cond, (0, 0, 0, max_len_cond - flat_img_cond.shape[0])))
            padded_img_cond_ids.append(F.pad(flat_img_cond_ids, (0, 0, 0, max_len_cond - flat_img_cond_ids.shape[0])))
            img_mask[i, :flat_img.shape[0]] = 1
            img_cond_mask[i, :flat_img_cond.shape[0]] = 1
        img = torch.stack(padded_img, dim=0)
        img_ids = torch.stack(padded_img_ids, dim=0)
        img_cond = torch.stack(padded_img_cond, dim=0)
        img_cond_ids = torch.stack(padded_img_cond_ids, dim=0)
        
    if isinstance(prompt, str):
        prompt = [prompt]
        
    bs = len(prompt)
    for idx in range(bs):
        if random.random() < proportion_empty_prompts:
            prompt[idx] = ""
        elif isinstance(prompt[idx], (list)):
            prompt[idx] = random.choice(prompt[idx]) if is_train else prompt[idx][0]
        if random.random() < proportion_empty_images:
            img_cond[idx].zero_()
    
    if text_emb is not None:
        txt = torch.stack([item["txt"] for item in text_emb], dim=0).to(img.device)
    else:
        txt = t5(prompt)

    if img_embedder is not None:
        with torch.no_grad():
            global_img_cond = [img_embedder(raw_img_cond[i]) for i in range(bs)]
        global_img_cond = torch.cat(global_img_cond, dim=0).to(img.device)
        
        if not is_training:
            txt = torch.cat([txt, global_img_cond], dim=1) # prompt + image
            # txt = txt
            # txt = global_img_cond
        elif random.random() < 0.3:
            txt = torch.cat([txt, global_img_cond], dim=1) # prompt + image
        elif random.random() < 0.6:
            txt = global_img_cond # image only
        else:
            txt = txt # prompt only

    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    txt_mask = torch.ones(bs, txt.shape[1], device=txt.device, dtype=torch.int32)
        
    if text_emb is not None:
        vec = torch.stack([item["vec"] for item in text_emb], dim=0).to(img.device)
    else:
        vec = clip(prompt)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "img_cond": img_cond,
        "img_cond_ids": img_cond_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
        "img_mask": img_mask.to(img.device),
        "img_cond_mask": img_cond_mask.to(img.device),
        "txt_mask": txt_mask.to(txt.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
