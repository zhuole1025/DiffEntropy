import os
import torch

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

# from diffusers import StableDiffusion3Pipeline
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from models.transformer import SD3Transformer2DModelTokenMerge
from pipeline import StableDiffusion3Pipeline
# fix random seed
torch.manual_seed(25)

torch_dtype = torch.float16
h = 1280
w = 1280
guidance_scale = 7.0
prompt = "a photo of a cat holding a sign that says hello world"
ckpt_path = "/home/pgao/zl/zl/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/b1148b4028b9ec56ebd36444c193d56aeff7ab56"

transformer = SD3Transformer2DModelTokenMerge.from_pretrained(ckpt_path, subfolder="transformer")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae")
text_encoder = CLIPTextModelWithProjection.from_pretrained(ckpt_path, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(ckpt_path, subfolder="text_encoder_2")
tokenizer_2 = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer_2")
text_encoder_3 = T5EncoderModel.from_pretrained(ckpt_path, subfolder="text_encoder_3")
tokenizer_3 = T5TokenizerFast.from_pretrained(ckpt_path, subfolder="tokenizer_3")

transformer.to(torch_dtype)
vae.to(torch.float32)
text_encoder.to(torch_dtype)
text_encoder_2.to(torch_dtype)
text_encoder_3.to(torch_dtype)

# pipe = StableDiffusion3Pipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
pipe = StableDiffusion3Pipeline(
    transformer=transformer,
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    text_encoder_3=text_encoder_3,
    tokenizer_3=tokenizer_3,
)
pipe.to("cuda")

with {
    "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
    "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
}["fp16"]:
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=30,
        height=h,
        width=w,
        guidance_scale=guidance_scale,
    ).images[0]

os.makedirs("samples", exist_ok=True)
image.save(f'samples/{h}x{w}_gs{guidance_scale}_{prompt.replace(" ", "_")}.png')