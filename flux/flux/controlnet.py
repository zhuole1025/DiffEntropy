from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding, zero_module)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class ControlNetFlux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, double_depth=2, single_depth=2, backbone_depth=2, backbone_depth_single=0, compute_loss=False):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(double_depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(single_depth)
            ]
        )
        # add ControlNet blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(backbone_depth):
            controlnet_block = nn.Linear(self.hidden_size, self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
        self.single_controlnet_blocks = nn.ModuleList([])
        for _ in range(backbone_depth_single):
            controlnet_block = nn.Linear(self.hidden_size, self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.single_controlnet_blocks.append(controlnet_block)
        self.cond_img_in = zero_module(nn.Linear(self.in_channels, self.hidden_size, bias=True))
        
        self.compute_loss = compute_loss
        if self.compute_loss:
            self.decoder = LastLayer(self.hidden_size, 1, self.out_channels)
        
    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        bb_timesteps: Tensor,
        y: Tensor,
        controlnet_cond: Tensor = None,
        guidance: Tensor = None,
        txt_mask: Tensor = None,
        img_mask: Tensor = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        # running on sequences img
        img = self.img_in(img)
        if controlnet_cond is not None:
            controlnet_cond = self.cond_img_in(controlnet_cond)
            img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(bb_timesteps, 256))
        if timesteps is not None:
            vec = vec + self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        double_block_res_samples = []
        for block in self.double_blocks:
            img, txt, _, _, _ = block(img=img, txt=txt, vec=vec, pe=pe, img_mask=img_mask, txt_mask=txt_mask)
            double_block_res_samples.append(img)
        
        img = torch.cat((txt, img), 1)
        attn_mask = torch.cat((txt_mask, img_mask), 1)
        single_block_res_samples = []
        for block in self.single_blocks:
            img, _, _ = block(img, vec=vec, pe=pe, attn_mask=attn_mask)
            single_block_res_samples.append(img)
        
        out_double_block_feats = ()
        for idx, controlnet_block in enumerate(self.controlnet_blocks):
            block_res_sample = double_block_res_samples[idx % len(double_block_res_samples)]
            block_res_sample = controlnet_block(block_res_sample)
            out_double_block_feats = out_double_block_feats + (block_res_sample,)
        
        out_single_block_feats = ()
        for idx, controlnet_block in enumerate(self.single_controlnet_blocks):
            block_res_sample = single_block_res_samples[idx % len(single_block_res_samples)]
            block_res_sample = controlnet_block(block_res_sample)
            out_single_block_feats = out_single_block_feats + (block_res_sample,)
            
        out_dict = {
            "double_block_feats": out_double_block_feats,
            "single_block_feats": out_single_block_feats
        }
        
        if self.compute_loss and self.training:
            img = img[:, txt.shape[1] :, ...]
            img = self.decoder(img, vec)
            out_dict["output"] = img
            
        return out_dict
    
    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.double_blocks) + list(self.single_blocks) + list(self.controlnet_blocks) + list(self.single_controlnet_blocks)

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.double_blocks) + list(self.single_blocks) + list(self.controlnet_blocks) + list(self.single_controlnet_blocks)
