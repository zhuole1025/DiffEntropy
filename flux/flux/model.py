from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn

from .modules.layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding, ControlNetGate


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
    attn_token_select: bool
    mlp_token_select: bool
    zero_init: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
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
                    attn_token_select=params.attn_token_select,
                    mlp_token_select=params.mlp_token_select,
                    zero_init=params.zero_init,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, attn_token_select=params.attn_token_select, mlp_token_select=params.mlp_token_select)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        
        self.learnable_gate = params.learnable_gate
        if self.learnable_gate:
            self.double_controlnet_gates = nn.ModuleList(
                [
                    ControlNetGate(self.hidden_size)
                    for _ in range(params.depth)
                ]   
            )
            self.single_controlnet_gates = nn.ModuleList(
                [
                    ControlNetGate(self.hidden_size)
                    for _ in range(params.depth_single_blocks)
                ]   
            )
        
        # self.img_cond_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # nn.init.zeros_(self.img_cond_in.weight)
        # nn.init.zeros_(self.img_cond_in.bias)

    def forward(
        self,
        img: Tensor,
        timesteps: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        txt_mask: Tensor = None,
        img_cond: Tensor = None,
        img_cond_ids: Tensor = None,
        img_mask: Tensor = None,
        img_cond_mask: Tensor = None,
        double_controls: list = None,
        single_controls: list = None,
        double_gate: float = 1.0,
        single_gate: float = 1.0,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        # running on sequences img
        img = self.img_in(img)
        # img_cond = self.img_cond_in(img_cond)
        # img = img + img_cond
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        token_select_list = []
        token_logits_list = []
        gate_logits_list = []
        for idx, block in enumerate(self.double_blocks):
            img, txt, img_cond, sub_token_select, token_logits = block(img=img, txt=txt, vec=vec, pe=pe, img_mask=img_mask, txt_mask=txt_mask, cond=img_cond, cond_mask=img_cond_mask)
            if (sub_token_select is not None) and (token_logits is not None):
                token_select_list.append(sub_token_select)
                token_logits_list.append(token_logits)
            # controlnet residual
            if double_controls is not None:
                # gate = torch.tanh(self.controlnet_gates[idx](torch.cat((img, img + controls[idx % len(controls)]), dim=-1), vec))
                # print(timesteps[0].item(), idx, gate.abs().mean().item(), gate.abs().var().item(), (controls[idx % len(controls)] * gate).mean().item(), img.mean().item())
                if self.learnable_gate:
                    gate = torch.tanh(self.double_controlnet_gates[idx](img, double_controls[idx % len(double_controls)], vec))
                    img = img + double_controls[idx % len(double_controls)] * gate
                else:
                    img = img + double_controls[idx % len(double_controls)] * double_gate
                # gate_logits_list.append(gate)
        
        # pe = pe[:, :, img_cond_ids.shape[1]:, ...]
        img = torch.cat((txt, img), 1)
        attn_mask = torch.cat((txt_mask, img_mask), 1)
        for idx, block in enumerate(self.single_blocks):
            img, sub_token_select, token_logits = block(img, vec=vec, pe=pe, attn_mask=attn_mask)
            if (sub_token_select is not None) and (token_logits is not None):
                token_select_list.append(sub_token_select)
                token_logits_list.append(token_logits)
            
            if single_controls is not None:
                if self.learnable_gate:
                    gate = torch.tanh(self.single_controlnet_gates[idx](img, single_controls[idx % len(single_controls)], vec))
                    img = img + single_controls[idx % len(single_controls)] * gate
                else:
                    img = img + single_controls[idx % len(single_controls)] * single_gate
                
        # if img_cond is not None:
            # img = img[:, img_cond.shape[1] :, ...]
        img = img[:, txt.shape[1] :, ...]
        
        # gate_logits = torch.stack(gate_logits_list, dim=1) if len(gate_logits_list) > 0 else None
        # token_select = torch.stack(token_select_list, dim=1) if len(token_select_list) > 0 else None
        # token_logits = torch.stack(token_logits_list, dim=1) if len(token_logits_list) > 0 else None

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return {
            "output": img,
            # "gate_logits": gate_logits,
            # "token_select": token_select,
            # "token_logits": token_logits
        }

    def forward_with_cfg(
        self,
        img: Tensor,
        timesteps: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        img_cond: Tensor = None,
        img_cond_ids: Tensor = None,
        img_mask: Tensor = None,
        img_cond_mask: Tensor = None,
        double_controls: list = None,
        single_controls: list = None,
        txt_cfg_scale: float = 1.0,
        img_cfg_scale: float = 1.0,
        double_gate: float = 1.0,
        single_gate: float = 1.0,
    ) -> Tensor:
        # half = img[: len(img) // 3]
        # combined = torch.cat([half, half, half], dim=0)
        model_out = self.forward(
            img=img, 
            timesteps=timesteps, 
            img_ids=img_ids, 
            img_cond=img_cond, 
            img_cond_ids=img_cond_ids, 
            txt=txt, 
            txt_ids=txt_ids, 
            y=y, 
            guidance=guidance, 
            txt_mask=txt_mask, 
            img_mask=img_mask, 
            img_cond_mask=img_cond_mask, 
            double_controls=double_controls,
            single_controls=single_controls,
            double_gate=double_gate,
            single_gate=single_gate
        )["output"]

        # cond_v, txt_cond_v, uncond_v = torch.split(model_out, len(model_out) // 3, dim=0)
        # cond_v = txt_cond_v + txt_cfg_scale * (cond_v - txt_cond_v)
        # eps = torch.cat([cond_v, txt_cond_v, uncond_v], dim=0)
        return model_out

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.double_blocks) + list(self.single_blocks) + [self.final_layer]

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.double_blocks) + list(self.single_blocks) + [self.final_layer]
