from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules.layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding, ControlNetGate, DiscriminatorHead
from flux.modules.lora import LinearLora, replace_linear_with_lora
from transport.utils import mean_flat, time_shift, get_lin_function, expand_t_like_x

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

    def __init__(self, params: FluxParams, num_discriminator_heads: int = 0):
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
        
        self.discriminator = None
        if num_discriminator_heads > 0:
            self.discriminator = nn.ModuleList(
                [
                    DiscriminatorHead(self.hidden_size)
                    for _ in range(num_discriminator_heads)
                ]
            )
        
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
        controls: tuple = None,
        classify_mode: bool = False,
        height: int = None,
        width: int = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        double_blocks_out = []
        for idx, block in enumerate(self.double_blocks):
            img, txt, img_cond, sub_token_select, token_logits = block(img=img, txt=txt, vec=vec, pe=pe, img_mask=img_mask, txt_mask=txt_mask, cond=img_cond, cond_mask=img_cond_mask)
            # controlnet residual
            if controls is not None:
                img = img + controls[idx % len(controls)]
            if classify_mode:
                double_blocks_out.append(img)
        
        img = torch.cat((txt, img), 1)
        attn_mask = torch.cat((txt_mask, img_mask), 1)
        # single_blocks_out = []
        for block in self.single_blocks:
            img, sub_token_select, token_logits = block(img, vec=vec, pe=pe, attn_mask=attn_mask)
            # if classify_mode:
                # single_blocks_out.append(img[:, txt.shape[1] :, ...])
        
        if classify_mode:
            outputs = []
            all_blocks_out = double_blocks_out
            for feature, head in zip(all_blocks_out, self.discriminator):
                feature = feature.permute(0, 2, 1).reshape(feature.shape[0], -1, height, width)
                outputs.append(head(feature))
            outputs = torch.cat(outputs, dim=1) # (N, num_heads, H * W)
            return outputs
        
        img = img[:, txt.shape[1] :, ...]
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return {
            "output": img,
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
        controls: tuple = None,
        txt_cfg_scale: float = 1.0,
        img_cfg_scale: float = 1.0,
    ) -> Tensor:
        half = img[: len(img) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(
            img=combined, 
            timesteps=timesteps, 
            img_ids=img_ids, 
            txt=txt, 
            txt_ids=txt_ids, 
            y=y, 
            guidance=guidance, 
            txt_mask=txt_mask, 
            img_mask=img_mask, 
        )["output"]

        cond_v, uncond_v = torch.split(model_out, len(model_out) // 2, dim=0)
        cond_v = uncond_v + txt_cfg_scale * (cond_v - uncond_v)
        eps = torch.cat([cond_v, uncond_v], dim=0)
        return eps

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
        modules = list(self.double_blocks) + list(self.single_blocks) + [self.final_layer]
        if self.discriminator is not None:
            modules += list(self.discriminator)
        return modules

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        modules = list(self.double_blocks) + list(self.single_blocks) + [self.final_layer]
        if self.discriminator is not None:
            modules += list(self.discriminator)
        return modules


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )
    
    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
                

class FluxUnifiedWrapper(nn.Module):
    def __init__(self, params: FluxParams, num_discriminator_heads: int = 0, dtype: torch.dtype = torch.bfloat16, snr_type: str = "uniform", do_shift: bool = False, grid_size: int = 1):
        super().__init__()
        
        self.guidance = FluxLoraWrapper(params=params, num_discriminator_heads=num_discriminator_heads).to(dtype)
        torch.cuda.empty_cache()
        self.generator = Flux(params=params, num_discriminator_heads=0).to(dtype)
        torch.cuda.empty_cache()
        
        self.snr_type = snr_type
        self.do_shift = do_shift
        self.grid_size = grid_size
    
    def load_state_dict(self, guidance_state_dict: dict, generator_state_dict: dict):
        self.guidance.load_state_dict(guidance_state_dict, strict=False, assign=True)
        self.generator.load_state_dict(generator_state_dict, strict=False, assign=True)
    
    def parameter_count(self) -> tuple[int, int]:
        return self.guidance.parameter_count(), self.generator.parameter_count()
    
    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return self.guidance.get_fsdp_wrap_module_list() + self.generator.get_fsdp_wrap_module_list()

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return self.guidance.get_checkpointing_wrap_module_list() + self.generator.get_checkpointing_wrap_module_list()
    
    def sample(self, x1, x0=None, snr_type=None):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        if x0 is not None:
            x0 = x0.to(x1.device)
        elif isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = torch.randn_like(x1)
        t0, t1 = 0.0, 1.0

        if snr_type is None:
            snr_type = self.snr_type
            
        if snr_type == "controlnet":
            t = torch.rand((len(x1),))
            alpha = 0.3
            t = torch.where(t < alpha, 0.0, 
                torch.where(t >= 1-alpha, 1.0,
                    torch.rand_like(t)))
        elif snr_type.startswith("uniform"):
            if "_" in snr_type:
                _, t0, t1 = snr_type.split("_")
                t0, t1 = float(t0), float(t1)
            t = torch.rand((len(x1),)) * (t1 - t0) + t0
        elif snr_type == "lognorm":
            u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
        else:
            raise NotImplementedError("Not implemented snr_type %s" % snr_type)

        if self.do_shift:
            base_shift: float = 0.5
            max_shift: float = 1.15
            mu = get_lin_function(y1=base_shift, y2=max_shift)(x1.shape[1])
            t = time_shift(mu, 1.0, t)

        t = t.to(x1[0])
        return t, x0, x1
    
    def compute_distribution_matching_loss(self, x1, x0=None, model_kwargs=None):
        original_x1 = x1 
        with torch.no_grad():
            t, x0, x1 = self.sample(x1, x0)
            t_expanded = expand_t_like_x(t, x1)
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            ut = x1 - x0
            
            # fake score
            self.guidance.set_lora_scale(1.0)
            out_dict = self.guidance(xt, timesteps=1 - t, **model_kwargs)
            model_output = -out_dict["output"]  
            pred_fake_image = xt + model_output * (1 - t_expanded)

            # real score
            self.guidance.set_lora_scale(0.0)
            out_dict = self.guidance(xt, timesteps=1 - t, **model_kwargs)
            model_output = -out_dict["output"]  
            pred_real_image = xt + model_output * (1 - t_expanded)

            p_real = (x1 - pred_real_image)
            p_fake = (x1 - pred_fake_image)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2], keepdim=True) 
            grad = torch.nan_to_num(grad)
            
        loss = 0.5 * F.mse_loss(original_x1.float(), (original_x1 - grad).detach().float(), reduction="mean")
        
        loss_dict = {
            "loss_dm": loss 
        }
        
        dm_log_dict = {
            "dmtrain_noisy_latents": xt.detach().float(),
            "dmtrain_pred_real_image": pred_real_image.detach().float(),
            "dmtrain_pred_fake_image": pred_fake_image.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict
    
    def compute_cls_logits(self, x1, x0=None, model_kwargs=None):
        t, x0, x1 = self.sample(x1, x0)
        t_expanded = expand_t_like_x(t, x1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0
        self.guidance.set_lora_scale(1.0)
        logits = self.guidance(xt, timesteps=1 - t, classify_mode=True, **model_kwargs)
        return logits

    def compute_generator_clean_cls_loss(self, 
        x1, x0=None, model_kwargs=None
    ):
        logits = self.compute_cls_logits(x1, x0, model_kwargs)
        loss_dict = {}
        loss_dict["gen_cls_loss"] = mean_flat(F.softplus(-logits))
        return loss_dict 
    
    def compute_guidance_clean_cls_loss(self, real_image, fake_image, x0=None, model_kwargs=None):
        pred_realism_on_real = self.compute_cls_logits(real_image.detach(), x0, model_kwargs)
        pred_realis_on_fake = self.compute_cls_logits(fake_image.detach(), x0, model_kwargs)
        
        log_dict = {
            "pred_realism_on_real": mean_flat(torch.sigmoid(pred_realism_on_real).detach()),
            "pred_realism_on_fake": mean_flat(torch.sigmoid(pred_realism_on_fake).detach())
        }

        classification_loss = mean_flat(F.softplus(pred_realism_on_fake)) + mean_flat(F.softplus(-pred_realism_on_real))
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 
        
    
    def compute_loss_fake(self, x1, x0=None, model_kwargs=None):
        t, x0, x1 = self.sample(x1.detach(), x0)
        t_expanded = expand_t_like_x(t, x1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0
            
        self.guidance.set_lora_scale(0.0)
        model_kwargs['guidance'] = torch.ones_like(model_kwargs['guidance'], device=x1.device, dtype=x1.dtype)
        out_dict = self.guidance(xt, timesteps=1 - t, **model_kwargs)
        model_output = -out_dict["output"]
        pred_fake_image = xt + model_output * (1 - t_expanded)
        loss_fake = mean_flat(((model_output - ut) ** 2))
        loss_dict = {
            "loss_fake_mean": loss_fake,
        }
        fake_log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_latents.detach().float(),
            "faketrain_x0_pred": fake_x0_pred.detach().float()
        }
        
        return loss_dict, fake_log_dict
    
    def forward(self, model_kwargs: dict, generator_turn: bool = False, guidance_turn: bool = False, compute_generator_gradient: bool = True, visual: bool = False):
        loss_dict = {}
        log_dict = {}
        
        # generator_turn:
        if generator_turn:
            height, width = model_kwargs['height'], model_kwargs['width']
            
            # prepare noisy data using forward diffusion
            x1 = model_kwargs.pop('img')
            x0 = model_kwargs.pop('noise')
            t, x0, x1 = self.sample(x1, x0)
            t_expanded = expand_t_like_x(t, x1)
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            ut = x1 - x0
            
            if compute_generator_gradient:
                out_dict = self.generator(xt, timesteps=1 - t, **model_kwargs)
                model_output = -out_dict["output"]  
            else:
                with torch.no_grad():
                    out_dict = self.generator(xt, timesteps=1 - t, **model_kwargs)
                    model_output = -out_dict["output"]  
                
            generated_image = xt + model_output * (1 - t_expanded)
            if compute_generator_gradient:
                self.guidance.requires_grad_(False)
                dm_dict, dm_log_dict = self.compute_distribution_matching_loss(generated_image, x0, model_kwargs)
                loss_dict.update(dm_dict)
                log_dict.update(dm_log_dict)
                
                cls_loss_dict = self.compute_generator_clean_cls_loss(generated_image, x0, model_kwargs)
                loss_dict.update(cls_loss_dict)
                self.guidance.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}
        
            # if visual:
            #     decode_key = [
            #         "dmtrain_pred_real_image", "dmtrain_pred_fake_image"
            #     ]

            #     with torch.no_grad():
            #         if compute_generator_gradient:
            #             for key in decode_key:
            #                 latents = log_dict[key].detach()[:self.num_visuals]
            #                 latents = self.unpack(latents, height, width)
            #                 log_dict[key+"_decoded"] = self.decode_image(latents) 
                    
            #         latents = generated_image[:self.num_visuals].detach()
            #         latents = self.unpack(latents, height, width)
            #         log_dict["generated_image"] = self.decode_image(latents)

            #         latents = x1[:self.num_visuals].detach()
            #         latents = self.unpack(latents, height, width)
            #         log_dict["original_clean_image"] = self.decode_image(latents)

            # log_dict["guidance_data_dict"] = {
            #     "image": generated_image.detach(),
            #     # "text_embedding": text_embedding.detach(),
            #     # "pooled_text_embedding": pooled_text_embedding.detach(),
            #     # "uncond_embedding": uncond_embedding.detach(),
            #     # "real_train_dict": real_train_dict,
            #     # "unet_added_conditions": unet_added_conditions,
            #     # "uncond_unet_added_conditions": uncond_unet_added_conditions
            # }

            # log_dict['denoising_timestep'] = t

        # guidance turn
        if guidance_turn:
            generated_image = model_kwargs.pop('img')
            x0 = model_kwargs.pop('noise')
            fake_dict, fake_log_dict = self.compute_loss_fake(generated_image, x0, model_kwargs)
            
            loss_dict = fake_dict 
            log_dict = fake_log_dict

            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dictreal_image.detach(),
                fake_image=generated_image
            )
            
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)
            
        return loss_dict, log_dict
