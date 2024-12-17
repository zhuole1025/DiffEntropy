import enum
import math
import contextlib
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from . import path
from .integrators import ode, sde
from .utils import mean_flat, time_shift, get_lin_function, expand_t_like_x


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, model_type=ModelType.VELOCITY, path_type=PathType.LINEAR, train_eps=0, sample_eps=0, snr_type="uniform", do_shift=True, vae=None, grid_size=1):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.do_shift = do_shift
        self.vae = vae
        self.num_visuals = grid_size * grid_size
        
    def decode_image(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents + self.vae.config.shift_factor
        image = self.vae.decode(latents).sample.float()
        return image 
    
    def unpack(self, latents, height, width):
        latents = rearrange(latents, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', h=height, w=width, ph=2, pw=2)
        return latents
    
    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = torch.tensor(z.size())
        N = torch.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - torch.sum(x**2) / 2.0
        return torch.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

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
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

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

    def compute_distribution_matching_loss(self, guidance_model, x1, x0=None, model_kwargs=None):
        original_x1 = x1 
        with torch.no_grad():
            t, x0, x1 = self.sample(x1, x0)
            t_expanded = expand_t_like_x(t, x1)
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            ut = x1 - x0
            
            # fake score
            guidance_model.set_lora_scale(1.0)
            out_dict = guidance_model(xt, timesteps=1 - t, **model_kwargs)
            model_output = -out_dict["output"]  
            pred_fake_image = xt + model_output * (1 - t_expanded)

            # real score
            guidance_model.set_lora_scale(0.0)
            out_dict = guidance_model(xt, timesteps=1 - t, **model_kwargs)
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
    
    def compute_cls_logits(self, guidance_model, x1, x0=None, model_kwargs=None):
        t, x0, x1 = self.sample(x1, x0)
        t_expanded = expand_t_like_x(t, x1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0
        guidance_model.set_lora_scale(1.0)
        logits = guidance_model(xt, timesteps=1 - t, classify_mode=True, **model_kwargs)
        return logits

    def compute_generator_clean_cls_loss(self, 
        guidance_model, x1, x0=None, model_kwargs=None
    ):
        logits = self.compute_cls_logits(guidance_model, x1, x0, model_kwargs)
        loss_dict = {}
        loss_dict["gen_cls_loss"] = mean_flat(F.softplus(-logits))
        return loss_dict 
    
    def compute_guidance_clean_cls_loss(self, guidance_model, real_image, fake_image, x0=None, model_kwargs=None):
        pred_realism_on_real = self.compute_cls_logits(guidance_model, real_image.detach(), x0, model_kwargs)
        pred_realis_on_fake = self.compute_cls_logits(guidance_model, fake_image.detach(), x0, model_kwargs)
        
        log_dict = {
            "pred_realism_on_real": mean_flat(torch.sigmoid(pred_realism_on_real).detach()),
            "pred_realism_on_fake": mean_flat(torch.sigmoid(pred_realism_on_fake).detach())
        }

        classification_loss = mean_flat(F.softplus(pred_realism_on_fake)) + mean_flat(F.softplus(-pred_realism_on_real))
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 
        
    
    def compute_loss_fake(self, guidance_model, x1, x0=None, model_kwargs=None):
        t, x0, x1 = self.sample(x1.detach(), x0)
        t_expanded = expand_t_like_x(t, x1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0
            
        model.set_lora_scale(0.0)
        model_kwargs['guidance'] = torch.ones_like(model_kwargs['guidance'], device=x1.device, dtype=x1.dtype)
        out_dict = guidance_model(xt, timesteps=1 - t, **model_kwargs)
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
             

    def training_losses(self, model, guidance_model=None, model_kwargs=None, real_kwargs=None, generator_turn=False, guidance_turn=False, compute_generator_gradient=True, visual=False):
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
                out_dict = model(xt, timesteps=1 - t, **model_kwargs)
                model_output = -out_dict["output"]  
            else:
                with torch.no_grad():
                    out_dict = model(xt, timesteps=1 - t, **model_kwargs)
                    model_output = -out_dict["output"]  
                
            generated_image = xt + model_output * (1 - t_expanded)
            if compute_generator_gradient:
                guidance_model.requires_grad_(False)
                dm_dict, dm_log_dict = self.compute_distribution_matching_loss(guidance_model, generated_image, x0, model_kwargs)
                loss_dict.update(dm_dict)
                log_dict.update(dm_log_dict)
                
                cls_loss_dict = self.compute_generator_clean_cls_loss(guidance_model, generated_image, x0, model_kwargs)
                loss_dict.update(cls_loss_dict)
                guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}
        
            if visual:
                decode_key = [
                    "dmtrain_pred_real_image", "dmtrain_pred_fake_image"
                ]

                with torch.no_grad():
                    if compute_generator_gradient:
                        for key in decode_key:
                            latents = log_dict[key].detach()[:self.num_visuals]
                            latents = self.unpack(latents, height, width)
                            log_dict[key+"_decoded"] = self.decode_image(latents) 
                    
                    latents = generated_image[:self.num_visuals].detach()
                    latents = self.unpack(latents, height, width)
                    log_dict["generated_image"] = self.decode_image(latents)

                    latents = x1[:self.num_visuals].detach()
                    latents = self.unpack(latents, height, width)
                    log_dict["original_clean_image"] = self.decode_image(latents)

            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                # "text_embedding": text_embedding.detach(),
                # "pooled_text_embedding": pooled_text_embedding.detach(),
                # "uncond_embedding": uncond_embedding.detach(),
                # "real_train_dict": real_train_dict,
                # "unet_added_conditions": unet_added_conditions,
                # "uncond_unet_added_conditions": uncond_unet_added_conditions
            }

            log_dict['denoising_timestep'] = t

        # guidance turn
        if guidance_turn:
            generated_image = model_kwargs.pop('img')
            x0 = model_kwargs.pop('noise')
            fake_dict, fake_log_dict = self.compute_loss_fake(guidance_model, generated_image, x0, model_kwargs)
            
            loss_dict = fake_dict 
            log_dict = fake_log_dict

            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dictreal_image.detach(),
                fake_image=generated_image
            )
            
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)

        return loss_dict, log_dict

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                model(x, t, **kwargs), x, t
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=True,
        time_shifting_factor=None, 
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        # for flux
        drift = lambda x, t, model, **kwargs: -self.drift(x, torch.ones_like(t) * (1 - t), model, **kwargs)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            do_shift=do_shift,
            time_shifting_factor=time_shifting_factor,
        )

        return _ode.sample