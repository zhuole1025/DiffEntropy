import torch as th
from torchdiffeq import odeint
from .utils import time_shift, get_lin_function

class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return (
            xhat + 0.5 * self.dt * (K1 + K2),
            xhat,
        )  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        do_shift=True,
        time_shifting_factor=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.do_shift = do_shift
        self.t = th.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, model_kwargs, controlnet=None, controlnet_kwargs=None):
        device = x[0].device if isinstance(x, tuple) else x.device
        if controlnet is not None:
            x_cond = controlnet_kwargs.pop("controlnet_cond") 
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            if controlnet is not None:
                t_cond = th.ones(x_cond.size(0)).to(device) * 1.0
                noise = th.randn_like(x_cond)
                xt_cond = x_cond * t_cond.view(-1, 1, 1) + noise * (1 - t_cond).view(-1, 1, 1)
                controls = controlnet(xt_cond, timesteps=1 - t_cond, bb_timesteps=1 - t, **controlnet_kwargs)
                model_kwargs["controls"] = controls
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        if self.do_shift:
            mu = get_lin_function(y1=0.5, y2=1.15)(x.shape[1])
            t = time_shift(mu, 1.0, t)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples