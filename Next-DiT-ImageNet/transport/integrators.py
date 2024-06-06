import torch as th
from torchdiffeq import odeint
from tqdm import tqdm

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
        return xhat + 0.5 * self.dt * (K1 + K2), xhat  # at last time point we do not perform the heun step

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
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples


class ode_imp:
    """Self implemented ODE solver class using Euler method"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        num_steps,
        time_shifting_factor=None,
        end_time=0,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.end_time = end_time
        self.t = th.linspace(t0, t1, num_steps)
        if time_shifting_factor == 0:
            print('Using S shape time schedule!')
            t_1 = 1 / (1 + th.exp(-6 * (self.t - 0.8)))
            t_2 = 1 - 1 / (1 + th.exp(20 * (self.t - 0.8)))
            self.t = th.where(self.t < 0.8, t_1, t_2)
        else:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
            self.end_time = end_time / (end_time + time_shifting_factor - time_shifting_factor * end_time)
        print(time_shifting_factor == 0, self.t)
        
    def sample(self, x, model, **model_kwargs):
        """Euler method for ODE"""
        device = x[0].device if isinstance(x, list) else x.device
        all_x = [x.detach().cpu()]
        for idx, t0, t1 in tqdm(zip(range(len(self.t) - 1), self.t[:-1], self.t[1:])):
            with th.no_grad():
                dt = t1 - t0
                t0 = th.ones(x[0].size(0)).to(device) * t0 if isinstance(x, list) else th.ones(x.size(0)).to(device) * t0
                model_output = self.drift(x, t0, model, **model_kwargs)
                if isinstance(x, th.Tensor):
                    x = x + model_output * dt
                else:
                    x = [x[i] + model_output[i] * dt for i in range(len(x))]
                all_x.append(x.detach().cpu())
        all_x = th.stack(all_x, dim=0)
        return all_x