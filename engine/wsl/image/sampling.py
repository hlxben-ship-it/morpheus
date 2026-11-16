"""Sigma schedules and samplers for the Image Playground, written to match ComfyUI's math (comfy/samplers.py,
comfy/k_diffusion/sampling.py, comfy/model_sampling.py) so a picture made here can be compared with the same seed there.

Two model kinds:
  eps   SDXL: discrete DDPM schedule (scaled_linear 0.00085..0.012, 1000 steps), the model predicts the noise.
  flow  Qwen-Image: rectified flow, sigma in (0, 1], shifted (ComfyUI ModelSamplingAuraFlow), the model predicts velocity.

Every model is wrapped as denoise(x, sigma) -> denoised x0 (k-diffusion convention); the samplers only see that.

Parity limits (known, on purpose): dpmpp_2m_sde draws plain Gaussian noise where ComfyUI uses a Brownian tree (torchsde), so
it never matches ComfyUI pixel for pixel; and the dpmpp samplers use the eps-style log-sigma steps on flow models too (the
image schema offers them on SDXL only). euler / euler_ancestral / dpmpp_2m on SDXL and euler on Qwen are the parity set.
"""
import math

import torch

SAMPLERS = ("euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde")
SCHEDULERS = ("normal", "karras", "simple")


# ---- model sampling (the sigma table of a model) ----

class Discrete:
    """ComfyUI ModelSamplingDiscrete for eps models (SD 1.5 / SDXL)."""
    kind = "eps"

    def __init__(self, beta_start=0.00085, beta_end=0.012, timesteps=1000):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=torch.float64) ** 2
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.sigmas = (((1 - acp) / acp) ** 0.5).float()
        self.log_sigmas = self.sigmas.log()

    @property
    def sigma_min(self): return self.sigmas[0]

    @property
    def sigma_max(self): return self.sigmas[-1]

    def timestep(self, sigma):
        """The (integer) model timestep closest to sigma in log space."""
        dists = sigma.log().reshape(1, -1) - self.log_sigmas.reshape(-1, 1).to(sigma.device)
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def sigma(self, t):
        t = torch.clamp(torch.as_tensor(t, dtype=torch.float32), 0, len(self.sigmas) - 1)
        lo, hi = t.floor().long(), t.ceil().long()
        w = t.frac()
        return ((1 - w) * self.log_sigmas[lo] + w * self.log_sigmas[hi]).exp()


class Flow:
    """ComfyUI ModelSamplingDiscreteFlow with an AuraFlow shift (multiplier 1): sigma = shift*t / (1 + (shift-1)*t)."""
    kind = "flow"

    def __init__(self, shift=3.1, timesteps=1000):
        self.shift = shift
        t = torch.arange(1, timesteps + 1, dtype=torch.float32) / timesteps
        self.sigmas = self._shift(t)
        self.log_sigmas = self.sigmas.log()

    def _shift(self, t):
        return self.shift * t / (1 + (self.shift - 1) * t)

    @property
    def sigma_min(self): return self.sigmas[0]

    @property
    def sigma_max(self): return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma  # multiplier 1: the model takes sigma itself

    def sigma(self, t):
        return self._shift(torch.as_tensor(t, dtype=torch.float32))


# ---- schedulers ----

def normal(ms, steps):
    """ComfyUI normal_scheduler (sgm=False)."""
    start, end = float(ms.timestep(ms.sigma_max)), float(ms.timestep(ms.sigma_min))
    append_zero = True
    if math.isclose(float(ms.sigma(end)), 0, abs_tol=1e-5):
        steps += 1; append_zero = False
    ts = torch.linspace(start, end, steps)
    sig = [float(ms.sigma(t)) for t in ts]
    return torch.tensor(sig + ([0.0] if append_zero else []), dtype=torch.float32)


def karras(ms, steps, rho=7.0):
    smin, smax = float(ms.sigma_min), float(ms.sigma_max)
    ramp = torch.linspace(0, 1, steps)
    lo, hi = smin ** (1 / rho), smax ** (1 / rho)
    sig = (hi + ramp * (lo - hi)) ** rho
    return torch.cat([sig, sig.new_zeros([1])]).float()


def simple(ms, steps):
    """ComfyUI simple_scheduler: evenly spaced entries of the model's own sigma table."""
    ss = len(ms.sigmas) / steps
    return torch.tensor([float(ms.sigmas[-(1 + int(x * ss))]) for x in range(steps)] + [0.0], dtype=torch.float32)


SCHEDULE = {"normal": normal, "karras": karras, "simple": simple}


def sigmas_for(ms, scheduler, steps, denoise=1.0):
    """ComfyUI KSampler: with denoise < 1 it schedules steps/denoise and keeps the last steps+1 sigmas."""
    if scheduler not in SCHEDULE:
        raise ValueError(f"Scheduler: one of {', '.join(SCHEDULERS)}.")
    if denoise >= 0.9999:
        return SCHEDULE[scheduler](ms, steps)
    if denoise <= 0:
        return torch.zeros(0)
    total = int(steps / denoise)
    return SCHEDULE[scheduler](ms, total)[-(steps + 1):]


# ---- noise ----

def initial_noise(shape, seed):
    """ComfyUI prepare_noise: float32 randn on the CPU from torch.manual_seed(seed)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float32, device="cpu")


def noise_scaling(kind, sigma, noise, latent, max_denoise):
    """Where sampling starts (ComfyUI EPS / CONST noise_scaling)."""
    if kind == "flow":
        return sigma * noise + (1.0 - sigma) * latent
    scale = torch.sqrt(1.0 + sigma ** 2) if max_denoise else sigma
    return noise * scale + latent


class NoiseSampler:
    """k-diffusion default_noise_sampler: a generator on the latent's device seeded with the job seed."""

    def __init__(self, x, seed):
        self.g = torch.Generator(device=x.device).manual_seed(int(seed))

    def __call__(self, x):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=self.g)


# ---- samplers (k-diffusion) ----

def _ancestral(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    down = (sigma_to ** 2 - up ** 2) ** 0.5
    return down, up


def _ancestral_flow(sigma, sigma_next, eta=1.0):
    """ComfyUI's RF variant of euler_ancestral (sample_euler_ancestral_RF)."""
    if sigma_next == 0:
        return 0.0, 0.0, 1.0
    down = sigma_next * (1 + (sigma_next / sigma - 1) * eta)
    alpha_ip1, alpha_down = 1 - sigma_next, 1 - down
    renoise = (sigma_next ** 2 - down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
    return down, renoise, alpha_ip1 / alpha_down


def sample(name, denoise, x, sigmas, kind, seed, callback=None, eta=1.0, stopped=lambda: False):
    """Run one sampler. denoise(x, sigma_tensor) -> x0. callback(i, total, x0) after each step."""
    if name not in SAMPLERS:
        raise ValueError(f"Sampler: one of {', '.join(SAMPLERS)}.")
    n = len(sigmas) - 1
    noise = NoiseSampler(x, seed)
    s_in = x.new_ones([x.shape[0]])
    old_d = None; h_last = None
    for i in range(n):
        if stopped():
            raise InterruptedError("Stopped.")
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        x0 = denoise(x, sigmas[i] * s_in)
        if name == "euler":
            d = (x - x0) / s
            x = x + d * (s_next - s)
        elif name == "euler_ancestral":
            if kind == "flow":
                down, renoise, ratio = _ancestral_flow(s, s_next, eta)
                if s_next == 0:
                    x = x0
                else:
                    x = ratio * (x0 + (x - x0) / s * down) + noise(x) * renoise if renoise else ratio * (x0 + (x - x0) / s * down)
            else:
                down, up = _ancestral(s, s_next, eta)
                d = (x - x0) / s
                x = x + d * (down - s)
                if s_next > 0:
                    x = x + noise(x) * up
        elif name == "dpmpp_2m":
            t, t_next = -math.log(s), (-math.log(s_next) if s_next > 0 else None)
            if s_next == 0:
                x = x0
            else:
                h = t_next - t
                if old_d is None:
                    x = (s_next / s) * x - math.expm1(-h) * x0
                else:
                    r = h_last / h
                    d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * old_d
                    x = (s_next / s) * x - math.expm1(-h) * d
                h_last = h
            old_d = x0
        elif name == "dpmpp_2m_sde":  # midpoint solver, eta 1, s_noise 1 (ComfyUI defaults)
            if s_next == 0:
                x = x0
            else:
                t, t_next = -math.log(s), -math.log(s_next)
                h = t_next - t
                eta_h = eta * h
                x = (s_next / s) * math.exp(-eta_h) * x + (-math.expm1(-h - eta_h)) * x0
                if old_d is not None:
                    r = h_last / h
                    x = x + 0.5 * (-math.expm1(-h - eta_h)) * (1 / r) * (x0 - old_d)
                if eta:
                    x = x + noise(x) * s_next * math.sqrt(-math.expm1(-2 * eta_h))
                h_last = h
            old_d = x0
        if callback:
            callback(i + 1, n, x0)
    return x
