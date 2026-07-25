"""Gaussian diffusion schedule + training loss + DDPM/DDIM samplers, operating
on VAE latents. Cosine beta schedule (Nichol & Dhariwal, "Improved DDPM")."""
import math

import torch


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


class GaussianDiffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    def _extract(self, arr, t, shape):
        out = arr.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise

    def training_loss(self, model, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        return torch.mean((pred_noise - noise) ** 2)

    @torch.no_grad()
    def ddpm_sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            sqrt_recip_a = self._extract(self.sqrt_recip_alphas, t, x.shape)
            beta_t = self._extract(self.betas, t, x.shape)
            sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            mean = sqrt_recip_a * (x - beta_t / sqrt_om * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                var = self._extract(self.posterior_variance, t, x.shape)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, num_steps=50, eta=0.0):
        step_indices = torch.linspace(0, self.timesteps - 1, num_steps, device=device).long()
        step_indices = torch.unique(step_indices, sorted=True)
        x = torch.randn(shape, device=device)
        for i in reversed(range(len(step_indices))):
            t = step_indices[i]
            t_batch = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[step_indices[i - 1]] if i > 0 else torch.tensor(1.0, device=device)
            x0_pred = (x - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)) if i > 0 else torch.tensor(0.0, device=device)
            dir_xt = torch.sqrt(torch.clamp(1 - a_prev - sigma**2, min=0.0)) * pred_noise
            noise = torch.randn_like(x) if (i > 0 and eta > 0) else torch.zeros_like(x)
            x = torch.sqrt(a_prev) * x0_pred + dir_xt + sigma * noise
        return x


class EMA:
    """Exponential moving average of model parameters, used only for sampling."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
