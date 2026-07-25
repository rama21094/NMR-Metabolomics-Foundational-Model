"""1D convolutional VAE that compresses a length-131072 spectrum into a
small (latent_channels, latent_length) latent grid. This is the "VAE" half
of the latent-diffusion design: the diffusion model is trained to generate
in this compressed latent space rather than on the raw 131072-point signal,
exactly as Stable Diffusion's VAE compresses pixel space before its U-Net.
"""
import torch
import torch.nn as nn

from common1d import Down1D, ResBlock1D, Up1D, norm


class Encoder1D(nn.Module):
    def __init__(self, in_channels, base_channels, channel_mult, latent_channels):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        stages = []
        ch = base_channels
        for mult in channel_mult:
            out_ch = base_channels * mult
            stages.append(ResBlock1D(ch, out_ch))
            stages.append(Down1D(out_ch))
            ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.mid = ResBlock1D(ch, ch)
        self.out_norm = norm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(ch, 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.in_conv(x)
        for stage in self.stages:
            h = stage(h)
        h = self.mid(h)
        h = self.out_conv(self.out_act(self.out_norm(h)))
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class Decoder1D(nn.Module):
    def __init__(self, out_channels, base_channels, channel_mult, latent_channels):
        super().__init__()
        ch = base_channels * channel_mult[-1]
        self.in_conv = nn.Conv1d(latent_channels, ch, kernel_size=3, padding=1)
        self.mid = ResBlock1D(ch, ch)
        stages = []
        rev_mult = list(reversed(channel_mult))
        for i, mult in enumerate(rev_mult):
            next_mult = rev_mult[i + 1] if i + 1 < len(rev_mult) else 1
            out_ch = base_channels * next_mult
            stages.append(Up1D(ch))
            stages.append(ResBlock1D(ch, out_ch))
            ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.out_norm = norm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
        self.out_sigmoid = nn.Sigmoid()  # data is row-min-max normalized to [0, 1]

    def forward(self, z):
        h = self.in_conv(z)
        h = self.mid(h)
        for stage in self.stages:
            h = stage(h)
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return self.out_sigmoid(h)


class VAE1D(nn.Module):
    def __init__(self, base_channels=32, channel_mult=(1, 2, 2, 4, 4, 6, 6), latent_channels=8):
        super().__init__()
        self.channel_mult = tuple(channel_mult)
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** len(self.channel_mult)
        self.encoder = Encoder1D(1, base_channels, self.channel_mult, latent_channels)
        self.decoder = Decoder1D(1, base_channels, self.channel_mult, latent_channels)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar


def vae_loss(recon, target, mean, logvar, weight=None, kl_weight=1e-4):
    if weight is None:
        recon_loss = torch.mean((recon - target) ** 2)
    else:
        recon_loss = torch.mean(weight * (recon - target) ** 2)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=(1, 2))
    kl_loss = kl_per_sample.mean() / mean.shape[1] / mean.shape[2]
    return recon_loss + kl_weight * kl_loss, recon_loss.detach(), kl_loss.detach()
