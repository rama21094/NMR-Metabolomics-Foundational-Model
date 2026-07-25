"""Time-conditioned 1D U-Net that predicts the noise added to a VAE latent
at diffusion timestep t. This is the "diffusion" half of the latent-diffusion
design -- architecturally the same family of network Stable Diffusion uses
for its denoiser, adapted from 2D images to a 1D latent sequence."""
import math

import torch
import torch.nn as nn

from common1d import Down1D, SelfAttention1D, Up1D, norm


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlockT1D(nn.Module):
    """ResBlock with time conditioning injected via scale-shift (AdaGN)."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = norm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, 2 * out_channels)
        self.norm2 = norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.time_proj(t_emb)[:, :, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        latent_channels=8,
        base_channels=128,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions_from_end=(0, 1),  # attend at the last N downsample stages
        time_dim=512,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(latent_channels, base_channels, kernel_size=3, padding=1)

        n_stages = len(channel_mult)
        attn_stage_idx = {n_stages - 1 - i for i in attn_resolutions_from_end}

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = base_channels
        self.down_channels = [ch]
        for stage_idx, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage_blocks.append(ResBlockT1D(ch, out_ch, time_dim))
                ch = out_ch
                self.down_channels.append(ch)
            self.down_blocks.append(stage_blocks)
            self.down_attn.append(SelfAttention1D(ch) if stage_idx in attn_stage_idx else None)
            is_last = stage_idx == n_stages - 1
            self.downsamplers.append(Down1D(ch) if not is_last else None)
            if not is_last:
                # forward() pushes an extra skip (same channel count) right after downsampling
                self.down_channels.append(ch)

        self.mid_block1 = ResBlockT1D(ch, ch, time_dim)
        self.mid_attn = SelfAttention1D(ch)
        self.mid_block2 = ResBlockT1D(ch, ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for stage_idx, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = self.down_channels.pop()
                stage_blocks.append(ResBlockT1D(ch + skip_ch, out_ch, time_dim))
                ch = out_ch
            self.up_blocks.append(stage_blocks)
            self.up_attn.append(SelfAttention1D(ch) if stage_idx in attn_stage_idx else None)
            is_first = stage_idx == 0
            self.upsamplers.append(Up1D(ch) if not is_first else None)

        self.out_norm = norm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(ch, latent_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        skips = [h]
        for stage_blocks, attn, downsampler in zip(self.down_blocks, self.down_attn, self.downsamplers):
            for block in stage_blocks:
                h = block(h, t_emb)
                skips.append(h)
            if attn is not None:
                h = attn(h)
                skips[-1] = h
            if downsampler is not None:
                h = downsampler(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for stage_blocks, attn, upsampler in zip(self.up_blocks, self.up_attn, self.upsamplers):
            for block in stage_blocks:
                h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
            if attn is not None:
                h = attn(h)
            if upsampler is not None:
                h = upsampler(h)

        return self.out_conv(self.out_act(self.out_norm(h)))
