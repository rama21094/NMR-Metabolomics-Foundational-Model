"""Shared 1D building blocks for the VAE and diffusion U-Net."""
import torch
import torch.nn as nn


def norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock1D(nn.Module):
    """Two conv/norm/SiLU stages with a residual skip. No time conditioning
    (used by the VAE encoder/decoder)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = norm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Down1D(nn.Module):
    """Stride-2 conv, halves sequence length exactly (length must be even)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up1D(nn.Module):
    """Stride-2 transposed conv, doubles sequence length exactly."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class SelfAttention1D(nn.Module):
    """Standard multi-head self-attention over the sequence dimension."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        # x: (B, C, L)
        h = self.norm(x).transpose(1, 2)  # (B, L, C)
        out, _ = self.attn(h, h, h, need_weights=False)
        out = out.transpose(1, 2)  # (B, C, L)
        return x + out
