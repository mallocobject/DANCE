import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShrink(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch),
            nn.BatchNorm1d(ch),
            nn.LeakyReLU(),
            nn.Linear(ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        x_raw = x
        x_abs = x.abs()
        x_stat = self.gap(x_abs)
        x_coef = torch.flatten(x_stat, 1)
        x_coef = self.fc(x_coef)
        x_threshold = x_stat * x_coef.unsqueeze(2)
        x = x_abs - x_threshold
        x = torch.sign(x_raw) * F.relu(x)
        return x


class SpatialEnhance(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                ch,
                ch // 2,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(ch // 2),
            nn.LeakyReLU(),
            nn.Conv1d(
                ch // 2,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        x_raw = x
        x_abs = x.abs()
        x_attention = self.conv(x_abs)
        x = x_raw * x_attention.expand_as(x_raw)
        return x


class CIAD(nn.Module):
    def __init__(self, ch: int, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_shrink = ChannelShrink(ch)
        self.spatial_enhance = SpatialEnhance(ch, spatial_kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.channel_shrink(x)
        x = self.spatial_enhance(x)
        return x
