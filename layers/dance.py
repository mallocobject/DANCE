import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM
from .eca import ECA
from .se import SE


class ATNC(nn.Module):
    """
    Adaptive Threshold Noise Canceller
    """

    def __init__(self, ch: int):
        super().__init__()
        mid_ch = ch
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid_ch),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Linear(mid_ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_sign = x.sign()
        x_abs = x.abs()
        stat = self.gap(x_abs)
        coef = stat.squeeze(-1)
        coef = self.fc(coef)
        x_threshold = stat * coef.unsqueeze(-1)
        x_denoised = F.relu(x_abs - x_threshold)

        return x_sign * x_denoised


class STEM(nn.Module):
    """
    Spatio-Temporal Enhancement Module
    """

    def __init__(self, ch: int, kernel_size: int = 7):
        super().__init__()
        mid_ch = ch * 2
        self.attn = nn.Sequential(
            nn.Conv1d(
                ch,
                mid_ch,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(
                mid_ch,
                mid_ch,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(
                mid_ch,
                ch,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class DANCE(nn.Module):
    """
    DANCE: Dual Adaptive Noise Cancellation and Enhancement
    """

    def __init__(self, ch: int):
        super().__init__()
        self.atnc = ATNC(ch)
        self.stem = STEM(ch)

    def forward(self, x: torch.Tensor):
        x = self.atnc(x)
        x = self.stem(x)
        return x


class DANCE_inv(nn.Module):
    """
    DANCE: Dual Adaptive Noise Cancellation and Enhancement
    """

    def __init__(self, ch: int):
        super().__init__()
        self.atnc = ATNC(ch)
        self.stem = STEM(ch)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.atnc(x)
        return x
