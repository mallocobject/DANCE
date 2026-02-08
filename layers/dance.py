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

    def __init__(self, ch: int, *args, **kwargs):
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

        self.vis_cache = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_sign = x.sign()
        x_abs = x.abs()
        stat = self.gap(x_abs)
        coef = stat.squeeze(-1)
        coef = self.fc(coef)
        x_threshold = stat * coef.unsqueeze(-1)
        x_denoised = F.relu(x_abs - x_threshold)

        if not self.training:
            self.vis_cache["input_abs"] = x_abs.detach().cpu()
            self.vis_cache["threshold"] = x_threshold.detach().cpu()
            self.vis_cache["output_abs"] = x_denoised.detach().cpu()

        return x_sign * x_denoised


class AREM(nn.Module):
    """
    Adaptive Local Enhancement Module
    """

    def __init__(self, ch: int, kernel_size: int = 7):
        super().__init__()
        mid_ch = ch * 2
        self.enh_mask = nn.Sequential(
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
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.enh_mask(x))


class EAREM(nn.Module):
    """
    Enhanced Adaptive Local Enhancement Module
    """

    def __init__(self, ch: int | None = None, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_pool = torch.cat([avg_out, max_out], dim=1)

        mask = self.sigmoid(self.conv(x_pool))
        return x * (mask)


class DANCE(nn.Module):
    """
    DANCE: Dual Adaptive Noise Cancellation and Enhancement
    """

    def __init__(self, ch: int, kernel_size: int = 7):
        super().__init__()
        self.atnc = ATNC(ch)
        self.arem = EAREM()

    def forward(self, x: torch.Tensor):
        x = self.atnc(x)
        x = self.arem(x)
        return x


class DANCE_inv(nn.Module):
    """
    DANCE: Dual Adaptive Noise Cancellation and Enhancement
    """

    def __init__(self, ch: int, kernel_size: int = 7):
        super().__init__()
        self.arem = EAREM()
        self.atnc = ATNC(ch)

    def forward(self, x: torch.Tensor):
        x = self.arem(x)
        x = self.atnc(x)
        return x
