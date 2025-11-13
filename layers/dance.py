import torch
import torch.nn as nn
import torch.nn.functional as F


class CAC(nn.Module):
    """
    Channel Adaptive Compression
    """

    def __init__(self, ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Linear(ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_stat: bool = False):
        B, C, L = x.shape
        x_raw = x
        x_abs = x.abs()
        x_stat = self.gap(x_abs)
        x_coef = torch.flatten(x_stat, 1)
        x_coef = self.fc(x_coef)
        x_threshold = x_stat * x_coef.unsqueeze(2)
        x = x_abs - x_threshold
        x = torch.sign(x_raw) * F.relu(x)

        if return_stat:
            return x, x_stat
        else:
            return x, None


class CSE(nn.Module):
    """
    Channel-Spatial Excitation
    """

    def __init__(self, ch: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                ch,
                ch // reduction,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(ch // reduction),
            nn.ReLU(),
            nn.Conv1d(
                ch // reduction,
                ch,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, aux_stat: torch.Tensor = None):
        B, C, L = x.shape
        x_attn = self.conv(x)

        if aux_stat is not None:
            pass

        x = x * x_attn
        return x


class DANCE(nn.Module):
    """
    Dual Adaptive Noise Compression and Core Excitation
    """

    def __init__(self, ch: int):
        super().__init__()
        self.cac = CAC(ch)
        self.cse = CSE(ch)

    def forward(self, x: torch.Tensor):
        x, aux = self.cac(x, return_stat=True)
        x = self.cse(x, aux_stat=aux)
        return x
