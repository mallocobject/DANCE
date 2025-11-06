import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShrink(nn.Module):
    def __init__(self, ch: int, reduction: int = 2):
        super().__init__()
        # Channels Attention
        self.avg_gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(ch, ch // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(ch // reduction),
            nn.ReLU(),
            nn.Conv1d(ch // reduction, ch, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x_raw = x
        x_abs = x.abs()
        avg_tmp = self.avg_gap(x)
        average = avg_tmp
        x_attn = self.sigmoid(self.fc(avg_tmp))
        x = x_abs - x_attn * average
        x = torch.sign(x_raw) * torch.max(x, torch.zeros_like(x))
        return x


class SpatialShrink(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 5):
        super().__init__()

        # Spatial Attention
        self.conv = nn.Conv1d(
            ch, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x_raw = x
        x_abs = x.abs()
        x_att = self.sigmoid(self.conv(x_abs))
        x = x_abs - x_att * x_abs
        x = torch.sign(x_raw) * torch.max(x, torch.zeros_like(x))
        return x


class CIAD(nn.Module):
    def __init__(self, ch: int, reduction: int = 2, kernel_size: int = 5):
        super().__init__()
        self.cs = ChannelShrink(ch, reduction)
        self.ss = SpatialShrink(ch, kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.cs(x)
        x = self.ss(x)
        return x


if __name__ == "__main__":
    x = torch.randn(100, 16, 256)
    net = CIAD(16)
    x = net(x)
    print(x.shape)
