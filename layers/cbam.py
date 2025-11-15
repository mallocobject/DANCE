import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)

        y = self.fc(avg_y) + self.fc(max_y)
        y = y.view(b, c, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, ch: int, reduction: int = 2, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.ca(x)
        x = self.sa(x)
        return x


if __name__ == "__main__":
    x = torch.randn(100, 2, 256)
    net = CBAM(2)
    x = net(x)
    print(x.shape)
