import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, ch: int | None = None, kernel_size: int = 7):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


if __name__ == "__main__":
    x = torch.randn(100, 2, 256)
    net = ECA()
    x = net(x)
    print(x.shape)
