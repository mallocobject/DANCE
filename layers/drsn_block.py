import torch
import torch.nn as nn
import torch.nn.functional as F

from .shcink import Shrink


class DRSNBlock(nn.Module):
    expansion: int = 1

    def __init__(self, chin: int, chout: int, stride: int = 1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv1d(chin, chout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(chout),
            nn.ReLU(),
            nn.Conv1d(
                chout,
                chout * DRSNBlock.expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(chout * DRSNBlock.expansion),
            Shrink(chout),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or chin != DRSNBlock.expansion * chout:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    chin,
                    DRSNBlock.expansion * chout,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(DRSNBlock.expansion * chout),
            )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.residual_function(x) + self.shortcut(x))


if __name__ == "__main__":
    x = torch.randn(100, 25, 24)
    drsn = DRSNBlock(25, 25)
    x = drsn(x)
    print(x.shape)
