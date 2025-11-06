import torch
import torch.nn as nn
import torch.nn.functional as F


class Shrink(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch),
            # nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Linear(ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        x = x_abs - x
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x


if __name__ == "__main__":
    x = torch.randn(100, 25, 24)
    sh = Shrink(25, 1)
    x = sh(x)
    print(x.shape)
