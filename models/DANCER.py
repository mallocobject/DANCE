import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import DANCE, CAC


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(EncBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            DANCE(out_channels),  # non-linear layer
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=True):
        super(DecBlock, self).__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )
        if act:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)

        return x


class DANCER(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        channels = [2, 16, 32, 64, 128]
        kernal_size = [13, 7, 7, 7]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="linear")
        self.bottle_neck = nn.Sequential(
            nn.Conv1d(
                channels[-1],
                channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(),
        )

        for i in range(4):
            self.encoder.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernal_size[i],
                )
            )
            self.decoder.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],
                    out_channels=channels[-(i + 2)],
                    kernel_size=kernal_size[-(i + 1)],
                    act=True if i != 3 else False,
                )
            )

    def forward(self, x):
        encfeature = []
        for i in range(4):
            x = self.encoder[i](x)
            encfeature.append(x)
            x = self.down(x)
        x = self.bottle_neck(x)
        for i in range(4):
            x = self.up(x)
            x += encfeature[-(i + 1)]
            x = self.decoder[i](x)
        return x


if __name__ == "__main__":
    x = torch.rand(16, 2, 256)
    model = DANCER()
    print(model)
    y = model(x)
    print(y.shape)
