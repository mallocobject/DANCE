import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import ECA, DANCE


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
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=True):
        super(DecBlock, self).__init__()
        self.act = act
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        if act:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class ACDAE(nn.Module):
    def __init__(self) -> None:
        super(ACDAE, self).__init__()

        channels = [2, 16, 32, 64, 128]
        kernel_size = [13, 7, 7, 7]
        self.EncList = nn.ModuleList()
        self.DecList = nn.ModuleList()
        self.ens = nn.ModuleList()

        self.bottle_neck = nn.Sequential(
            nn.Conv1d(
                channels[-1],
                channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(),
        )
        self.down = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="linear")

        for i in range(4):
            self.EncList.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size[i],
                )
            )
            self.ens.append(ECA())
            self.DecList.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],
                    out_channels=channels[-(i + 2)],
                    kernel_size=kernel_size[-(i + 1)],
                    act=True if i != 3 else False,
                )
            )

    def forward(self, x):
        encfeature = []
        for i in range(4):
            x = self.EncList[i](x)
            encfeature.append(x)
            x = self.down(x)

        x = self.bottle_neck(x)

        for i in range(4):
            x = self.up(x)
            x = self.ens[i](x)
            x += encfeature[-(i + 1)]
            x = self.DecList[i](x)

        return x


if __name__ == "__main__":
    x = torch.rand(16, 2, 256)
    model = ACDAE()
    output = model(x)
    print(model)  # Print the model architecture
    print(output.shape)  # Should match the input shape (16, 2, 256)
