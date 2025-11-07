import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import ECA


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(EncBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.pool(self.conv(x)))


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DecBlock, self).__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="linear")
        self.relu = nn.LeakyReLU()

        self.eca = ECA()

    def forward(self, x):
        return self.eca(self.relu(self.upsample(self.conv(x))))


class ACDAE(nn.Module):
    def __init__(self) -> None:
        super(ACDAE, self).__init__()

        channels = [2, 16, 32, 64, 128]
        Kernal_Size = [13, 7, 7, 7]
        self.EncList = nn.ModuleList()
        self.DecList = nn.ModuleList()

        for i in range(4):
            self.EncList.append(
                EncBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=Kernal_Size[i],
                )
            )
            self.DecList.append(
                DecBlock(
                    in_channels=channels[-(i + 1)],
                    out_channels=channels[-(i + 2)],
                    kernel_size=Kernal_Size[-(i + 1)],
                )
            )

    def forward(self, x):
        encfeature = []
        for i in range(3):
            x = self.EncList[i](x)
            encfeature.append(x)

        x = self.EncList[3](x)

        for i in range(3):
            x = self.DecList[i](x)
            x += encfeature[-(i + 1)]
        return self.DecList[3](x)


if __name__ == "__main__":
    x = torch.rand(16, 2, 256)
    model = ACDAE()
    output = model(x)
    print(output.shape)  # Should match the input shape (16, 2, 256)
    print(model)  # Print the model architecture
