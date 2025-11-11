import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import warnings

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import DANCE


class APReLU(nn.Module):
    """An implementation of APReLU(Adaptively Parametric ReLU) from the paper 'Deep Residual Networks With Adaptively Parametric Rectifier Linear Units for Fault Diagnosis'

    Args:
        channels (int): the number of channels of the input.

    Examples:
        >>> m = APReLU(64)
        >>> tensor_1 = torch.randn(2, 64, 32) # batch_size should be greater than 1 since a batch norm layer is used.
        >>> output = m(tensor_1)
        >>> output.shape
        torch.Size([2, 64, 32])
    """

    def __init__(self, channels):
        super(APReLU, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        fcnList = [
            nn.Linear(2 * channels, 2 * channels),
            nn.BatchNorm1d(2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        ]
        self.fcn = nn.Sequential(*fcnList)

    def forward(self, x):
        zerox = torch.zeros_like(x)
        posx = torch.max(x, zerox)
        negx = torch.min(x, zerox)

        concatx = torch.concat(
            [self.gap(posx).squeeze(-1), self.gap(negx).squeeze(-1)], dim=1
        )
        concatx = self.fcn(concatx)
        return posx + concatx.unsqueeze(2) * negx


class EncoderCell(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=2,
        using_APReLU=True,
    ):
        super(EncoderCell, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            # CIAD(out_channels),
        )
        if using_APReLU:
            self.activate = APReLU(out_channels)
        else:
            self.activate = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.activate(out)
        out = self.bn(out)
        # print(out.shape)
        return out


class DeNoiseEnc(nn.Module):
    def __init__(self, using_APReLU=True):
        super(DeNoiseEnc, self).__init__()
        self.conv_kernel = [11, 5, 5, 5]
        self.out_channels = [16, 32, 64, 128]
        self.EncoderList = nn.ModuleList()
        input_channel = 2
        for i in range(4):
            self.EncoderList.add_module(
                "cell{}".format(i),
                EncoderCell(
                    input_channel,
                    self.out_channels[i],
                    self.conv_kernel[i],
                    (self.conv_kernel[i] - 1) // 2,
                    using_APReLU=using_APReLU,
                ),
            )
            input_channel = self.out_channels[i]

    def forward(self, x):
        out = []
        for cell in self.EncoderList:
            x = cell(x)
            out.append(x)
        return out


class DAM(nn.Module):
    """Dual Attention module from the paper 'Dual Attention Convolutional Neural Network Based on Adaptive Parametric ReLU for Denoising ECG Signals with Strong Noise'

    This module contains a spatial attention and a channel attention.

    Args:
        channels (int): the number of channels of the input.

    Examples:
        >>> m = DAM(64)
        >>> tensor_1 = torch.randn(2, 64, 32) # batch_size should be greater than 1 since a batch norm layer is used.
        >>> output = m(tensor_1)
        >>> output.shape
        torch.Size([2, 64, 32])
    """

    def __init__(self, channels):
        super(DAM, self).__init__()
        # Channel Attention
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        fcnList = [
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        ]
        self.fcn1 = nn.Sequential(*fcnList)
        self.fcn2 = nn.Sequential(*fcnList)

        # Spatial Attention
        self.cap = nn.AdaptiveAvgPool1d(1)
        self.cmp = nn.AdaptiveMaxPool1d(1)
        self.convsa = nn.Conv1d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_transpose = x.transpose(1, 2)
        # Channel Attention
        gapx = self.gap(x).squeeze(2)
        gmpx = self.gmp(x).squeeze(2)
        gapx = self.fcn1(gapx)
        gmpx = self.fcn2(gmpx)
        Cattn = self.sigmoid(gapx + gmpx).unsqueeze(-1)

        # Spatial Attn
        capx = self.cap(x_transpose).transpose(1, 2)
        cmpx = self.cmp(x_transpose).transpose(1, 2)
        catcp = torch.cat((capx, cmpx), dim=1)
        Sattn = self.sigmoid(self.convsa(catcp).squeeze(1)).unsqueeze(-2)
        x = Cattn * x
        x = Sattn * x
        return x


class DecoderCell(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=2,
        using_APReLU=True,
        last=False,
    ):
        super(DecoderCell, self).__init__()
        self.last = last

        self.deconv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        if using_APReLU:
            self.activate = APReLU(out_channels)
        else:
            self.activate = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(out_channels)

        if last == False:
            self.dam = DAM(out_channels)

    def forward(self, x):
        outx = self.deconv(x)
        outx = self.activate(outx)
        outx = self.bn(outx)
        if self.last is not True:
            outx = self.dam(outx)
        # print(outx.shape)
        return outx


def alignment_add(tensor1, tensor2, alignment_opt="trunc"):
    """add with auto-alignment

    Using for the transpose convolution. Transpose convolution will cause the size of the output uncertain. However, in the unet structure, the size of the output should be the same as the input. So, we need to align the size of the output with the input.

    Args:
        tensor1: the first tensor
        tensor2: the second tensor, only the last dim is not same as the first tensor
        alignment_opt: the alignment option, can be 'trunc' or 'padding'

    Examples:
        >>> tensor1 = torch,randn(1, 2, 3)
        >>> tensor2 = torch.randn(1, 2, 4)
        >>> tensor3 = alignment_add(tensor1, tensor2)
        >>> tensor3.shape
        torch.Size([1, 2, 3])

    """

    assert (
        tensor1.shape[0:-1] == tensor2.shape[0:-1]
    ), "the shape of the first tensor should be the same as the second tensor"
    short_tensor = tensor1 if tensor1.shape[-1] < tensor2.shape[-1] else tensor2
    long_tensor = tensor1 if tensor1.shape[-1] >= tensor2.shape[-1] else tensor2
    if alignment_opt == "trunc":
        return short_tensor + long_tensor[..., : short_tensor.shape[-1]]
    elif alignment_opt == "padding":
        return long_tensor + F.pad(
            short_tensor, (0, long_tensor.shape[-1] - short_tensor.shape[-1])
        )


class DeNoiseDec(nn.Module):

    def __init__(
        self,
    ):
        super(DeNoiseDec, self).__init__()
        self.conv_kernel = [5, 5, 5, 11]
        self.out_channels = [64, 32, 16, 2]
        DecoderList = []
        in_channels = 128
        for i in range(4):
            if i != 3:
                DecoderList.append(
                    DecoderCell(
                        in_channels,
                        self.out_channels[i],
                        self.conv_kernel[i],
                        (self.conv_kernel[i] - 1) // 2,
                        using_APReLU=True,
                    )
                )
            else:
                DecoderList.append(
                    DecoderCell(
                        in_channels,
                        self.out_channels[i],
                        self.conv_kernel[i],
                        (self.conv_kernel[i] - 1) // 2,
                        using_APReLU=True,
                        last=True,
                    )
                )
            in_channels = self.out_channels[i]
        self.DecoderList = nn.ModuleList(DecoderList)

    def forward(self, xlist):
        y3 = self.DecoderList[0](xlist[-1])
        y2 = self.DecoderList[1](alignment_add(y3, xlist[-2]))
        y1 = self.DecoderList[2](alignment_add(y2, xlist[-3]))
        y0 = self.DecoderList[3](alignment_add(y1, xlist[-4]))

        return y0


class DACNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.enc = DeNoiseEnc()
        self.dec = DeNoiseDec()

    def forward(self, x):
        return self.dec(self.enc(x))


if __name__ == "__main__":
    x = torch.randn(10, 2, 256)
    model = DACNN()
    print(model)
    y = model(x)
    print(y.shape)
