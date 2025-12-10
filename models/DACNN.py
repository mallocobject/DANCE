import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from layers import APReLU, DAM


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
        self.conv_kernel = [13, 7, 7, 7]
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

        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            output_padding=1,
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
        self.conv_kernel = [7, 7, 7, 13]
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
