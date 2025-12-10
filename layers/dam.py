import torch
import torch.nn as nn
import torch.nn.functional as F


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
