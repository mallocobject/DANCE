import torch
import torch.nn as nn
import torch.nn.functional as F


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
