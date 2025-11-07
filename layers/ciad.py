import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelShrink(nn.Module):
    """
    ChannelShrink (通道收缩): 对应于小波的尺度/频率维度
    不同通道捕获信号的不同频率成分, 为每个"频率子带"学习自适应的收缩策略, 识别哪些频段可能包含更多噪声
    """

    def __init__(self, ch: int, reduction: int = 2):
        super().__init__()
        # Channels Attention
        self.avg_gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(ch, ch // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(ch // reduction, ch, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x_raw = x
        x_abs = x.abs()
        avg_tmp = self.avg_gap(x)
        average = avg_tmp
        x_attn = self.sigmoid(self.fc(avg_tmp))
        x = x_abs - x_attn * average
        x = torch.sign(x_raw) * torch.max(x, torch.zeros_like(x))
        return x


class SpatialShrink(nn.Module):
    """
    SpatialShrink (空间收缩): 对应于小波的空间/时间维度
    通过局部卷积操作, 识别噪声在时间轴上的分布模式, 实现空间自适应的阈值处理
    """

    def __init__(self, ch: int, kernel_size: int = 5):
        super().__init__()

        # Spatial Attention
        self.conv = nn.Conv1d(
            ch, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x_raw = x
        x_abs = x.abs()
        x_att = self.sigmoid(self.conv(x_abs))
        x = x_abs - x_att * x_abs  # 使用x_abs而非全局平均, 实现局部自适应
        x = torch.sign(x_raw) * torch.max(x, torch.zeros_like(x))
        return x


# class SpatialShrink(nn.Module):
#     """
#     SpatialShrink (空间收缩): 对应于小波的空间/时间维度
#     通过局部卷积操作, 识别噪声在时间轴上的分布模式, 实现空间自适应的阈值处理
#     """

#     def __init__(self, ch: int, kernel_size: int = 3):
#         super().__init__()

#         # Spatial Attention
#         self.conv = nn.Conv1d(
#             1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: torch.Tensor):
#         x_raw = x
#         x_abs = x.abs()
#         avg_tmp = torch.mean(x_abs, dim=1, keepdim=True)
#         average = avg_tmp
#         x_att = self.sigmoid(self.conv(avg_tmp))
#         x = x_abs - x_att * average
#         x = torch.sign(x_raw) * torch.max(x, torch.zeros_like(x))
#         return x


class CIAD(nn.Module):
    def __init__(self, ch: int, reduction: int = 2, kernel_size: int = 3):
        super().__init__()
        self.cs = ChannelShrink(ch, reduction)
        self.ss = SpatialShrink(ch, kernel_size)

    def forward(self, x: torch.Tensor):
        x = self.cs(x)
        x = self.ss(x)
        return x


if __name__ == "__main__":
    x = torch.randn(100, 16, 256)
    net = CIAD(16)
    x = net(x)
    print(x.shape)
