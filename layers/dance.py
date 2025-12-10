import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM
from .eca import ECA
from .se import SE


class ATNC(nn.Module):
    """
    Adaptive Threshold Noise Canceller
    """

    def __init__(self, ch: int):
        super().__init__()
        mid_ch = ch
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid_ch),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Linear(mid_ch, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_sign = x.sign()
        x_abs = x.abs()
        stat = self.gap(x_abs)
        coef = stat.squeeze(-1)
        coef = self.fc(coef)
        x_threshold = stat * coef.unsqueeze(-1)
        x_denoised = F.relu(x_abs - x_threshold)

        return x_sign * x_denoised


## for visualization purpose only
# class ATNC(nn.Module):
#     def __init__(self, ch: int):
#         super().__init__()
#         mid_ch = ch
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(ch, mid_ch),
#             nn.BatchNorm1d(mid_ch),
#             nn.ReLU(),
#             nn.Linear(mid_ch, ch),
#             nn.Sigmoid(),
#         )
#         # 增加一个用于缓存可视化数据的字典
#         self.vis_cache = {}

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, C, L)
#         x_sign = x.sign()
#         x_abs = x.abs()
#         stat = self.gap(x_abs)
#         coef = stat.squeeze(-1)
#         coef = self.fc(coef)
#         x_threshold = stat * coef.unsqueeze(-1)

#         # --- 核心修改：软阈值操作 ---
#         # 你的原代码: x_denoised = F.relu(x_abs - x_threshold)
#         # 建议加上 keepdim=True 等细节确保维度正确，不过原代码也是对的
#         x_denoised = F.relu(x_abs - x_threshold)

#         # --- 关键步骤：把中间结果存起来 ---
#         # 只在 eval 模式下存储，避免训练时占用显存
#         if not self.training:
#             self.vis_cache["input_abs"] = x_abs.detach().cpu()
#             self.vis_cache["threshold"] = x_threshold.detach().cpu()
#             self.vis_cache["output_abs"] = x_denoised.detach().cpu()

#         return x_sign * x_denoised


class AREM(nn.Module):
    """
    Adaptive Local Enhancement Module
    """

    def __init__(self, ch: int, kernel_size: int = 7):
        super().__init__()
        mid_ch = ch * 2
        self.enh_mask = nn.Sequential(
            nn.Conv1d(
                ch,
                mid_ch,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(
                mid_ch,
                mid_ch,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU(),
            nn.Conv1d(
                mid_ch,
                ch,
                kernel_size=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.enh_mask(x)


class DANCE(nn.Module):
    """
    DANCE: Dual Adaptive Noise Cancellation and Enhancement
    """

    def __init__(self, ch: int):
        super().__init__()
        self.atnc = ATNC(ch)
        self.arem = AREM(ch)

    def forward(self, x: torch.Tensor):
        x = self.atnc(x)
        x = self.arem(x)
        return x


class DANCE_inv(nn.Module):
    """
    DANCE: Dual Adaptive Noise Cancellation and Enhancement
    """

    def __init__(self, ch: int):
        super().__init__()
        self.aRem = AREM(ch)
        self.atnc = ATNC(ch)

    def forward(self, x: torch.Tensor):
        x = self.aRem(x)
        x = self.atnc(x)
        return x
