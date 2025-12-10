import torch
import torch.nn as nn
import torch.nn.functional as F


class DFLNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ALTSN(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_sign = x.sign()
        x_abs = x.abs()
        stat = self.gap(x_abs)
        coef = self.conv(stat)
        x_threshold = stat * coef
        x_denoised = F.relu(x_abs - x_threshold)

        return x_sign * x_denoised


class SSAM(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        mid_ch = ch * 2
        self.to_q1 = nn.Conv1d(ch, mid_ch, kernel_size=1, bias=False)
        self.to_k1 = nn.Conv1d(ch, mid_ch, kernel_size=1, bias=False)
        self.to_v1 = nn.Conv1d(ch, ch, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        q = self.to_q1(x)  # (B, C//2, L)
        k = self.to_k1(x)  # (B, C//2, L)
        v = self.to_v1(x)  # (B, C, L)

        attn = self.softmax(torch.matmul(q.transpose(1, 2), k))

        out = torch.matmul(v, attn)
        return out


class RDSAB(nn.Module):

    def __init__(self, ch: int):
        super().__init__()
        self.altsm = ALTSN(ch)
        self.ssam = SSAM(ch)

    def forward(self, x: torch.Tensor):
        x = self.altsm(x)
        x = self.ssam(x)
        return x


if __name__ == "__main__":
    model = RDSAB(64)
    input = torch.randn(1, 64, 100)
    output = model(input)
    print(output.shape)
