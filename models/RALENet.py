import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange

# --- 基础组件 ---


class PartialConv_1d(nn.Module):
    """CVPR 2023 'Run, Don’t Walk': 1D 部分卷积实现"""

    def __init__(self, dim, n_div, forward_type="split_cat"):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv1d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False
        )
        self.forward_type = forward_type

    def forward(self, x):
        if self.forward_type == "split_cat":
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            return torch.cat((x1, x2), 1)
        else:
            x = x.clone()
            x[:, : self.dim_conv3, :] = self.partial_conv3(x[:, : self.dim_conv3, :])
            return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class eca_layer_1d(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        return x * self.sigmoid(y).expand_as(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        local_enhence=False,
        use_partial=True,
        use_eca=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.local_enhence = local_enhence
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.eca = eca_layer_1d() if use_eca else nn.Identity()
        if local_enhence:
            self.leconv = (
                PartialConv_1d(hidden_features, hidden_features)
                if use_partial
                else nn.Conv1d(
                    hidden_features,
                    hidden_features,
                    3,
                    1,
                    1,
                    groups=hidden_features,
                    bias=False,
                )
            )

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        if self.local_enhence:
            x = rearrange(x, "b l c -> b c l")
            x = self.act(self.leconv(x))
            x = rearrange(x, "b c l -> b l c")
        x = self.fc2(x)
        x = rearrange(x, "b l c -> b c l")
        x = self.eca(x)
        return self.drop(rearrange(x, "b c l -> b l c"))


# --- 位置编码与注意力机制 ---


class AbsPositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.0, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros((1, max_len, num_hiddens))
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, num_hiddens, 2) * -(math.log(10000.0) / num_hiddens)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].to(x.device)
        return self.dropout(x)


class MSAttention(nn.Module):
    def __init__(
        self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask[:, :, :N, :N]
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        local_enhence=False,
        pe="abs",
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            dim, int(dim * mlp_ratio), drop=drop, local_enhence=local_enhence
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.pos_enc = AbsPositionalEncoding(dim) if pe == "abs" else nn.Identity()

    def forward(self, x, mask=None):
        def _attn_part(x, m):
            return self.drop_path(self.attn(self.norm1(self.pos_enc(x)), mask=m))

        def _mlp_part(x):
            return self.drop_path(self.mlp(self.norm2(x)))

        x = x + (
            checkpoint.checkpoint(_attn_part, x, mask)
            if self.use_checkpoint
            else _attn_part(x, mask)
        )
        x = x + (
            checkpoint.checkpoint(_mlp_part, x) if self.use_checkpoint else _mlp_part(x)
        )
        return x


# --- 下采样与上采样 ---


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        if x.shape[1] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        x = torch.cat([x[:, 0::2, :], x[:, 1::2, :]], -1)
        return self.reduction(self.norm(x))


class PatchSeparate(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(dim // 2, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        x = rearrange(x, "b l (c1 c2) -> b (c1 l) c2", c1=2)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        drop_path=0.0,
        local_enhence=False,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_heads,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    local_enhence=local_enhence,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim) if downsample else None

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return self.downsample(x) if self.downsample else x


class RelativePositionEmbedding(nn.Module):
    def __init__(self, Length, whole_length, num_heads):
        super().__init__()
        self.Length, self.whole_length = Length, whole_length
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Length - 1), num_heads)
        )
        coords = torch.arange(Length)
        relative_coords = coords[:, None] - coords[None, :] + Length - 1
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self):
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.Length, self.Length, -1)
        bias = bias.permute(2, 0, 1).contiguous()
        pad = self.whole_length - self.Length
        return F.pad(bias, (0, pad, 0, pad)).unsqueeze(0)


# --- 主模型 ---


class RALENet(nn.Module):
    def __init__(self, double_ch: bool = True, high_level_enhence=True):
        super().__init__()

        if double_ch:
            chs = [8, 16, 32, 64, 128]
            hds = [2, 4, 8, 16, 32]
            lens = [256, 128, 64, 32, 16]
        else:
            chs = [4, 8, 16, 32, 64]
            hds = [1, 2, 4, 8, 16]
            lens = [512, 256, 128, 64, 32]

        self.conv1 = nn.Sequential(
            nn.Conv1d(2 if double_ch else 1, chs[0], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(chs[0]),
        )
        self.rwattns = nn.ModuleList(
            [RelativePositionEmbedding(32, lens[i], hds[i]) for i in range(4)]
        )

        # Encoder
        self.enc1 = BasicLayer(
            chs[0], 2, hds[0], local_enhence=high_level_enhence, downsample=PatchMerging
        )
        self.enc2 = BasicLayer(
            chs[1], 2, hds[1], local_enhence=high_level_enhence, downsample=PatchMerging
        )
        self.enc3 = BasicLayer(
            chs[2], 2, hds[2], local_enhence=high_level_enhence, downsample=PatchMerging
        )
        self.enc4 = BasicLayer(
            chs[3], 2, hds[3], local_enhence=high_level_enhence, downsample=PatchMerging
        )

        self.mid = BasicLayer(chs[4], 2, hds[4], local_enhence=high_level_enhence)

        # Decoder
        self.dec4 = BasicLayer(
            chs[4],
            2,
            hds[4],
            local_enhence=high_level_enhence,
            downsample=PatchSeparate,
        )
        self.dec3 = BasicLayer(
            chs[3],
            2,
            hds[3],
            local_enhence=high_level_enhence,
            downsample=PatchSeparate,
        )
        self.dec2 = BasicLayer(
            chs[2],
            2,
            hds[2],
            local_enhence=high_level_enhence,
            downsample=PatchSeparate,
        )
        self.dec1 = BasicLayer(
            chs[1],
            2,
            hds[1],
            local_enhence=high_level_enhence,
            downsample=PatchSeparate,
        )

        self.transconv = nn.Conv1d(chs[0], 2 if double_ch else 1, 3, padding=1)

    def forward(self, x):
        identity = x = self.conv1(x)
        masks = [rw() for rw in self.rwattns]
        x = rearrange(x, "b c l -> b l c")

        # Encoder with skips
        s1 = x
        x = self.enc1(x, masks[0])
        s2 = x
        x = self.enc2(x, masks[1])
        s3 = x
        x = self.enc3(x, masks[2])
        s4 = x
        x = self.enc4(x, masks[3])

        x = self.mid(x) + x

        # Decoder with skip adds
        x = self.dec4(x) + s4
        x = self.dec3(x, masks[3]) + s3
        x = self.dec2(x, masks[2]) + s2
        x = self.dec1(x, masks[1])

        x = rearrange(x, "b l c -> b c l")
        return self.transconv(x + identity)


if __name__ == "__main__":
    x = torch.rand(16, 2, 256)
    model = RALENet()
    print(model)
    y = model(x)
    print(y.shape)
