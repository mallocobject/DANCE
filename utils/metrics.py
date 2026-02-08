import torch


def compute_metrics(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    ecg: bool = True,
) -> dict:
    """
    计算 ECG 信号去噪指标：
    RMSE、PRD、SNR(batch 维度求平均)
    """

    # === 保证形状匹配 ===
    # 输入是 (batch, channels, length)
    if clean.ndim == 2:
        clean = clean.unsqueeze(0)
        denoised = denoised.unsqueeze(0)

    if ecg and mean is not None and std is not None:
        # 反标准化
        assert mean.shape[-1] == clean.shape[1] and std.shape[-1] == clean.shape[1]
        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)

        denoised = denoised * std + mean

    elif not ecg and mean is not None and std is not None:

        denoised = denoised * std + mean

    # 保证浮点精度（避免半精度误差）
    clean = clean.float()
    denoised = denoised.float()

    # === 计算指标 ===
    diff = clean - denoised

    if ecg:
        # RMSE（每样本取均值）
        rmse = torch.sqrt(torch.mean(diff**2, dim=[1, 2]))  # (batch,)
        rmse_mean = rmse.mean().item()

        # SNR（dB）
        sse = torch.sum(diff**2, dim=[1, 2])
        ssc = torch.sum(clean**2, dim=[1, 2])
        snr = 10 * torch.log10(ssc / sse)
        snr_mean = snr.mean().item()
    else:
        rmse = torch.sqrt(torch.mean(diff**2))  # (batch,)
        rmse_mean = rmse.item()

        # SNR（dB）
        sse = torch.sum(diff**2)
        ssc = torch.sum(clean**2)
        snr = 10 * torch.log10(ssc / sse)
        snr_mean = snr.item()

    return {"RMSE": rmse_mean, "SNR": snr_mean}
