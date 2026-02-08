# utils.py
import pywt
import numpy as np
from scipy.fft import rfft, irfft
from PyEMD import EMD


def wavelet_denoise(ecg_datas: np.ndarray, wavelet="db8"):
    """
    for emb -4 T[0] = ∞, other -> 0.5
    for emb -2 T[0] = 0.2, other -> 0.4
    for emb 0 T[0] = 0.2, other -> 0.3
    for emb 2 T[0] = 0.1, other -> 0.2
    for emb 4 T[0] = 0.1, other -> 0.2
    for bw -4 T[0] = 0.12, other -> 0.2
    for em -4 T[0] = ∞, other -> 0.4
    for ma -4 T[0] = 0.2, other -> 1.2
    ---------------------------
    """
    original_shape = ecg_datas.shape
    ndim = ecg_datas.ndim

    if ndim == 3:
        B, C, L = ecg_datas.shape
        ecg_datas = ecg_datas.reshape(-1, L)  # (B*C, L)
    elif ndim == 2:
        ecg_datas = ecg_datas.copy()
    else:
        raise ValueError("Input must be 2D or 3D array.")

    denoised_list = []
    for sig in ecg_datas:
        maxlev = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
        coeffs = pywt.wavedec(sig, wavelet, level=maxlev)
        coeffs[0] = np.zeros_like(1, coeffs[0])  # 低频系数置零
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            sigma = np.median(np.abs(detail)) / 0.6745
            if i == 0:
                # T = sigma * np.sqrt(2 * np.log(len(sig))) * 0.2
                pass
            else:
                T = sigma * np.sqrt(2 * np.log(len(sig))) * 0.5

            coeffs[i] = pywt.threshold(detail, T, mode="soft")

        recon = pywt.waverec(coeffs, wavelet)
        denoised_list.append(recon)

    denoised = np.array(denoised_list)
    return denoised.reshape(original_shape)


def emd_denoise(ecg_datas: np.ndarray):
    """
    for emb -4 imfs[1:-4, :]
    for emb -2 imfs[0:-3, :]
    for emb 0 imfs[0:-2, :]
    for emb 2 imfs[0:-1, :]
    for emb 4 imfs[:, :]
    for bw -4 imfs[0:-1, :]
    for em -4 imfs[0:-4, :]
    for ma -4 imfs[1:-1, :]
    ---------------------------
    """
    original_shape = ecg_datas.shape
    ndim = ecg_datas.ndim

    if ndim == 3:
        B, C, L = ecg_datas.shape
        ecg_datas = ecg_datas.reshape(-1, L)  # (B*C, L)
    elif ndim == 2:
        ecg_datas = ecg_datas.copy()
    else:
        raise ValueError("Input must be 2D or 3D array.")

    denoised_list = []
    for sig in ecg_datas:
        # 1. 初始化 EMD 对象
        emd = EMD()

        # 2. 对信号进行分解，得到 IMFs (Intrinsic Mode Functions)
        # IMFs 的形状为 (n_imfs, sig_len)
        imfs = emd.emd(sig)

        denoised_sig = np.sum(imfs[1:-4, :], axis=0)

        denoised_list.append(denoised_sig)

    denoised = np.array(denoised_list)
    return denoised.reshape(original_shape)
