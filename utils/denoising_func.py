# utils.py
import pywt
import numpy as np
from scipy.fft import rfft, irfft


def wavelet_denoise(ecg_datas: np.ndarray, wavelet="db8", mode="soft"):
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
    wave = pywt.Wavelet(wavelet)

    for sig in ecg_datas:
        maxlev = pywt.dwt_max_level(len(sig), wave.dec_len)
        coeffs = pywt.wavedec(sig, wavelet, level=maxlev, mode="symmetric")

        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            sigma = np.median(np.abs(detail)) / 0.6745
            T = sigma * np.sqrt(2 * np.log(len(sig)))

            coeffs[i] = pywt.threshold(detail, T, mode=mode)

        recon = pywt.waverec(coeffs, wavelet, mode="symmetric")
        denoised_list.append(recon)

    denoised = np.array(denoised_list)
    return denoised.reshape(original_shape)


def fft_denoise(ecg_datas: np.ndarray):
    original_shape = ecg_datas.shape
    ndim = ecg_datas.ndim

    if ndim == 3:
        B, C, L = ecg_datas.shape
        ecg_datas = ecg_datas.reshape(-1, L)
    elif ndim == 2:
        ecg_datas = ecg_datas.copy()
    else:
        raise ValueError("Input must be 2D or 3D array.")

    denoised_list = []

    for sig in ecg_datas:
        X = rfft(sig)
        mag = np.abs(X)
        # 频域 VisuShrink
        sigma = np.median(mag) / 0.6745
        T = sigma * np.sqrt(2 * np.log(len(sig)))
        X_deno = np.sign(X) * np.maximum(mag - T, 0)

        recon = irfft(X_deno, n=len(sig))
        denoised_list.append(recon)

    denoised = np.array(denoised_list)
    return denoised.reshape(original_shape)
