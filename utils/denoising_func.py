import pywt
import numpy as np
from scipy.fft import fft, ifft


def wavelet_denoise(ecg_datas: np.ndarray, threshold, wavelet="db8"):
    if len(ecg_datas.shape) == 2:
        datarec = []
        for data in ecg_datas:
            maxlev = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
            coeffs = pywt.wavedec(data, wavelet, level=maxlev)
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            datarec.append(pywt.waverec(coeffs, wavelet))
        return np.array(datarec)
    elif len(ecg_datas.shape) == 3:
        datarec = []
        for data in ecg_datas:
            datarec.append(
                wavelet_denoise(data, threshold=threshold, wavelet=wavelet)
            )  # 确保传递参数
        return np.array(datarec)


def fft_denoise(ecg_datas: np.ndarray, threshold):
    if len(ecg_datas.shape) == 2:
        new_ecg_data = []
        for ecg_data in ecg_datas:
            # Apply FFT to the input data
            ecg_fft = fft(ecg_data)

            # Calculate the magnitude of the FFT coefficients
            magnitude = np.abs(ecg_fft)

            # Find the threshold for noise reduction
            cutoff = threshold * np.max(magnitude)

            # Set coefficients below the threshold to zero (remove noise)
            ecg_fft[magnitude < cutoff] = 0

            # Reconstruct the signal using inverse FFT
            denoised_ecg = ifft(ecg_fft)
            new_ecg_data.append(denoised_ecg.real)
        return np.array(new_ecg_data)  # ✅ 关键修复：添加返回语句

    elif len(ecg_datas.shape) == 3:
        new_ecg_data = []
        for ecg_data in ecg_datas:
            new_ecg_data.append(fft_denoise(ecg_data, threshold=threshold))
        return np.array(new_ecg_data)
