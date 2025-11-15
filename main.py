# test_denoise.py
import numpy as np
import os
import json
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_metrics, wavelet_denoise, fft_denoise

# ==================== 配置 ====================
method = "wt"  # "wt" or "fft"
noisy_type = "emb"  # "bw", "em", "ma", "emb"
snr_db = 4  # -4, -2, 0, 2, 4
# ============================================

split_dir = "./data_split"
with open(os.path.join(split_dir, "split_info.json"), "r") as f:
    split_data = json.load(f)

indices = split_data["test_indices"]
noisy_signals = np.load(os.path.join(split_dir, f"noisy_{noisy_type}_snr_{snr_db}.npy"))
clean_signals = np.load(os.path.join(split_dir, "clean_signals.npy"))

# (N, L, C) → (N, C, L)
# noisy_signals = noisy_signals.transpose(0, 2, 1)
clean_signals = clean_signals.transpose(0, 2, 1)

# test_noisy = noisy_signals[indices]
test_clean = clean_signals[indices]

print(f"Number of test samples: {len(indices)}")
# print(f"Signal shape: {test_noisy.shape}")  # (N, C, L)

# # ==================== 原始 SNR ====================
# noisy_metrics = compute_metrics(
#     denoised=torch.tensor(test_noisy), clean=torch.tensor(test_clean)
# )
# print(
#     f"\n[原始] SNR: {noisy_metrics['SNR']:+.3f} dB, RMSE: {noisy_metrics['RMSE']:.4f}"
# )

# ==================== 去噪 ====================
for n_type in ["emb", "bw", "ma", "em"]:
    print(f"\n--- Denoising for noise type: {n_type.upper()} ---")
    for snr in [-4, -2, 0, 2, 4]:
        print(f"\n--- SNR: {snr} dB ---")
        noisy_path = os.path.join(split_dir, f"noisy_{n_type}_snr_{snr}.npy")
        noisy_data = np.load(noisy_path).transpose(0, 2, 1)
        test_noisy_data = noisy_data[indices]

        denoised_signals = wavelet_denoise(test_noisy_data)

        metrics = compute_metrics(
            denoised=torch.tensor(denoised_signals),
            clean=torch.tensor(test_clean),
        )

        print(
            f"Type: {n_type.upper()}, SNR: {metrics['SNR']:.4f} dB, RMSE: {metrics['RMSE']:.4f}"
        )

# # ==================== 去噪后 SNR ====================
# metrics = compute_metrics(
#     denoised=torch.tensor(denoised_signals),
#     clean=torch.tensor(test_clean),
# )

# print(f"\n=== 最终结果 ===")
# print(f"Method: {method.upper()}, Noisy: {noisy_type}, SNR: {snr_db} dB")
# print(f"原始 SNR : {noisy_metrics['SNR']:+.4f} dB")
# print(
#     f"去噪后 SNR: {metrics['SNR']:+.4f} dB  ↑ {metrics['SNR'] - noisy_metrics['SNR']:+.4f} dB"
# )
# print(f"RMSE     : {metrics['RMSE']:.4f}")
