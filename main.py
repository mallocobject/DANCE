import numpy as np
import os
import json
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_metrics, wavelet_denoise, fft_denoise

method = "fft"  # wt or fft
noisy_type = "emb"  # bw, em, ma, emb
snr_db = 4  # -4, -2, 0, 2, 4

split_dir = "./data_split"
with open(os.path.join(split_dir, "split_info.json"), "r") as f:
    split_data = json.load(f)

indices = split_data["test_indices"]
noisy_signals = np.load(os.path.join(split_dir, f"noisy_{noisy_type}_snr_{snr_db}.npy"))
clean_signals = np.load(os.path.join(split_dir, "clean_signals.npy"))

noisy_signals = noisy_signals.transpose(0, 2, 1)  # (N, C, L)
clean_signals = clean_signals.transpose(0, 2, 1)  #

print(f"Number of samples: {len(indices)}")

if method == "wt":
    denoised_signals = wavelet_denoise(noisy_signals[indices], threshold=0.074)
elif method == "fft":
    denoised_signals = fft_denoise(noisy_signals[indices], threshold=0.0012)

metrics = compute_metrics(
    denoised=torch.tensor(denoised_signals),
    clean=torch.tensor(clean_signals[indices]),
)

print(f"Method: {method}, Noisy Type: {noisy_type}, SNR: {snr_db} dB")
print(f"RMSE: {metrics['RMSE']:.4f}, SNR: {metrics['SNR']:.4f} dB")
