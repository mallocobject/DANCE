import torch
import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DANCE
from data_factory import ECGDataset
from utils import wavelet_denoise, compute_metrics, emd_denoise

dataset = ECGDataset(
    split="test",
    noise_type="ma",
    snr_db=-4,
    split_dir="./data_split",
)

mean, std = dataset.get_stats()

print(mean.shape, std.shape)

noisy_signal, clean_signal = dataset[:]

denoised_signals = emd_denoise(noisy_signal.numpy())

metrics_res = compute_metrics(
    torch.from_numpy(denoised_signals), clean_signal, mean, std
)

print("Denoising Results using Conventional Denoising:")
for metric_name, value in metrics_res.items():
    print(f"{metric_name}: {value:.4f}")
