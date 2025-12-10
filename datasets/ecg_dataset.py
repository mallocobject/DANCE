import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_metrics


class ECGDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        noise_type: str = "bw",
        snr_db: int = 0,
        split_dir: str = "./data_split",
    ):
        super().__init__()
        self.split = split
        self.split_dir = split_dir
        self.noise_type = noise_type
        self.snr_db = snr_db

        if snr_db not in [-4, -2, 0, 2, 4]:
            raise ValueError(f"Unsupported SNR level: {snr_db}")

        if noise_type not in ["bw", "em", "ma", "emb"]:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        split_path = os.path.join(split_dir, "split_info.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            self.split_data = json.load(f)

        if split == "train":
            self.indices = self.split_data["train_indices"]
        elif split == "test":
            self.indices = self.split_data["test_indices"]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.noisy_signals = np.load(
            os.path.join(split_dir, f"noisy_{noise_type}_snr_{snr_db}.npy")
        )
        self.clean_signals = np.load(os.path.join(split_dir, "clean_signals.npy"))

        train_noisy = self.noisy_signals[self.split_data["train_indices"]]

        self.__mean = np.mean(train_noisy, axis=(0, 1), keepdims=True)
        self.__std = np.std(train_noisy, axis=(0, 1), keepdims=True)

        if split == "train":
            self.noisy_signals = (self.noisy_signals - self.__mean) / self.__std
            self.clean_signals = (self.clean_signals - self.__mean) / self.__std
        else:
            self.noisy_signals = (self.noisy_signals - self.__mean) / self.__std

        self.noisy_signals = self.noisy_signals.transpose(
            0, 2, 1
        )  # (num_samples, 2, window_size)
        self.clean_signals = self.clean_signals.transpose(
            0, 2, 1
        )  # (num_samples, 2, window_size)

        # print(f"Loaded {split} dataset with {len(self.indices)} samples")

    def get_stats(self):
        return torch.FloatTensor(self.__mean), torch.FloatTensor(self.__std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]

        noisy_signal = self.noisy_signals[data_idx]
        clean_signal = self.clean_signals[data_idx]

        noisy_tensor = torch.FloatTensor(noisy_signal)
        clean_tensor = torch.FloatTensor(clean_signal)

        return noisy_tensor, clean_tensor


if __name__ == "__main__":

    train_dataset = ECGDataset(split="train", split_dir="./data_split")
    test_dataset = ECGDataset(split="test", split_dir="./data_split")

    clean = train_dataset[0][1]
    print(f"sample shape: {clean.shape}")

    print(f"trainset shape: {len(train_dataset)}")
    print(f"testset shape: {len(test_dataset)}")

    noisy, clean = train_dataset[0]
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(noisy[0].numpy(), label="Noisy ECG")
    plt.plot(clean[0].numpy(), label="Clean ECG")
    plt.legend()
    plt.title("ECG Signal Sample from Training Set, channel 0")

    plt.subplot(2, 1, 2)
    plt.plot(noisy[1].numpy(), label="Noisy ECG")
    plt.plot(clean[1].numpy(), label="Clean ECG")
    plt.legend()
    plt.title("ECG Signal Sample from Training Set, channel 1")

    plt.tight_layout()
    plt.show()
