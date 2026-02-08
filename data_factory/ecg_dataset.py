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

        meta_path = os.path.join(split_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r") as f:
            self.meta_info = json.load(f)

        if snr_db not in self.meta_info["snr_levels"]:
            raise ValueError(
                f"Unsupported SNR level: {snr_db}. Available: {self.meta_info['snr_levels']}"
            )

        if noise_type not in ["bw", "em", "ma", "emb"]:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        if split == "train":
            self.indices = self.meta_info["split"]["train_indices"]
        elif split == "test":
            self.indices = self.meta_info["split"]["test_indices"]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.noisy_signals = np.load(
            os.path.join(split_dir, f"noisy_{noise_type}_snr_{snr_db}.npy")
        )
        self.clean_signals = np.load(os.path.join(split_dir, "clean_all.npy"))

        train_noisy = self.noisy_signals[self.meta_info["split"]["train_indices"]]

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

        noisy_tensor = torch.from_numpy(self.noisy_signals[data_idx]).float()
        clean_tensor = torch.from_numpy(self.clean_signals[data_idx]).float()

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
