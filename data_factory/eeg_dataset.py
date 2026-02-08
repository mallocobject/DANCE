import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_metrics


class EEGDataset(Dataset):
    def __init__(
        self,
        split="train",
        type="EMG",
        snr_db: int = 0,
        split_dir="./DeepSeparator/data",
    ):
        self.split = split
        self.type = type
        self.split_dir = split_dir
        self.snr_db = snr_db

        train_noisy_path = os.path.join(
            self.split_dir, f"noisy_{self.type}_snr_{self.snr_db}_train.npy"
        )
        self.__mean = np.mean(np.load(train_noisy_path))
        self.__std = np.std(np.load(train_noisy_path))

        data_path = os.path.join(
            self.split_dir, f"noisy_{self.type}_snr_{self.snr_db}_{self.split}.npy"
        )
        labels_path = os.path.join(self.split_dir, "EEG_all_epochs.npy")

        self.data = np.load(data_path)
        self.labels = (
            np.load(labels_path)[:3000]
            if split == "train"
            else np.load(labels_path)[3000:3400]
        )

        if split == "train":
            self.data = (self.data - self.__mean) / self.__std
            self.labels = (self.labels - self.__mean) / self.__std
        else:
            self.data = (self.data - self.__mean) / self.__std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        data_tensor = torch.from_numpy(sample).float()
        label_tensor = torch.from_numpy(label).float()

        return data_tensor, label_tensor

    def get_stats(self):
        return torch.tensor(self.__mean, dtype=torch.float32), torch.tensor(
            self.__std, dtype=torch.float32
        )
