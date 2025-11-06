# split_manager.py
import os
import json
import numpy as np
from typing import Dict, List
import wfdb


class SplitManager:
    def __init__(
        self,
        mitdb_dir: str,
        nstdb_dir: str,
        window_size: int = 256,
        seed: int = 42,
    ):
        self.mitdb_dir = mitdb_dir
        self.nstdb_dir = nstdb_dir
        self.window_size = window_size
        self.seed = seed
        np.random.seed(seed)

    def _get_all_records(self) -> List[str]:
        """获取所有记录文件名（不带扩展名）"""
        return [
            f.split(".")[0] for f in os.listdir(self.mitdb_dir) if f.endswith(".dat")
        ]

    def _load_noise_segments(self) -> Dict[str, np.ndarray]:
        """加载三种单一噪声类型: bw, em, ma"""
        nstdb_files = ["bw", "em", "ma"]
        noise_segments = {}

        for noise_type in nstdb_files:
            rec = wfdb.rdrecord(os.path.join(self.nstdb_dir, noise_type))
            noise_signal = rec.p_signal

            # 随机采样10000个片段
            segments = []
            max_start = len(noise_signal) - self.window_size

            for _ in range(10000):
                start_idx = np.random.randint(0, max_start)
                seg = noise_signal[start_idx : start_idx + self.window_size, :]
                segments.append(seg)

            noise_segments[noise_type] = np.array(segments, dtype=np.float32)
            print(f"Loaded {len(segments)} segments for noise type: {noise_type}")

        return noise_segments

    def _load_clean_segments(self) -> np.ndarray:
        """加载干净ECG信号片段"""
        records = self._get_all_records()
        clean_segments = []

        for record_id in records:
            rec = wfdb.rdrecord(os.path.join(self.mitdb_dir, record_id))
            ecg_signal = rec.p_signal

            max_start = len(ecg_signal) - self.window_size
            num_segments_per_record = max(1, len(ecg_signal) // (self.window_size * 10))

            for _ in range(num_segments_per_record):
                start_idx = np.random.randint(0, max_start)
                seg = ecg_signal[start_idx : start_idx + self.window_size, :]
                clean_segments.append(seg)

        # 随机选择10000个片段
        if len(clean_segments) > 10000:
            indices = np.random.choice(len(clean_segments), 10000, replace=False)
            clean_segments = [clean_segments[i] for i in indices]
        elif len(clean_segments) < 10000:
            # 如果样本不足，使用替换采样
            indices = np.random.choice(len(clean_segments), 10000, replace=True)
            clean_segments = [clean_segments[i] for i in indices]

        return np.array(clean_segments, dtype=np.float32)

    def _calculate_snr_adjustment(
        self, clean_signal: np.ndarray, noise: np.ndarray, target_snr_db: float
    ) -> float:
        """计算调整噪声所需的缩放因子以达到目标SNR"""
        clean_power = np.mean(clean_signal**2, axis=(0, 1), keepdims=True)
        noise_power = np.mean(noise**2, axis=(0, 1), keepdims=True)

        target_noise_power = clean_power / (10 ** (target_snr_db / 10))
        scale_factor = np.sqrt(target_noise_power / noise_power)
        return scale_factor

    def _create_noisy_signals(
        self,
        clean_segments: np.ndarray,
        noise_segments: Dict[str, np.ndarray],
        snr_db: float,
    ) -> Dict[str, np.ndarray]:
        """为四种噪声类型创建带噪声的信号"""
        noisy_signals = {}

        # 创建三种单一噪声信号
        for noise_type, noise_data in noise_segments.items():
            noisy_segments = []

            for i, clean_sig in enumerate(clean_segments):
                noise_idx = i % len(noise_data)
                noise_segment = noise_data[noise_idx].copy()

                scale_factor = self._calculate_snr_adjustment(
                    clean_sig, noise_segment, snr_db
                )
                adjusted_noise = noise_segment * scale_factor
                noisy_sig = clean_sig + adjusted_noise
                noisy_segments.append(noisy_sig)

            noisy_signals[noise_type] = np.array(noisy_segments, dtype=np.float32)
            print(
                f"Created {len(noisy_segments)} {noise_type} noisy segments at SNR {snr_db}dB"
            )

        # 创建混合噪声信号（emb：三种噪声均等混合）
        mixed_noisy_segments = []
        for i, clean_sig in enumerate(clean_segments):
            mixed_noise = np.zeros((self.window_size, 2), dtype=np.float32)

            # 均等混合三种噪声
            for noise_type in ["bw", "em", "ma"]:
                noise_idx = i % len(noise_segments[noise_type])
                noise_segment = noise_segments[noise_type][noise_idx].copy()

                scale_factor = self._calculate_snr_adjustment(
                    clean_sig, noise_segment, snr_db
                )
                adjusted_noise = noise_segment * scale_factor / 3  # 均等分配
                mixed_noise += adjusted_noise

            noisy_sig = clean_sig + mixed_noise
            mixed_noisy_segments.append(noisy_sig)

        noisy_signals["emb"] = np.array(mixed_noisy_segments, dtype=np.float32)
        print(
            f"Created {len(mixed_noisy_segments)} emb (mixed) noisy segments at SNR {snr_db}dB"
        )

        return noisy_signals

    def _zscore_normalize(
        self, signals: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Z-score标准化"""
        normalized = (signals - mean) / (std)
        return normalized

    def save_split(
        self,
        split_dir: str,
        train_ratio: float = 0.8,
        snr_levels: List[float] = None,
    ):
        """保存数据集划分"""
        if snr_levels is None:
            snr_levels = [-4, -2, 0, 2, 4]  # 论文中使用的SNR级别

        print("Loading noise segments...")
        noise_segments = self._load_noise_segments()  # (10000, window_size, 2)

        print("Loading clean segments...")
        clean_segments = self._load_clean_segments()  # (10000, window_size, 2)
        print(f"Using {len(clean_segments)} clean segments")

        # 划分索引（4:1训练测试划分）
        n_total = len(clean_segments)
        indices = np.random.permutation(n_total)

        n_train = int(n_total * train_ratio)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # # 获取mean和std用于标准化
        # train_clean = clean_segments[train_indices]

        # clean_mean = np.mean(train_clean, axis=(0, 1), keepdims=True)
        # clean_std = np.std(train_clean, axis=(0, 1), keepdims=True)

        # # 标准化干净信号
        # clean_normalized = self._zscore_normalize(clean_segments, clean_mean, clean_std)

        # 确保目录存在
        os.makedirs(split_dir, exist_ok=True)

        # # 保存标准化后的干净信号
        # clean_segments = clean_segments.transpose(
        #     0, 2, 1
        # )  # (num_samples, 2, window_size)
        np.save(os.path.join(split_dir, "clean_signals.npy"), clean_segments)

        # 为每个SNR级别创建四种噪声类型的含噪信号
        for snr_db in snr_levels:
            print(f"\nProcessing SNR: {snr_db}dB")

            # 创建带噪声的信号
            noisy_signals_dict = self._create_noisy_signals(
                clean_segments, noise_segments, snr_db
            )

            for noise_type, noisy_data in noisy_signals_dict.items():
                # noisy_normalized = self._zscore_normalize(
                #     noisy_data, clean_mean, clean_std
                # )

                # noisy_data = noisy_data.transpose(
                #     0, 2, 1
                # )  # (num_samples, 2, window_size)
                filename = f"noisy_{noise_type}_snr_{snr_db}.npy"
                np.save(os.path.join(split_dir, filename), noisy_data)
                print(f"Saved {filename}")

        # 保存划分信息
        split_info = {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "total_samples": n_total,
            "train_ratio": train_ratio,
            "test_ratio": 1.0 - train_ratio,
            "window_size": self.window_size,
            "snr_levels": snr_levels,
            "noise_types": ["bw", "em", "ma", "emb"],  # 四种噪声类型
            # "clean_mean": clean_mean.tolist(),
            # "clean_std": clean_std.tolist(),
            "seed": self.seed,
        }

        # 保存划分信息
        split_path = os.path.join(split_dir, "split_info.json")
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

        print(f"\nSaved split info to {split_path}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Test samples: {len(test_indices)}")
        print(f"SNR levels: {snr_levels}")
        print(f"Noise types: bw, em, ma, emb")

        return split_info


if __name__ == "__main__":
    mitdb_dir = "./ECG-Data/mitdb"
    nstdb_dir = "./ECG-Data/nstdb"
    split_dir = "./data_split"

    manager = SplitManager(mitdb_dir, nstdb_dir)

    # 生成数据集
    manager.save_split(
        split_dir=split_dir, train_ratio=0.8, snr_levels=[-4, -2, 0, 2, 4]
    )
