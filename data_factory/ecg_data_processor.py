import os
import json
import numpy as np
import wfdb
import logging
import random
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ECGDataProcessor:
    def __init__(
        self,
        mitdb_dir: str,
        nstdb_dir: str,
        window_size: int = 256,
        target_sample_count: int = 10000,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.mitdb_dir = mitdb_dir
        self.nstdb_dir = nstdb_dir
        self.window_size = window_size
        self.target_sample_count = target_sample_count
        self.train_ratio = train_ratio
        self.seed = seed

        # 固定随机种子
        random.seed(seed)
        np.random.seed(seed)

    def _load_noise_source(self) -> Dict[str, np.ndarray]:
        """加载双通道噪声源 (NSTDB)"""
        noise_types = ["bw", "em", "ma"]
        noise_data = {}
        for nt in noise_types:
            path = os.path.join(self.nstdb_dir, nt)
            rec = wfdb.rdrecord(path)
            noise_data[nt] = rec.p_signal.astype(np.float32)  # (L, 2)
        return noise_data

    def _get_all_available_segments(self) -> List[Dict]:
        """扫描全库，提取所有不重叠片段的元数据"""
        records = sorted(
            list(
                set(
                    [
                        f.split(".")[0]
                        for f in os.listdir(self.mitdb_dir)
                        if f.endswith(".dat")
                    ]
                )
            )
        )
        all_meta_segments = []

        for rid in records:
            path = os.path.join(self.mitdb_dir, rid)
            header = wfdb.rdheader(path)
            sig_len = header.sig_len

            # 计算该记录能切出多少段
            num_segs = sig_len // self.window_size
            for i in range(num_segs):
                all_meta_segments.append(
                    {
                        "record_id": rid,
                        "start": i * self.window_size,
                        "end": (i + 1) * self.window_size,
                    }
                )

        logging.info(
            f"Database scan complete. Found {len(all_meta_segments)} potential unique segments."
        )
        return all_meta_segments

    def generate(self, output_dir: str, snr_levels: List[float] = [-4, -2, 0, 2, 4]):
        os.makedirs(output_dir, exist_ok=True)

        # 1. 提取所有元数据，并按 Record ID (病人) 归类
        all_meta = self._get_all_available_segments()
        records_dict = {}
        for m in all_meta:
            rid = m["record_id"]
            if rid not in records_dict:
                records_dict[rid] = []
            records_dict[rid].append(m)

        # 2. 按照病人记录划分训练/测试病人组 (Inter-patient Split)
        unique_records = list(records_dict.keys())
        random.shuffle(unique_records)

        split_idx = int(len(unique_records) * self.train_ratio)
        train_records = unique_records[:split_idx]
        test_records = unique_records[split_idx:]

        # 3. 分别构建训练和测试的候选池
        train_pool = [m for rid in train_records for m in records_dict[rid]]
        test_pool = [m for rid in test_records for m in records_dict[rid]]

        # 4. 精确计算目标样本数并抽样
        n_train_target = int(self.target_sample_count * self.train_ratio)
        n_test_target = self.target_sample_count - n_train_target

        if len(train_pool) < n_train_target or len(test_pool) < n_test_target:
            logging.warning(
                f"Train pool {len(train_pool)} < target {n_train_target} or Test pool {len(test_pool)} < target {n_test_target}. Using all available."
            )

        sampled_train_meta = random.sample(train_pool, n_train_target)
        sampled_test_meta = random.sample(test_pool, n_test_target)

        sampled_meta = sampled_train_meta + sampled_test_meta
        train_indices = list(range(0, n_train_target))
        test_indices = list(range(n_train_target, n_train_target + n_test_target))

        # 5. 读取实际信号数据
        clean_list = []
        record_cache = {}

        for meta in sampled_meta:
            rid = meta["record_id"]
            if rid not in record_cache:
                rec = wfdb.rdrecord(os.path.join(self.mitdb_dir, rid))
                record_cache[rid] = rec.p_signal.astype(np.float32)

            seg = record_cache[rid][meta["start"] : meta["end"], :]
            clean_list.append(seg)

        clean_all = np.array(clean_list, dtype=np.float32)  # (Target_N, 1024, 2)
        np.save(os.path.join(output_dir, "clean_all.npy"), clean_all)
        logging.info(f"Saved clean_all.npy with shape {clean_all.shape}")

        # 4. 合成噪声
        noise_sources = self._load_noise_source()
        for snr in snr_levels:
            for nt, n_full_sig in noise_sources.items():
                noisy_data = np.zeros_like(clean_all)
                for i in range(len(clean_all)):
                    c_seg = clean_all[i]
                    # 随机噪声截取
                    n_start = np.random.randint(0, len(n_full_sig) - self.window_size)
                    n_seg = n_full_sig[n_start : n_start + self.window_size, :]

                    # SNR 缩放
                    p_c = np.mean(c_seg**2)
                    p_n = np.mean(n_seg**2)
                    scale = (
                        np.sqrt(p_c / (p_n * (10 ** (snr / 10.0)))) if p_n > 0 else 0
                    )
                    noisy_data[i] = c_seg + n_seg * scale

                fname = f"noisy_{nt}_snr_{snr}.npy"
                np.save(os.path.join(output_dir, fname), noisy_data)
                logging.info(f"Saved: {fname}")

            noisy_emb_data = np.zeros_like(clean_all)
            for i in range(len(clean_all)):
                c_seg = clean_all[i]

                # 1. 分别从三种噪声源随机截取相同长度的片段
                mixed_noise_sum = np.zeros((self.window_size, 2), dtype=np.float32)
                for nt in ["bw", "em", "ma"]:
                    n_src = noise_sources[nt]
                    n_start = np.random.randint(0, len(n_src) - self.window_size)
                    mixed_noise_sum += n_src[n_start : n_start + self.window_size, :]

                # 2. 先平均 (取三者的算术平均)
                avg_noise = mixed_noise_sum / 3.0

                # 3. 针对“平均后的噪声”计算缩放因子
                p_c = np.mean(c_seg**2)
                p_n_avg = np.mean(avg_noise**2)

                scale_emb = (
                    np.sqrt(p_c / (p_n_avg * (10 ** (snr / 10.0))))
                    if p_n_avg > 0
                    else 0
                )

                # 4. 合成：干净信号 + 缩放后的平均噪声
                noisy_emb_data[i] = c_seg + avg_noise * scale_emb

            fname_emb = f"noisy_emb_snr_{snr}.npy"
            np.save(os.path.join(output_dir, fname_emb), noisy_emb_data)
            logging.info(f"Saved: {fname_emb} (Mixed Noise)")

        # 5. 保存索引和配置
        meta_info = {
            "config": {
                "window_size": self.window_size,
                "target_sample_count": len(sampled_meta),
                "train_ratio": self.train_ratio,
                "channels": 2,
            },
            "split": {
                "train_records": train_records,
                "test_records": test_records,
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
            "snr_levels": snr_levels,
        }
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta_info, f, indent=4)

        logging.info(
            f"Final Count: Train={len(train_indices)}, Test={len(test_indices)}"
        )


if __name__ == "__main__":
    processor = ECGDataProcessor(
        mitdb_dir="./mit-bih-arrhythmia-database",
        nstdb_dir="./mit-bih-noise-stress-test-database",
        window_size=256,
        target_sample_count=10000,
        train_ratio=0.8,
    )

    processor.generate(output_dir="./data_split", snr_levels=[-4, -2, 0, 2, 4])
