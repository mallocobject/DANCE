import argparse

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exps import ECGDenoisingExperiment


def parse_args():
    parser = argparse.ArgumentParser(description="ECG Denoising Experiment")

    # ====== 数据路径 ======
    parser.add_argument(
        "--split_dir",
        type=str,
        default="./data_split",
        help="Path to split directory containing data splits and files",
    )

    # ====== 模型与训练参数 ======
    parser.add_argument(
        "--model",
        type=str,
        default="U-Net",
        choices=["U-Net", "DANCER", "ACDAE", "DACNN"],
        help="Model architecture to use",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--noise_type", type=str, default="emb", choices=["bw", "em", "ma", "emb"]
    )
    parser.add_argument("--snr_db", type=int, default=0, choices=[-4, -2, 0, 2, 4])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="you know 42 is the answer to life, universe and everything",
    )

    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU id to use for training/testing"
    )

    # ====== 模型保存与加载 ======
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")

    # ====== 模式选择 ======
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    return parser.parse_args()


def main():
    args = parse_args()
    exp = ECGDenoisingExperiment(args)

    if args.mode == "train":
        exp.train()
    elif args.mode == "test":
        exp.test()


if __name__ == "__main__":
    main()
