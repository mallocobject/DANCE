import argparse

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exps import EEGDenoisingExperiment


def parse_args():
    parser = argparse.ArgumentParser(description="ECG Denoising Experiment")

    parser.add_argument(
        "--split_dir",
        type=str,
        default="./data_split",
        help="Path to split directory containing data splits and files",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="U-Net",
        choices=["U-Net", "DANCE", "ACDAE", "DACNN", "RALENet"],
        help="Model architecture to use",
    )

    parser.add_argument(
        "--plugin_type",
        type=str,
        default="DANCE",
        choices=["DANCE", "DANCE_inv", "DRSN", "AREM", "ATNC", "SE", "CBAM", "ECA"],
        help="Plugin type to use",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--noise_type", type=str, default="EMG", choices=["EMG", "EOG", "EOGEMG"]
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

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    return parser.parse_args()


def main():
    args = parse_args()
    exp = EEGDenoisingExperiment(args)

    if args.mode == "train":
        exp.train()
    elif args.mode == "test":
        exp.test()


if __name__ == "__main__":
    main()
