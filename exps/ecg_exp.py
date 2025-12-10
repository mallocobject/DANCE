import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
import numpy as np
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_metrics
from datasets import ECGDataset
from models import *


class ECGDenoisingExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.model_dict = {
            "U-Net": UNet,
            "DANCER": DANCER,
            "ACDAE": ACDAE,
            "DACNN": DACNN,
        }

        self.checkpoint = os.path.join(
            self.args.checkpoint_dir,
            f"best_{self.args.model}_{self.args.noise_type}_snr_{self.args.snr_db}.pth",
        )

        self.results_file = os.path.join(
            "./results",
            f"results_{self.args.model}_{self.args.noise_type}_snr_{self.args.snr_db}.txt",
        )

        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Unknown model type: {self.args.model}")
        model = self.model_dict[self.args.model]()
        return model

    def _get_dataloader(self, split: str):
        dataset = ECGDataset(
            split=split,
            noise_type=self.args.noise_type,
            snr_db=self.args.snr_db,
            split_dir=self.args.split_dir,
        )
        self.mean, self.std = dataset.get_stats()
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.args.seed),
        )
        return dataloader

    def _select_criterion(self):
        return nn.MSELoss()

    def _select_optimizer(self, model: nn.Module):
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        return optimizer

    def _select_scheduler(self, optimizer: optim.Optimizer):
        def lr_lambda(epoch):
            if epoch < 50:
                return 1.0
            elif epoch < 70:
                return 0.1
            else:
                return 0.05

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.args.epochs, eta_min=1e-5
        # )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return scheduler

    def train(self):
        metrics = {}

        print(f"🚀 Starting training")
        dataloader = self._get_dataloader("train")

        model = self._build_model()

        criterion = self._select_criterion()

        optimizer = self._select_optimizer(model)
        scheduler = self._select_scheduler(optimizer)

        model = model.to(self.device)

        for epoch in range(self.args.epochs):
            model.train()
            losses = []
            for x, label in dataloader:
                x, label = x.to(self.device), label.to(self.device)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, label)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            scheduler.step()

            avg_loss = np.mean(losses)

            print(
                f"--- Epoch {epoch+1}({scheduler.get_last_lr()[0]:.5f}): Loss: {avg_loss:.4f}"
            )

            if epoch == self.args.epochs - 1:
                metrics = self.test(model=model)

        torch.save(model.state_dict(), self.checkpoint)

        print(f"SNR: {metrics['SNR']:.2f}\nRMSE: {metrics['RMSE']:.4f}")
        print("Results saved to:", self.results_file)

    def test(self, model: nn.Module = None):
        test_dataloader = self._get_dataloader("test")

        if model is None:
            model = self._build_model()
            model.load_state_dict(
                torch.load(self.checkpoint, weights_only=True, map_location="cpu")
            )
            model = model.to(self.device)

        # ====== 测试阶段 ======
        model.eval()
        metrics = {"RMSE": [], "SNR": []}

        with torch.no_grad():
            for x, label in test_dataloader:
                x, label = x.to(self.device), label.to(self.device)

                outputs = model(x)
                metrics_res = compute_metrics(outputs, label, self.mean, self.std)
                for key in metrics:
                    metrics[key].append(metrics_res[key])

        # 计算平均指标
        for key in metrics:
            metrics[key] = np.mean(metrics[key])

        return metrics
