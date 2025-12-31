import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.titlesize"] = 20

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DANCER
from datasets import ECGDataset


model = DANCER()

ckpt = torch.load("./checkpoints/best_DANCER_emb_snr_-4.pth", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()


# noisy_input shape: [1, Channels, Length]
noise_type = "emb"
snr_db = -4

dataset = ECGDataset(
    split="test",
    noise_type=noise_type,
    snr_db=snr_db,
    split_dir="./data_split",
)

# idx = np.random.randint(0, len(dataset) - 1)
# print(idx)
idx = 6
# idx = 1439
noisy, clean = dataset[idx]
noisy_input = torch.tensor(noisy).unsqueeze(0).float()


with torch.no_grad():
    _ = model(noisy_input)


cache = model.encoder[0].conv[1].atnc.vis_cache


feat_in = cache["input_abs"][0].numpy()
feat_thresh = cache["threshold"][0].numpy()  # [C, 1]
feat_out = cache["output_abs"][0].numpy()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))


vmax = np.max(feat_in)


cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])


sns.heatmap(feat_in, ax=ax1, cmap="viridis", vmin=0, vmax=vmax, cbar=False)
ax1.set_title(
    "Input Feature Map (Before ATNC): Dense & Noisy", fontweight="bold", fontsize=20
)
ax1.set_ylabel("Channels", fontsize=18)
ax1.set_xticks([])
ax1.tick_params(axis="y", labelsize=16)


sns.heatmap(
    feat_out,
    ax=ax2,
    cmap="viridis",
    vmin=0,
    vmax=vmax,
    cbar=True,
    cbar_ax=cbar_ax,
    cbar_kws={"label": "Magnitude"},
)
ax2.set_title(
    "Output Feature Map (After ATNC): Sparse & Purified", fontweight="bold", fontsize=20
)
ax2.set_ylabel("Channels", fontsize=18)
ax2.set_xlabel("Time Steps", fontsize=18)

ax2.tick_params(axis="both", labelsize=16)

cbar = ax2.collections[0].colorbar
cbar.ax.yaxis.label.set_size(18)
cbar.ax.tick_params(labelsize=16)

plt.subplots_adjust(right=0.9)


plt.savefig("Figure7_Feature_Sparsity.svg", format="svg", bbox_inches="tight")
plt.show()

ch_idx = 5
time = np.arange(feat_in.shape[1])

fig, ax = plt.subplots(figsize=(8, 4))


ax.plot(
    time,
    feat_in[ch_idx],
    color="lightgray",
    label="Magnitude (|x|)",
    linewidth=2,
)


thresh_val = feat_thresh[ch_idx, 0]
ax.axhline(
    thresh_val,
    color="red",
    linestyle="--",
    label="Threshold (τ)",
    linewidth=1.5,
)

ax.fill_between(time, 0, thresh_val, color="red", alpha=0.1)


ax.plot(
    time,
    feat_out[ch_idx],
    color="#1f77b4",
    label="Magnitude (ReLU(|x|-τ))",
    linewidth=2.5,
)

ax.set_xlabel("Time Steps", fontsize=18)
ax.set_ylabel("Feature Magnitude", fontsize=18)
ax.legend(loc="upper right", frameon=True, edgecolor="black", fontsize=16, framealpha=1)
ax.set_title(
    f"Channel {ch_idx} Mechanism Visualization", fontweight="bold", fontsize=20
)
ax.tick_params(axis="both", labelsize=16)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("Figure_Mechanism_Curve.svg", format="svg", bbox_inches="tight")
plt.show()
