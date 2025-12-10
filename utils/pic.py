import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ==========================================
# 1. 信号生成函数 (Signal Generation)
# ==========================================
def generate_ecg_pulse(t):
    # 模拟一个简单的 QRS 波群 (Sinc 函数 + 高斯)
    return (
        3.0 * np.exp(-0.5 * ((t - 50) / 4) ** 2)
        - 1.0 * np.exp(-0.5 * ((t - 45) / 5) ** 2)
        - 1.0 * np.exp(-0.5 * ((t - 55) / 5) ** 2)
    )


def generate_noise(t, level=0.5):
    np.random.seed(42)  # 固定随机种子，保证每次图一样
    return np.random.normal(0, level, len(t))


# 准备数据
t = np.linspace(0, 100, 300)
clean_sig = generate_ecg_pulse(t)
noise = generate_noise(t, level=0.4)  # 高噪声
noisy_sig = clean_sig + noise

# 运算结果模拟
# Attention: 仅仅是线性缩放 (x 0.5)
attn_output = noisy_sig * 0.5

# Soft Thresholding: 模拟软阈值去噪 (直接用干净信号 + 极少量残留模拟效果)
# 这里的处理是为了视觉效果，数学上是 sign(x)*max(0, |x|-tau)
# 我们直接画一个干净的信号来代表理想的去噪效果
thresh_output = clean_sig * 0.9  # 稍微调整幅度

# ==========================================
# 2. 绘图设置 (Plotting Configuration)
# ==========================================
fig = plt.figure(figsize=(14, 8), dpi=150)
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# 定义颜色
COLOR_SIG = "#2ca02c"  # 绿色 (有效信号)
COLOR_NOISE = "#7f7f7f"  # 灰色 (背景噪声)
COLOR_INPUT = "#d62728"  # 红色 (输入波形)
COLOR_OUT_ATTN = "#1f77b4"  # 蓝色 (Attention输出)
COLOR_OUT_OURS = "#1f77b4"  # 蓝色 (Ours输出)
BG_ATTN = "#fff9c4"  # 浅黄背景 (Attention区域)
BG_OURS = "#e0f7fa"  # 浅青背景 (Ours区域)

# 创建两个大的背景区域来区分模块
# 使用 figure coordinate system
rect_attn = patches.Rectangle(
    (0.02, 0.52),
    0.96,
    0.44,
    transform=fig.transFigure,
    color=BG_ATTN,
    alpha=0.5,
    zorder=-1,
    ec="none",
    lw=0,
)
rect_ours = patches.Rectangle(
    (0.02, 0.04),
    0.96,
    0.44,
    transform=fig.transFigure,
    color=BG_OURS,
    alpha=0.5,
    zorder=-1,
    ec="none",
    lw=0,
)
fig.patches.extend([rect_attn, rect_ours])

# ==========================================
# 3. 绘制第一行：Attention Mechanism
# ==========================================

# --- 1.1 Input ---
ax1 = plt.subplot2grid((2, 5), (0, 0))
ax1.plot(t, noisy_sig, color=COLOR_INPUT, lw=1)
ax1.set_title("Input (Noisy)", fontsize=10, fontweight="bold")
ax1.axis("off")

# --- 1.2 Schematic (Bar Charts) ---
ax_mid_top = plt.subplot2grid((2, 5), (0, 1), colspan=3)
ax_mid_top.set_xlim(0, 10)
ax_mid_top.set_ylim(0, 6)
ax_mid_top.axis("off")
ax_mid_top.set_title(
    "Standard Attention (Homogeneous Scaling)", fontsize=12, fontweight="bold", pad=20
)

# 左侧柱状图 (Feature before)
# 模拟4个通道，每个通道由 噪声(灰) + 信号(绿) 组成
bars_x = [1, 2, 3, 4]
noise_h = [1.5, 1.2, 1.8, 1.4]  # 噪声基底
sig_h = [0.0, 4.0, 0.5, 0.0]  # 信号成分 (只有第2个有强信号)

for i, x in enumerate(bars_x):
    # 画噪声部分 (底)
    ax_mid_top.bar(
        x, noise_h[i], color=COLOR_NOISE, width=0.6, label="Noise" if i == 0 else ""
    )
    # 画信号部分 (顶)
    ax_mid_top.bar(
        x,
        sig_h[i],
        bottom=noise_h[i],
        color=COLOR_SIG,
        width=0.6,
        label="Signal" if i == 0 else "",
    )

# 箭头 + 操作符
ax_mid_top.annotate(
    "", xy=(6, 2.5), xytext=(4.5, 2.5), arrowprops=dict(arrowstyle="->", lw=2)
)
ax_mid_top.text(5.25, 2.8, r"$\times \omega$ (0.5)", fontsize=14, ha="center")

# 右侧柱状图 (Feature after) - 整体变小，比例不变
scale = 0.5
bars_x_out = [7, 8, 9, 10]
for i, x in enumerate(bars_x_out):
    # 噪声变小了，但还在！
    ax_mid_top.bar(x, noise_h[i] * scale, color=COLOR_NOISE, width=0.6, alpha=0.8)
    ax_mid_top.bar(
        x,
        sig_h[i] * scale,
        bottom=noise_h[i] * scale,
        color=COLOR_SIG,
        width=0.6,
        alpha=0.8,
    )

# --- 1.3 Output ---
ax3 = plt.subplot2grid((2, 5), (0, 4))
ax3.plot(t, attn_output, color=COLOR_OUT_ATTN, lw=1)  # 输出依然有噪声
ax3.set_title("Output (Still Noisy)\nConstant SNR", fontsize=10, fontweight="bold")
ax3.axis("off")


# ==========================================
# 4. 绘制第二行：Soft Thresholding (Ours)
# ==========================================

# --- 2.1 Input ---
ax4 = plt.subplot2grid((2, 5), (1, 0))
ax4.plot(t, noisy_sig, color=COLOR_INPUT, lw=1)
ax4.axis("off")

# --- 2.2 Schematic (Bar Charts) ---
ax_mid_bot = plt.subplot2grid((2, 5), (1, 1), colspan=3)
ax_mid_bot.set_xlim(0, 10)
ax_mid_bot.set_ylim(0, 6)
ax_mid_bot.axis("off")
ax_mid_bot.set_title(
    "Proposed DANCE (Feature Purification)", fontsize=12, fontweight="bold", pad=20
)

# 左侧柱状图 (同上)
for i, x in enumerate(bars_x):
    ax_mid_bot.bar(x, noise_h[i], color=COLOR_NOISE, width=0.6)
    ax_mid_bot.bar(x, sig_h[i], bottom=noise_h[i], color=COLOR_SIG, width=0.6)

# 箭头 + 操作符
ax_mid_bot.annotate(
    "", xy=(6, 2.5), xytext=(4.5, 2.5), arrowprops=dict(arrowstyle="->", lw=2)
)
ax_mid_bot.text(5.25, 2.8, r"$- \tau$ (Threshold)", fontsize=14, ha="center")

# 右侧柱状图 (Feature after) - 噪声被切除！
thresh = 1.8  # 假设阈值把噪声切干净了
for i, x in enumerate(bars_x_out):
    total_h = noise_h[i] + sig_h[i]
    new_h = max(0, total_h - thresh)  # 简单的硬阈值模拟视觉

    # 只有信号留下来 (画成绿色)
    # 如果 new_h > 0, 我们认为它是纯净信号 (虽然数学上软阈值是减法，但这里为了视觉强调"净化")
    if new_h > 0:
        ax_mid_bot.bar(x, new_h, color=COLOR_SIG, width=0.6)
    else:
        # 画一个极小的灰线表示0
        ax_mid_bot.plot([x - 0.3, x + 0.3], [0, 0], color="black", lw=1)

# --- 2.3 Output ---
ax6 = plt.subplot2grid((2, 5), (1, 4))
ax6.plot(t, thresh_output, color=COLOR_OUT_OURS, lw=1.2)  # 输出干净
ax6.set_title("Output (Clean)\nImproved SNR", fontsize=10, fontweight="bold")
ax6.axis("off")


# ==========================================
# 5. 添加图例 (Legend)
# ==========================================
# 在图的中间添加一个公共图例
legend_elements = [
    patches.Patch(facecolor=COLOR_SIG, label="Signal Info"),
    patches.Patch(facecolor=COLOR_NOISE, label="Background Noise"),
]
fig.legend(
    handles=legend_elements,
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    ncol=2,
    frameon=True,
    fontsize=12,
)

plt.tight_layout()
# 保存图片
plt.savefig("motivation_figure.png", dpi=300, bbox_inches="tight")
plt.show()
