# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns


# def plot_heatmap(
#     matrix,
#     title="Matrix Heatmap",
#     xlabel="Columns",
#     ylabel="Rows",
#     cmap="hot",
#     annot=False,
#     fmt=".2f",
#     figsize=(6, 6),
#     save_path=None,
# ):
#     """
#     绘制矩阵热力图

#     参数:
#     - matrix: 二维数组或矩阵
#     - title: 图标题
#     - xlabel: x轴标签
#     - ylabel: y轴标签
#     - cmap: 颜色映射 (viridis, plasma, hot, coolwarm, RdYlBu, Blues, 等)
#     - annot: 是否在格子中显示数值
#     - fmt: 数值格式
#     - figsize: 图像大小
#     - save_path: 保存路径
#     """

#     plt.figure(figsize=figsize)

#     # 使用seaborn绘制热力图
#     ax = sns.heatmap(
#         matrix,
#         cmap=cmap,
#         annot=annot,
#         fmt=fmt,
#         linewidths=0.5,
#         linecolor="white",
#         cbar_kws={"shrink": 0.8},
#         square=True,
#     )

#     plt.title(title, fontsize=16, fontweight="bold", pad=20)
#     plt.xlabel(xlabel, fontsize=12, fontweight="bold")
#     plt.ylabel(ylabel, fontsize=12, fontweight="bold")

#     # 调整布局
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=400, bbox_inches="tight")
#         print(f"热力图已保存至: {save_path}")

#     plt.show()


# if __name__ == "__main__":
#     # 示例矩阵
#     data = np.random.rand(6, 6)

#     # 绘制热力图
#     plot_heatmap(data, save_path="heatmap.png")
