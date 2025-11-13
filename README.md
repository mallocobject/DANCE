# DANCE: Dual Adaptive Noise-Cancellation and Enhancement for ECG Signals

## 网络概述

**DANCE** 是一种轻量级、端到端的 ECG 信号去噪与特征增强模块，专为心电信号在强噪声环境下的鲁棒处理而设计。该模块通过 **通道自适应压缩（CAC）** 与 **通道-空间激励（CSE）** 的**双重自适应协同机制**，实现对噪声的精准抑制与 QRS 复合波、P/T 波等**临床关键特征的显著增强**。

模型采用全卷积+全局统计的混合架构，**无需外部参考信号**，从单通道含噪 ECG 直接学习纯净波形，在保证高去噪性能的同时，兼具 **参数效率高、可解释性强、易于嵌入移动设备** 的优势。

---

## 💡 核心设计

### 🔥 双重自适应降噪与增强 (DANCE)

DANCE 由两个串行子模块构成，分别从**全局统计压缩**与**局部时空增强**两个维度协同作用：

---

#### **1. 通道自适应压缩 (CAC) —— 自适应噪声软阈值收缩**

- **输入**：含噪 ECG 特征 `x ∈ ℝ^{B×C×L}`
- **核心机制**：
  1. 绝对值映射 `|x|` → 抑制符号扰动，聚焦能量
  2. **全局平均池化（GAP）** 提取通道级统计
  3. **轻量 MLP** 生成 **通道自适应阈值系数**
  4. 动态阈值 `τ_c = stat_c × σ(·)` → **软阈值收缩**
  5. **符号恢复 + ReLU** → 输出 **软阈值去噪特征**

> **优势**：
> - 精确复现 **传统软阈值去噪** 原理，但 **阈值可学习 + 通道自适应**
> - 有效抑制 **肌电、工频、基线漂移**
> - 自动保留 **QRS 高能量尖峰**

---

#### **2. 通道-空间激励 (CSE) —— 关键波形局部增强**

- **输入**：CAC 处理后的稀疏特征
- **核心机制**：
  1. 绝对值输入 → 聚焦能量分布
  2. 两层 1D 卷积（`kernel=7`）捕获 **±3 点局部依赖**
  3. 通道压缩 → 空间-通道联合注意力
  4. Sigmoid 激活 → 生成 `α ∈ [0,1]^{C×L}` 注意力图
  5. 逐元素加权 → **增强 QRS 峰值，抑制平坦区残余噪声**

> **优势**：
> - 显式建模 **QRS 波的局部时序模式**
> - 提升 **P 波、T 波可检测性**
> - 增强模型**对心律失常特征的敏感性**

---



## 📊 模型对比结果

<table>
  <thead>
    <tr>
      <th rowspan="3">Methods</th>
      <th colspan="5">SNR(dB)</th>
      <th colspan="5">RMSE</th>
    </tr>
    <tr>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FFT</td>
      <td>1.5032</td>
      <td>3.4867</td>
      <td>5.4740</td>
      <td>7.4686</td>
      <td>9.4679</td>
      <td>0.4009</td>
      <td>0.3190</td>
      <td>0.2539</td>
      <td>0.2019</td>
      <td>0.1604</td>
    </tr>
    <tr>
      <td>DWT</td>
      <td>1.7186</td>
      <td>3.6650</td>
      <td>5.6204</td>
      <td>7.5788</td>
      <td>9.5565</td>
      <td>0.3920</td>
      <td>0.3131</td>
      <td>0.2499</td>
      <td>0.2015</td>
      <td>0.1587</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>7.1792</td>
      <td>7.9122</td>
      <td>8.7832</td>
      <td>9.6826</td>
      <td>10.6419</td>
      <td>0.2098</td>
      <td>0.1920</td>
      <td>0.1731</td>
      <td>0.1555</td>
      <td>0.1387</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>8.0310</td>
      <td>8.8935</td>
      <td>9.7998</td>
      <td>10.7158</td>
      <td>11.5086</td>
      <td>0.1884</td>
      <td>0.1702</td>
      <td>0.1526</td>
      <td>0.1377</td>
      <td>0.1253</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>7.9996</td>
      <td>8.8147</td>
      <td>9.7136</td>
      <td>10.7407</td>
      <td>11.4073</td>
      <td>0.1910</td>
      <td>0.1734</td>
      <td>0.1560</td>
      <td>0.1385</td>
      <td>0.1281</td>
    </tr>
    <tr>
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>8.9458</strong></td>
      <td><strong>9.7525</strong></td>
      <td><strong>10.5912</strong></td>
      <td><strong>11.4206</strong></td>
      <td><strong>12.2659</strong></td>
      <td><strong>0.1743</strong></td>
      <td><strong>0.1575</strong></td>
      <td><strong>0.1424</strong></td>
      <td><strong>0.1290</strong></td>
      <td><strong>0.1168</strong></td>
    </tr>
  </tbody>
</table>

> 表：混合噪声 (emb) 下的不同方法去噪性能对比

---

## 🔬 DANCE 模块消融实验

<table>
  <thead>
    <tr>
      <th rowspan="3">Methods</th>
      <th colspan="5">SNR(dB)</th>
      <th colspan="5">RMSE</th>
    </tr>
    <tr>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline (U-Net)</td>
      <td>7.1792</td>
      <td>7.9122</td>
      <td>8.7832</td>
      <td>9.6826</td>
      <td>10.6419</td>
      <td>0.2098</td>
      <td>0.1920</td>
      <td>0.1731</td>
      <td>0.1555</td>
      <td>0.1387</td>
    </tr>
    <tr>
      <td>+ Channel Shrink</td>
      <td>8.6951</td>
      <td>9.5403</td>
      <td>10.4438</td>
      <td>11.2464</td>
      <td>12.0682</td>
      <td>0.1781</td>
      <td>0.1605</td>
      <td>0.1439</td>
      <td>0.1307</td>
      <td>0.1185</td>
    </tr>
    <tr>
      <td><strong>+ Channel & Spatial Shrink</strong></td>
      <td><strong>8.9458</strong></td>
      <td><strong>9.7525</strong></td>
      <td><strong>10.5912</strong></td>
      <td><strong>11.4206</strong></td>
      <td><strong>12.2659</strong></td>
      <td><strong>0.1743</strong></td>
      <td><strong>0.1575</strong></td>
      <td><strong>0.1424</strong></td>
      <td><strong>0.1290</strong></td>
      <td><strong>0.1168</strong></td>
    </tr>
  </tbody>
</table>

> 表：DANCE 模块逐步增强带来的性能提升

---

<table>
  <thead>
    <tr>
      <th rowspan="3">Methods</th>
      <th colspan="5">SNR(dB)</th>
      <th colspan="5">RMSE</th>
    </tr>
    <tr>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FFT</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>DWT</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>9.4762</td>
      <td>10.0018</td>
      <td>10.7111</td>
      <td>11.5845</td>
      <td>12.4903</td>
      <td>0.1635</td>
      <td>0.1540</td>
      <td>0.1423</td>
      <td>0.1296</td>
      <td>0.1174</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>10.3958</td>
      <td>11.1785</td>
      <td>11.8643</td>
      <td>12.7837</td>
      <td>13.6798</td>
      <td>0.1484</td>
      <td>0.1363</td>
      <td>0.1260</td>
      <td>0.1150</td>
      <td>0.1046</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>10.4238</td>
      <td>11.0144</td>
      <td>11.8713</td>
      <td>12.6993</td>
      <td>13.5933</td>
      <td>0.1478</td>
      <td>0.1385</td>
      <td>0.1267</td>
      <td>0.1168</td>
      <td>0.1062</td>
    </tr>
    <tr style="background-color: #ecf3ecff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>11.4086</strong></td>
      <td><strong>12.1023</strong></td>
      <td><strong>12.9022</strong></td>
      <td><strong>13.7191</strong></td>
      <td><strong>14.4682</strong></td>
      <td><strong>0.1351</strong></td>
      <td><strong>0.1245</strong></td>
      <td><strong>0.1143</strong></td>
      <td><strong>0.1053</strong></td>
      <td><strong>0.0973</strong></td>
    </tr>
  </tbody>
</table>
基线漂移 (bw) 下的去噪性能对比 (10 runs)

---

<table>
  <thead>
    <tr>
      <th rowspan="3">Methods</th>
      <th colspan="5">SNR(dB)</th>
      <th colspan="5">RMSE</th>
    </tr>
    <tr>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FFT</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>DWT</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>7.0895</td>
      <td>7.7027</td>
      <td>8.3145</td>
      <td>9.1401</td>
      <td>10.0686</td>
      <td>0.2157</td>
      <td>0.2004</td>
      <td>0.1860</td>
      <td>0.1679</td>
      <td>0.1493</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>8.2003</td>
      <td>8.8572</td>
      <td>9.5993</td>
      <td>10.4399</td>
      <td>11.1198</td>
      <td>0.1912</td>
      <td>0.1762</td>
      <td>0.1609</td>
      <td>0.1456</td>
      <td>0.1334</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>8.2452</td>
      <td>8.9140</td>
      <td>9.6430</td>
      <td>10.2493</td>
      <td>11.1859</td>
      <td>0.1899</td>
      <td>0.1754</td>
      <td>0.1602</td>
      <td>0.1490</td>
      <td>0.1325</td>
    </tr>
    <tr style="background-color: #ecf3ecff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
    </tr>
  </tbody>
</table>
肌肉伪迹 (ma) 下的去噪性能对比 (10 runs)

---

<table>
  <thead>
    <tr>
      <th rowspan="3">Methods</th>
      <th colspan="5">SNR(dB)</th>
      <th colspan="5">RMSE</th>
    </tr>
    <tr>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
      <th>-4dB</th>
      <th>-2dB</th>
      <th>0dB</th>
      <th>2dB</th>
      <th>4dB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FFT</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>DWT</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr style="background-color: #ecf3ecff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
    </tr>
  </tbody>
</table>
电机移动伪迹 (em) 下的去噪性能对比 (10 runs)

---

## 🎨 去噪效果可视化

![Denoising Comparison](./ecg_denoising_comparison.png)

*图：不同去噪方法在 -4 dB 噪声水平下对双通道 ECG 信号的去噪效果对比。*

---

## 🏆 性能总结

实验结果表明，**DANCER 网络** 在多种噪声水平下均表现出较高的信噪比提升与较低的重建误差
模型通过 **双重自适应机制（CAC + CSE）** 有效抑制噪声干扰，同时保持 ECG 信号的关键生理特征

---

**by Dan Liu, Tianhai Xie @IIP-2025**