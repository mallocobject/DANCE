# DANCER: 基于双重自适应机制的ECG信号去噪网络

## 网络概述

**DANCER** 是一种基于 U-Net 结构的心电信号（ECG）去噪网络，内部集成了核心创新模块 **DANCE (Dual Adaptive Noise-Compression and Core-Excitation)**
该模块通过 **通道自适应压缩（CAC）** 与 **通道-空间激励（CSE）** 的协同机制，实现对噪声的自适应抑制与关键特征的增强 
模型采用端到端训练方式，从含噪 ECG 信号直接学习到纯净信号，在保证高去噪性能的同时兼顾参数效率与可解释性

> **说明：** DANCE 为核心创新模块，DANCER 表示完整去噪网络，用于与其他模型进行性能对比

---

## 💡 核心设计

### 🔥 双重自适应去噪 (DANCE)

- **通道自适应压缩 (CAC)：**  
  基于全局统计信息与可学习阈值机制，自适应压缩噪声成分，抑制特征层噪声干扰
  模块利用绝对值映射与通道注意力机制，为每个通道生成动态阈值，实现特征的软收缩与非线性恢复

- **通道-空间激励 (CSE)：**  
  通过一维卷积捕获局部依赖关系，在通道与空间维度上分配权重，从而强化关键特征响应并抑制冗余成分

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

<!-- <table>
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

--- -->

## 🎨 去噪效果可视化

![Denoising Comparison](./ecg_denoising_comparison.png)

*图：不同去噪方法在 -4 dB 噪声水平下对双通道 ECG 信号的去噪效果对比。*

---

## 🏆 性能总结

实验结果表明，**DANCER 网络** 在多种噪声水平下均表现出较高的信噪比提升与较低的重建误差
模型通过 **双重自适应机制（CAC + CSE）** 有效抑制噪声干扰，同时保持 ECG 信号的关键生理特征

---

**by Dan Liu, Tianhai Xie @IIP-2025**