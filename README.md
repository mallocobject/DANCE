# DANCER: ECG信号去噪网络

## 网络概述

**DANCER** 是一种基于UNet结构的心电信号(ECG)去噪网络, 内部集成了核心创新模块 **DANCE (Dual Adaptive Noise-Compression and Core-Excitation)**。  
该模块通过通道自适应压缩(CAC)与通道-空间激励(CSE)的协同机制, 实现对噪声的自适应抑制与关键信号特征的增强。  
模型以端到端方式从含噪ECG信号直接学习到纯净信号, 在保证高去噪性能的同时兼顾轻量化与可解释性。

> **说明:** DANCE 为核心创新模块, DANCER 表示完整去噪网络, 用于与其他模型进行对比实验。

---

## 💡 核心特点

### 🔥 双重自适应去噪 (DANCE)

- **通道自适应压缩 (CAC):**  
  通过全局统计信息与可学习阈值机制, 自适应压缩噪声成分, 有效抑制特征层噪声干扰。  
  该模块基于绝对值映射与通道注意力, 实现特征的动态阈值收缩与非线性恢复。  

- **通道-空间激励 (CSE):**  
  通过一维卷积建模局部依赖关系, 在通道与空间维度上动态分配权重, 强化关键特征响应并抑制冗余信息。  

---

### 🎯 渐进式特征融合

- **跨尺度特征选择:**  
  在解码阶段, 注意力门控机制选择性融合多尺度特征, 提高重建精度与上下文一致性。  
- **语义一致性保持:**  
  通过残差与跳跃连接保持编码器与解码器间的语义一致性, 促进高层与低层信息的协同流动。

---

### ⚡ 轻量化高效设计

- **结构优化:**  
  模块采用小卷积核结构与通道压缩设计, 在保持感受野的同时降低计算负担。  
- **参数量精简:**  
  使用分层降维与共享权重策略, 显著减少参数数量与显存占用, 提升运行效率。

---

### 🚀 端到端优化

- **直接信号映射:**  
  网络采用端到端训练策略, 从原始含噪信号直接学习映射到纯净信号, 提升整体收敛速度与泛化性能。  
- **多层协同监督:**  
  结合重建误差与特征相似度损失, 提升网络的稳定性与去噪鲁棒性。

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
      <td>6.9607 ± 0.0358</td>
      <td>7.8065 ± 0.0387</td>
      <td>8.6614 ± 0.0409</td>
      <td>9.6005 ± 0.0791</td>
      <td>10.5637 ± 0.0421</td>
      <td>0.2151 ± 0.0010</td>
      <td>0.1945 ± 0.0008</td>
      <td>0.1759 ± 0.0009</td>
      <td>0.1573 ± 0.0014</td>
      <td>0.1402 ± 0.0008</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>7.8718 ± 0.1154</td>
      <td>8.6885 ± 0.0899</td>
      <td>9.6326 ± 0.1123</td>
      <td>10.5301 ± 0.1150</td>
      <td>11.4497 ± 0.1791</td>
      <td>0.1921 ± 0.0022</td>
      <td>0.1733 ± 0.0016</td>
      <td>0.1553 ± 0.0018</td>
      <td>0.1400 ± 0.0015</td>
      <td>0.1260 ± 0.0027</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>8.0482 ± 0.1703</td>
      <td>8.8348 ± 0.1252</td>
      <td>9.6317 ± 0.1241</td>
      <td>10.5137 ± 0.1573</td>
      <td>11.4005 ± 0.1607</td>
      <td>0.1899 ± 0.0032</td>
      <td>0.1728 ± 0.0028</td>
      <td>0.1575 ± 0.0026</td>
      <td>0.1416 ± 0.0026</td>
      <td>0.1279 ± 0.0025</td>
    </tr>
    <tr style="background-color: #ecf3ecff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>8.2786 ± 0.</strong></td>
      <td><strong>9.0644 ± 0.</strong></td>
      <td><strong>10.0717 ± 0.</strong></td>
      <td><strong>10.7572 ± 0.</strong></td>
      <td><strong>11.6682 ± 0.</strong></td>
      <td><strong>0.1952 ± 0.</strong></td>
      <td><strong>0.1767 ± 0.</strong></td>
      <td><strong>0.1606 ± 0.</strong></td>
      <td><strong>0.1482 ± 0.</strong></td>
      <td><strong>0.1318 ± 0.</strong></td>
    </tr>
  </tbody>
</table>
混合噪声 (emb) 下的去噪性能对比

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

---

## 🔬 CIAD消融实验分析

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
      <td>6.9733 ± 0.0676</td>
      <td>7.7794 ± 0.0560</td>
      <td>8.6639 ± 0.0724</td>
      <td>9.6611 ± 0.0417</td>
      <td>10.5980 ± 0.0597</td>
      <td>0.1839 ± 0.0015</td>
      <td>0.1633 ± 0.0010</td>
      <td>0.1450 ± 0.0011</td>
      <td>0.1278 ± 0.0006</td>
      <td>0.1117 ± 0.0008</td>
    </tr>
    <tr>
      <td>+ Channel Shrink</td>
      <td>8.3010 ± 0.0560</td>
      <td>9.1325 ± 0.0451</td>
      <td>9.9971 ± 0.0898</td>
      <td>10.8336 ± 0.0753</td>
      <td>11.6638 ± 0.0435</td>
      <td>0.1845 ± 0.0012</td>
      <td>0.1669 ± 0.0010</td>
      <td>0.1505 ± 0.0015</td>
      <td>0.1365 ± 0.0011</td>
      <td>0.1235 ± 0.0006</td>
    </tr>
    <tr>
      <td>+ Spatial Shrink</td>
      <td>7.1127</td>
      <td>8.0044</td>
      <td>8.8369</td>
      <td>9.8145</td>
      <td>10.7461</td>
      <td>0.</td>
      <td>0.</td>
      <td>0.</td>
      <td>0.</td>
      <td>0.</td>
    </tr>
    <tr>
      <td><strong>+ Channel & Spatial Shrink</strong></td>
      <td><strong>7.9458</strong></td>
      <td><strong>8.7579</strong></td>
      <td><strong>9.5577</strong></td>
      <td><strong>10.3215</strong></td>
      <td><strong>11.1991</strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
      <td><strong></strong></td>
    </tr>
  </tbody>
</table>

---

## 🎨 去噪效果可视化

### 多方法去噪对比
![Denoising Comparison](./output.png)

*不同去噪方法在-4dB噪声级下对双通道ECG信号的去噪效果对比*

---

## 🏆 性能总结

AGS-UNet在多个噪声水平下均表现出色, 特别是在低信噪比条件下(-4dB至2dB)的SNR和RMSE指标均优于对比方法. 该网络通过双重注意力机制实现了对ECG信号中噪声的有效抑制, 同时保持了重要的生理特征信息

**by Dan Liu, Tianhai Xie @IIP-2025**