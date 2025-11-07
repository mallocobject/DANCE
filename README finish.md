# AGS-UNet: ECG信号去噪网络

## 网络概述

AGS-UNet (Attention Gated Shrinkage UNet) 是一种专为ECG信号去噪任务设计的深度学习网络. 该网络创新性地融合了双重注意力机制与轻量化设计, 在保证高效推理速度的同时, 实现了卓越的噪声抑制性能

## 💡 核心特点

### 🔥 双重注意力协同去噪

- **特征级自适应去噪**：编码器集成的收缩模块通过可学习阈值机制, 实现特征层面的智能噪声过滤
- **空间感知增强**：跳跃连接中的注意力门控单元, 对特征映射进行空间重要性重加权

### 🎯 渐进式特征融合

- **跨尺度特征选择**：注意力门控机制智能筛选多尺度特征, 实现精准的特征传递
- **语义一致性保持**：确保编码器与解码器间的特征语义对齐, 促进信息流畅传递

### ⚡ 轻量化高效设计

- **优化卷积结构**：采用小尺度卷积核, 平衡感受野与计算效率
- **精简参数量**：在保持性能的前提下, 降低模型复杂度和计算开销

### 🚀 端到端优化

- **直接信号映射**：构建从含噪信号 (ECG双导联信号) 到纯净信号的端到端学习框架

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
      <td>DACNN</td>
      <td>8.0467 ± </td>
      <td>8.8087</td>
      <td>9.4517</td>
      <td>10.7303 ± </td>
      <td>11.4378</td>
      <td>0.1838 ± 0.0015</td>
      <td></td>
      <td></td>
      <td>0.1305 ± 0.0011</td>
      <td></td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>8.1675</td>
      <td>8.9059</td>
      <td>9.5592</td>
      <td>10.6778</td>
      <td>11.2051</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr style="background-color: #ecf3ecff;">
      <td><strong>CIADNet (ours)</strong></td>
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
      <td><strong>CIADNet (ours)</strong></td>
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
      <td><strong>CIADNet (ours)</strong></td>
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
      <td><strong>CIADNet (ours)</strong></td>
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
      <td>7.8181</td>
      <td>8.5531</td>
      <td>9.3777</td>
      <td>10.1528</td>
      <td>11.0423.</td>
      <td>0.</td>
      <td>0.</td>
      <td>0.</td>
      <td>0.</td>
      <td>0.</td>
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