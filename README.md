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
      <td>CBAM-UNet</td>
      <td>9.0012</td>
      <td>9.8849</td>
      <td>10.9736</td>
      <td>11.8059</td>
      <td>12.8272</td>
      <td>0.1683</td>
      <td>0.1511</td>
      <td>0.1334</td>
      <td>0.1204</td>
      <td>0.1070</td>
    </tr>
    <tr>
      <td>ECA-UNet (ACDAE)</td>
      <td>8.1096</td>
      <td>9.1517</td>
      <td>10.1912</td>
      <td>11.2488</td>
      <td>12.3446</td>
      <td>0.1860</td>
      <td>0.1648</td>
      <td>0.1458</td>
      <td>0.1281</td>
      <td>0.1132</td>
    </tr>
    <tr>
      <td>SE-UNet</td>
      <td>9.1097</td>
      <td>9.8424</td>
      <td>10.8138</td>
      <td>11.8796</td>
      <td>12.8323</td>
      <td>0.1662</td>
      <td>0.1521</td>
      <td>0.1354</td>
      <td>0.1187</td>
      <td>0.1069</td>
    </tr>
    <tr>
      <td>CS-UNet</td>
      <td>9.0712</td>
      <td>9.6873</td>
      <td>10.6865</td>
      <td>11.6704</td>
      <td>12.7099</td>
      <td>0.1678</td>
      <td>0.1543</td>
      <td>0.1375</td>
      <td>0.1224</td>
      <td>0.1078</td>
    </tr>
    <tr style="background-color: #ecf3ecff;">
      <td><strong>CIAD-UNet (ours)</strong></td>
      <td><strong>9.1616</strong></td>
      <td><strong>10.0953</strong></td>
      <td><strong>10.9872</strong></td>
      <td><strong>11.9503</strong></td>
      <td><strong>12.8933</strong></td>
      <td><strong>0.1647</strong></td>
      <td><strong>0.1480</strong></td>
      <td><strong>0.1331</strong></td>
      <td><strong>0.1187</strong></td>
      <td><strong>0.1061</strong></td>
    </tr>
  </tbody>
</table>

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
      <td>8.2768</td>
      <td>9.2625</td>
      <td>10.2548</td>
      <td>11.4056</td>
      <td>12.4331</td>
      <td>0.1830</td>
      <td>0.1632</td>
      <td>0.1447</td>
      <td>0.1260</td>
      <td>0.1117</td>
    </tr>
    <tr>
      <td>+ Channel Shrink</td>
      <td>8.9327</td>
      <td>9.7825</td>
      <td>10.5592</td>
      <td>11.5708</td>
      <td>12.7295</td>
      <td>0.1697</td>
      <td>0.1532</td>
      <td>0.1396</td>
      <td>0.1235</td>
      <td>0.1082</td>
    </tr>
    <tr>
      <td>+ Spatial Shrink</td>
      <td>8.4643</td>
      <td>9.3659</td>
      <td>10.3896</td>
      <td>11.4077</td>
      <td>12.5814</td>
      <td>0.1784</td>
      <td>0.1602</td>
      <td>0.1422</td>
      <td>0.1257</td>
      <td>0.1091</td>
    </tr>
    <tr>
      <td><strong>+ Channel & Spatial Shrink</strong></td>
      <td><strong>9.1616</strong></td>
      <td><strong>10.0953</strong></td>
      <td><strong>10.9872</strong></td>
      <td><strong>11.9503</strong></td>
      <td><strong>12.8933</strong></td>
      <td><strong>0.1647</strong></td>
      <td><strong>0.1480</strong></td>
      <td><strong>0.1331</strong></td>
      <td><strong>0.1187</strong></td>
      <td><strong>0.1061</strong></td>
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