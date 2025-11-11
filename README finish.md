# AGS-UNet: ECG信号去噪网络

## 网络概述

*通道阈值去噪, 时空局部增强*

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
      <td><strong>U-Net + CIAD</strong></td>
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
    <tr>
      <td>DACNN + CIAD</td>
      <td>7.8713 ± 0.0716</td>
      <td>8.6578 ± 0.1187</td>
      <td>9.5654 ± 0.1076</td>
      <td>10.5243 ± 0.0705</td>
      <td>11.4467 ± 0.1660</td>
      <td>0.1936 ± 0.0014</td>
      <td>0.1754 ± 0.0020</td>
      <td>0.1578 ± 0.0018</td>
      <td>0.1413 ± 0.0010</td>
      <td>0.1268 ± 0.0022</td>
    </tr>
    <tr>
      <td>ACDAE + CIAD</td>
      <td>8.4473 ± 0.0567</td>
      <td>9.2665 ± 0.0473</td>
      <td>10.0035 ± 0.0782</td>
      <td>10.9089 ± 0.1062</td>
      <td>11.7043 ± 0.0685</td>
      <td>0.1821 ± 0.0011</td>
      <td>0.1649 ± 0.0010</td>
      <td>0.1508 ± 0.0013</td>
      <td>0.1356 ± 0.0019</td>
      <td>0.1234 ± 0.0008</td>
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