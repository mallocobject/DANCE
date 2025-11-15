# DANCE: Dual Adaptive Noise-Cancellation and Enhancement for ECG Signals

## 网络概述

**DANCE** (Dual Adaptive Noise Cancellation and Enhancement) 是一种轻量级的 ECG 信号去噪与特征增强模块，专为强噪声环境下的心电信号鲁棒处理设计。通过**自适应阈值噪声消除(ATNC)**与**时空增强模块(STEM)**的协同机制，实现对噪声的精准抑制，同时显著增强**QRS复合波、P/T波**等临床关键特征。

模型采用全卷积与全局统计的混合架构，**无需外部参考信号**，可直接从单通道含噪ECG中学习纯净波形。具备**参数效率高、可解释性强、易于集成**的优势，可作为插件模块灵活嵌入各类ECG处理网络。

---

## 核心设计

### 双重自适应降噪与增强 (DANCE)

**DANCE** 由 **ATNC**（自适应阈值噪声消除模块）与 **STEM**（时空增强模块）两个子模块串行组成，分别实现 **全局能量感知的软阈值去噪** 与 **局部时空上下文的特征增强**，形成独特的信号净化与激励机制。

**DANCE 整体流程**：
$$
\mathbf{x} 
\xrightarrow{\text{ATNC}} \hat{\mathbf{x}}\ \text{(去噪)} 
\xrightarrow{\text{STEM}} \mathbf{y}\ \text{(增强)}
$$


---

### 1. 自适应阈值噪声消除 (ATNC)

**通道级动态软阈值收缩模块**

**前向传播过程**：

1. **信号分解**
   $$
   \mathbf{s} = \operatorname{sign}(\mathbf{x}), \quad \mathbf{a} = |\mathbf{x}|
   $$
   其中 $\mathbf{x} \in \mathbb{R}^{B \times C \times L}$ 为输入特征

2. **通道级全局统计提取**
   $$
   \mathbf{g} = \operatorname{GAP}(\mathbf{a}) = \frac{1}{L}\sum_{i=1}^{L}\mathbf{a}[:,:,i] \in \mathbb{R}^{B \times C \times 1}
   $$

3. **自适应阈值系数生成**
   $$
   \boldsymbol{\alpha} = \sigma\left(\mathbf{W}_2 \operatorname{ReLU}(\operatorname{BN}\left(\mathbf{W}_1 \mathbf{g}^\top + \mathbf{b}_1\right)) + \mathbf{b}_2\right) \in \mathbb{R}^{B \times C}
   $$

4. **动态阈值计算**
   $$
   \boldsymbol{\tau} = \mathbf{g} \odot \boldsymbol{\alpha}_{:,:,\text{None}} \in \mathbb{R}^{B \times C \times 1}
   $$

5. **软阈值去噪处理**
   $$
   \hat{\mathbf{a}} = \operatorname{ReLU}(\mathbf{a} - \boldsymbol{\tau}), \quad \hat{\mathbf{x}} = \mathbf{s} \odot \hat{\mathbf{a}}
   $$

---

### 2. 时空增强模块 (STEM)

**局部时空注意力激励模块**

**前向传播过程**：

1. **通道扩展与特征融合**
   $$
   \mathbf{h}_1 = \operatorname{ReLU}\left(\operatorname{BN}\left(\mathbf{W}_{\text{in}} * \hat{\mathbf{x}}\right)\right) \in \mathbb{R}^{B \times 2C \times L}
   $$

2. **深度可分离时序建模**
   $$
   \mathbf{h}_2 = \operatorname{ReLU}\left(\operatorname{BN}\left(\mathbf{W}_{\text{dw}} * \mathbf{h}_1\right)\right)
   $$
   其中 $\mathbf{W}_{\text{dw}}$ 为深度可分离卷积，组数=$2C$

3. **注意力图生成**
   $$
   \boldsymbol{\beta} = \sigma\left(\mathbf{W}_{\text{out}} * \mathbf{h}_2\right) \in [0,1]^{B \times C \times L}
   $$

4. **选择性特征增强**
   $$
   \mathbf{y} = \hat{\mathbf{x}} \odot \boldsymbol{\beta}
   $$

---

## 📊 模型性能对比

在**基线漂移(bw)、肌电噪声(ma)、电极运动(em)、复合干扰(emb)**四类噪声、**-4dB至+4dB**五个信噪比水平下的综合性能对比：


<table>
  <thead>
    <tr>
      <th rowspan="2">Noise Type</th>
      <th rowspan="2">Methods</th>
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
      <td rowspan="6">bw</td>
      <tr>
      <td>DWT</td>
      <td>-3.9960</td>
      <td>-2.0165</td>
      <td>-0.0457</td>
      <td>1.9118</td>
      <td>3.8495</td>
      <td>0.7294</td>
      <td>0.5805</td>
      <td>0.4623</td>
      <td>0.3687</td>
      <td>0.2946</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>9.5292</td>
      <td>10.1480</td>
      <td>10.7549</td>
      <td>11.6637</td>
      <td>12.5922</td>
      <td>0.1633</td>
      <td>0.1515</td>
      <td>0.1413</td>
      <td>0.1285</td>
      <td>0.1157</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>10.6044</td>
      <td>11.3291</td>
      <td>12.4020</td>
      <td>13.1558</td>
      <td>13.8094</td>
      <td>0.1449</td>
      <td>0.1344</td>
      <td>0.1193</td>
      <td>0.1108</td>
      <td>0.1034</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>10.4280</td>
      <td>10.9470</td>
      <td>11.9196</td>
      <td>12.6076</td>
      <td>13.4526</td>
      <td>0.1464</td>
      <td>0.1384</td>
      <td>0.1257</td>
      <td>0.1174</td>
      <td>0.1066</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>11.6337</strong></td>
      <td><strong>12.3434</strong></td>
      <td><strong>13.1573</strong></td>
      <td><strong>13.9295</strong></td>
      <td><strong>14.8686</strong></td>
      <td><strong>0.1321</strong></td>
      <td><strong>0.1219</strong></td>
      <td><strong>0.1109</strong></td>
      <td><strong>0.1034</strong></td>
      <td><strong>0.0930</strong></td>
    </tr>
    <tr>
      <td rowspan="6">ma</td>
      <tr>
      <td>DWT</td>
      <td>-2.8382</td>
      <td>-0.9080</td>
      <td>0.9988</td>
      <td>2.8774</td>
      <td>4.7232</td>
      <td>0.6453</td>
      <td>0.5157</td>
      <td>0.4130</td>
      <td>0.3317</td>
      <td>0.2673</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>7.2130</td>
      <td>7.7668</td>
      <td>8.3700</td>
      <td>9.3446</td>
      <td>10.1713</td>
      <td>0.2124</td>
      <td>0.1989</td>
      <td>0.1841</td>
      <td>0.1637</td>
      <td>0.1475</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>8.3792</td>
      <td>9.0164</td>
      <td>9.7187</td>
      <td>10.3924</td>
      <td>11.3412</td>
      <td>0.1876</td>
      <td>0.1738</td>
      <td>0.1585</td>
      <td>0.1454</td>
      <td>0.1300</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>8.3792</td>
      <td>8.8704</td>
      <td>9.5265</td>
      <td>10.2566</td>
      <td>11.2493</td>
      <td>0.1872</td>
      <td>0.1745</td>
      <td>0.1616</td>
      <td>0.1495</td>
      <td>0.1317</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>9.1507</strong></td>
      <td><strong>9.8982</strong></td>
      <td><strong>10.6039</strong></td>
      <td><strong>11.3618</strong></td>
      <td><strong>12.1724</strong></td>
      <td><strong>0.1731</strong></td>
      <td><strong>0.1590</strong></td>
      <td><strong>0.1450</strong></td>
      <td><strong>0.1324</strong></td>
      <td><strong>0.1192</strong></td>
    </tr>
    <tr>
      <td rowspan="6">em</td>
      <tr>
      <td>DWT</td>
      <td>-3.9738</td>
      <td>-2.0048</td>
      <td>-0.0463</td>
      <td>1.8975</td>
      <td>3.8210</td>
      <td>0.7271</td>
      <td>0.5793</td>
      <td>0.4620</td>
      <td>0.3690</td>
      <td>0.2953</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>8.2633</td>
      <td>9.0273</td>
      <td>9.8888</td>
      <td>10.7124</td>
      <td>11.5022</td>
      <td>0.1826</td>
      <td>0.1668</td>
      <td>0.1510</td>
      <td>0.1369</td>
      <td>0.1246</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>9.1970</td>
      <td>10.0596</td>
      <td>10.9456</td>
      <td>11.8838</td>
      <td>12.7189</td>
      <td>0.1644</td>
      <td>0.1483</td>
      <td>0.1330</td>
      <td>0.1199</td>
      <td>0.1089</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>8.8527</td>
      <td>9.7433</td>
      <td>10.6783</td>
      <td>11.5174</td>
      <td>12.1582</td>
      <td>0.1705</td>
      <td>0.1548</td>
      <td>0.1392</td>
      <td>0.1259</td>
      <td>0.1165</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>10.1728</strong></td>
      <td><strong>11.1074</strong></td>
      <td><strong>12.0221</strong></td>
      <td><strong>12.8331</strong></td>
      <td><strong>13.8240</strong></td>
      <td><strong>0.1488</strong></td>
      <td><strong>0.1331</strong></td>
      <td><strong>0.1197</strong></td>
      <td><strong>0.1088</strong></td>
      <td><strong>0.0977</strong></td>
    </tr>
    <tr>
      <td rowspan="6">emb</td>
      <tr>
      <td>DWT</td>
      <td>-3.9179</td>
      <td>-1.9532</td>
      <td>-0.0003</td>
      <td>1.9376</td>
      <td>3.8539</td>
      <td>0.7226</td>
      <td>0.5759</td>
      <td>0.4596</td>
      <td>0.3673</td>
      <td>0.2942</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>7.1990</td>
      <td>8.0031</td>
      <td>8.8498</td>
      <td>9.7801</td>
      <td>10.7494</td>
      <td>0.2098</td>
      <td>0.1904</td>
      <td>0.1715</td>
      <td>0.1542</td>
      <td>0.1369</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>8.1537</td>
      <td>8.9806</td>
      <td>9.9291</td>
      <td>10.9754</td>
      <td>11.7381</td>
      <td>0.1857</td>
      <td>0.1680</td>
      <td>0.1508</td>
      <td>0.1334</td>
      <td>0.1225</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>8.1164</td>
      <td>8.7772</td>
      <td>9.6396</td>
      <td>10.4104</td>
      <td>11.2274</td>
      <td>0.1874</td>
      <td>0.1729</td>
      <td>0.1570</td>
      <td>0.1433</td>
      <td>0.1302</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>DANCER (ours)</strong></td>
      <td><strong>9.1636</strong></td>
      <td><strong>10.0173</strong></td>
      <td><strong>10.9101</strong></td>
      <td><strong>11.7258</strong></td>
      <td><strong>12.5284</strong></td>
      <td><strong>0.1692</strong></td>
      <td><strong>0.1526</strong></td>
      <td><strong>0.1365</strong></td>
      <td><strong>0.1244</strong></td>
      <td><strong>0.1132</strong></td>
    </tr>
  </tbody>
</table>

---

## 🔬 消融实验分析

基于U-Net基线，逐步引入DANCE子模块的性能提升：

<table>
  <thead>
    <tr>
      <th rowspan="2">Noise Type</th>
      <th rowspan="2">Methods</th>
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
      <td rowspan="4">bw</td>
      <tr>
      <td>Baseline (U-Net)</td>
      <td>9.5292</td>
      <td>10.1480</td>
      <td>10.7549</td>
      <td>11.6637</td>
      <td>12.5922</td>
      <td>0.1633</td>
      <td>0.1515</td>
      <td>0.1413</td>
      <td>0.1285</td>
      <td>0.1157</td>
    </tr>
    <tr>
      <td>+ ATNC</td>
      <td>11.1428</td>
      <td>11.8738</td>
      <td>12.6695</td>
      <td>13.6302</td>
      <td>14.4109</td>
      <td>0.1388</td>
      <td>0.1280</td>
      <td>0.1161</td>
      <td>0.1066</td>
      <td>0.0979</td>
    </tr>
    <tr style="background-color: #afeef7ff;">
      <td><strong>+ ATNC & STEM</strong></td>
      <td><strong>11.6337</strong></td>
      <td><strong>12.3434</strong></td>
      <td><strong>13.1573</strong></td>
      <td><strong>13.9295</strong></td>
      <td><strong>14.8686</strong></td>
      <td><strong>0.1321</strong></td>
      <td><strong>0.1219</strong></td>
      <td><strong>0.1109</strong></td>
      <td><strong>0.1034</strong></td>
      <td><strong>0.0930</strong></td>
    </tr>
    <tr>
      <td rowspan="4">ma</td>
      <tr>
      <td>Baseline (U-Net)</td>
      <td>7.2130</td>
      <td>7.7668</td>
      <td>8.3700</td>
      <td>9.3446</td>
      <td>10.1713</td>
      <td>0.2124</td>
      <td>0.1989</td>
      <td>0.1841</td>
      <td>0.1637</td>
      <td>0.1475</td>
    </tr>
    <tr>
      <td>+ ATNC</td>
      <td>8.8483</td>
      <td>9.5550</td>
      <td>10.2312</td>
      <td>10.9499</td>
      <td>11.7030</td>
      <td>0.1787</td>
      <td>0.1656</td>
      <td>0.1511</td>
      <td>0.1387</td>
      <td>0.1249</td>
    </tr>
    <tr style="background-color: #afeef7ff;">
      <td><strong>+ ATNC & STEM</strong></td>
      <td><strong>9.1507</strong></td>
      <td><strong>9.8982</strong></td>
      <td><strong>10.6039</strong></td>
      <td><strong>11.3618</strong></td>
      <td><strong>12.1724</strong></td>
      <td><strong>0.1731</strong></td>
      <td><strong>0.1590</strong></td>
      <td><strong>0.1450</strong></td>
      <td><strong>0.1324</strong></td>
      <td><strong>0.1192</strong></td>
    </tr>
    <tr>
      <td rowspan="4">em</td>
      <tr>
      <td>Baseline (U-Net)</td>
      <td>8.2633</td>
      <td>9.0273</td>
      <td>9.8888</td>
      <td>10.7124</td>
      <td>11.5022</td>
      <td>0.1826</td>
      <td>0.1668</td>
      <td>0.1510</td>
      <td>0.1369</td>
      <td>0.1246</td>
    </tr>
    <tr>
      <td>+ ATNC</td>
      <td>9.7462</td>
      <td>10.6601</td>
      <td>11.5066</td>
      <td>12.2579</td>
      <td>13.2199</td>
      <td>0.1562</td>
      <td>0.1399</td>
      <td>0.1268</td>
      <td>0.1153</td>
      <td>0.1040</td>
    </tr>
    <tr style="background-color: #afeef7ff;">
      <td><strong>+ ATNC & STEM</strong></td>
      <td><strong>10.1728</strong></td>
      <td><strong>11.1074</strong></td>
      <td><strong>12.0221</strong></td>
      <td><strong>12.8331</strong></td>
      <td><strong>13.8240</strong></td>
      <td><strong>0.1488</strong></td>
      <td><strong>0.1331</strong></td>
      <td><strong>0.1197</strong></td>
      <td><strong>0.1088</strong></td>
      <td><strong>0.0977</strong></td>
    </tr>
    <tr>
      <td rowspan="4">emb</td>
      <tr>
      <td>Baseline (U-Net)</td>
      <td>7.1990</td>
      <td>8.0031</td>
      <td>8.8498</td>
      <td>9.7801</td>
      <td>10.7494</td>
      <td>0.2098</td>
      <td>0.1904</td>
      <td>0.1715</td>
      <td>0.1542</td>
      <td>0.1369</td>
    </tr>
    <tr>
      <td>+ ATNC</td>
      <td>8.8165</td>
      <td>9.5850</td>
      <td>10.4658</td>
      <td>11.3711</td>
      <td>12.0413</td>
      <td>0.1762</td>
      <td>0.1597</td>
      <td>0.1437</td>
      <td>0.1291</td>
      <td>0.1191</td>
    </tr>
    <tr style="background-color: #afeef7ff;">
      <td><strong>+ ATNC & STEM</strong></td>
      <td><strong>9.1636</strong></td>
      <td><strong>10.0173</strong></td>
      <td><strong>10.9101</strong></td>
      <td><strong>11.7258</strong></td>
      <td><strong>12.5284</strong></td>
      <td><strong>0.1692</strong></td>
      <td><strong>0.1526</strong></td>
      <td><strong>0.1365</strong></td>
      <td><strong>0.1244</strong></td>
      <td><strong>0.1132</strong></td>
    </tr>
  </tbody>
</table>

---

## 🎨 去噪效果可视化

![Denoising Comparison](./ecg_denoising_comparison.png)

*图示：不同去噪方法在-4dB噪声水平下对双通道ECG信号的去噪效果对比*

---

## 🏆 性能总结

实验结果表明，**DANCE模块**在多种噪声类型和信噪比水平下均表现出优异的性能：

- **全面的性能优势**：在bw、ma、em、emb四类噪声中，DANCE均取得最佳的信噪比提升和最低的重建误差
- **有效的模块协同**：消融实验证明ATNC与STEM的协同作用，两者结合可获得最大性能增益
- **鲁棒的特征保持**：在有效抑制噪声的同时，能够保持ECG信号的临床关键特征

---

**by Dan Liu, Tianhai Xie @IIP-2025**