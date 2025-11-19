# DANCE: Dual Adaptive Noise-Cancellation and Enhancement for ECG Signals

## 网络概述

**DANCE**（Dual Adaptive Noise-Cancellation and Enhancement）是一种轻量级、可解释的 ECG 信号去噪与特征增强模块，由两个互补的子模块组成：

- **ATNC** — Adaptive Threshold Noise Cancellation（自适应阈值噪声消除）
- **ALEM** — Adaptive Local Enhancement Module（自适应局部增强）

DANCE 可在强噪声背景下恢复关键 ECG 波形（P/QRS/T），具备参数效率高、可解释性强、易于集成等优势。

---

## 核心设计

### 双模块结构：DANCE

DANCE 采用串行处理策略，将去噪与增强解耦：

**DANCE 整体流程：**

$$
\mathbf{x} 
\xrightarrow{\text{ATNC}} \hat{\mathbf{x}}\ \text{(去噪)} 
\xrightarrow{\text{ALEM}} \mathbf{y}\ \text{(增强)}
$$

---

## 1. 自适应阈值噪声消除 (Adaptive Threshold Noise Cancellation, ATNC)

**通道级动态软阈值收缩模块**

### 前向传播过程

#### 1. 信号分解（符号与绝对值）
$$
\mathbf{s} = \mathrm{sign}(\mathbf{x}), \quad 
\mathbf{a} = |\mathbf{x}|
$$

其中 $\mathbf{x} \in \mathbb{R}^{B \times C \times L}$ 为输入特征图。

#### 2. 通道级全局统计量提取（Global Average Pooling）
$$
\mathbf{g} = \mathrm{GAP}(\mathbf{a}) = \frac{1}{L} \sum_{i=1}^{L} \mathbf{a}[:,:,i] \in \mathbb{R}^{B \times C \times 1}
$$

#### 3. 自适应阈值系数生成（小 MLP）
$$
\boldsymbol{\alpha} = \sigma\!\left( \mathbf{W}_2 \, \mathrm{ReLU}\!\left( \mathrm{BN}(\mathbf{W}_1 \mathbf{g}[:,:] + \mathbf{b}_1) \right) + \mathbf{b}_2 \right) \in \mathbb{R}^{B \times C}
$$

（注：先将 $\mathbf{g}$ 从 $B \times C \times 1$ 压平成 $B \times C$ 后送入全连接层）

#### 4. 动态阈值计算（逐通道广播）
$$
\boldsymbol{\tau} = \mathbf{g} \odot \boldsymbol{\alpha}[:,:,\text{None}] \in \mathbb{R}^{B \times C \times 1}
$$

#### 5. 软阈值去噪（Soft Thresholding）
$$
\hat{\mathbf{a}} = \mathrm{ReLU}(\mathbf{a} - \boldsymbol{\tau}), \quad 
\hat{\mathbf{x}} = \mathbf{s} \odot \hat{\mathbf{a}}
$$

---

## 2. 自适应局部增强模块 (Adaptive Local Enhancement Module, ALEM)

**基于局部时序建模的特征权重增强**

### 前向传播过程

#### 1. 通道扩展与特征融合
$$
\mathbf{h}_1 = \mathrm{ReLU}\left( \mathrm{BN}\left( \mathbf{W}_{\text{in}} * \hat{\mathbf{x}} \right) \right) 
\in \mathbb{R}^{B \times 2C \times L}
$$

$\mathbf{W}_{\text{in}}$ 为 $1\!\times\!1$ 点卷积，将通道数扩展至 2C。

#### 2. 深度可分离时序建模（Depthwise Temporal Modeling）
$$
\mathbf{h}_2 = \mathrm{ReLU}\left( \mathrm{BN}\left( \mathbf{W}_{\text{dw}} * \mathbf{h}_1 \right) \right)
\in \mathbb{R}^{B \times 2C \times L}
$$

其中 $\mathbf{W}_{\text{dw}}$ 为深度可分离卷积（groups = 2C），用于捕获局部时序依赖。

#### 3. 增强掩码生成（Sigmoid 激活）
$$
\boldsymbol{\beta} = \sigma\!\left( \mathbf{W}_{\text{out}} * \mathbf{h}_2 \right) 
\in [0,1]^{B \times C \times L}
$$

$\mathbf{W}_{\text{out}}$ 为 $1\!\times\!1$ 卷积，将通道压缩回 C 并生成逐位置、逐通道的增强权重。

#### 4. 选择性特征增强
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
      <td rowspan="5">bw</td>
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
      <td rowspan="5">ma</td>
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
      <td rowspan="5">em</td>
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
      <td rowspan="5">emb</td>
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

## 📈 DANCE与多种注意力模块性能对比

### 实验设计
在相同的U-Net基线架构上，评估DANCE模块与不同的注意力机制在emb噪声类型下的去噪性能：

<table>
  <thead>
    <tr>
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
      <td>+ SE</td>
      <td>8.0922</td>
      <td>8.8490</td>
      <td>9.7130</td>
      <td>10.6370</td>
      <td>11.5692</td>
      <td>0.1887</td>
      <td>0.1717</td>
      <td>0.1560</td>
      <td>0.1400</td>
      <td>0.1260</td>
    </tr>
    <tr>
      <td>+ CBAM</td>
      <td>8.4976</td>
      <td>9.1831</td>
      <td>9.8560</td>
      <td>10.6545</td>
      <td>11.7688</td>
      <td>0.1828</td>
      <td>0.1659</td>
      <td>0.1530</td>
      <td>0.1391</td>
      <td>0.1227</td>
    </tr>
    <tr>
      <td>+ ECA</td>
      <td>7.5755</td>
      <td>8.4286</td>
      <td>9.1713</td>
      <td>10.0636</td>
      <td>10.9302</td>
      <td>0.1970</td>
      <td>0.1797</td>
      <td>0.1646</td>
      <td>0.1487</td>
      <td>0.1346</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>+ DANCE</strong></td>
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

## 🧩 组件组合分析

### 实验设计
在相同的U-Net基线架构上，将ATNC模块分别与不同的注意力机制组合，评估在emb噪声类型下的去噪性能：

<table>
  <thead>
    <tr>
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
      <td>+ ATNC & SE</td>
      <td>9.0955</td>
      <td>9.9871</td>
      <td>10.8194</td>
      <td>11.6121</td>
      <td>12.4966</td>
      <td>0.1712</td>
      <td><strong>0.1522</strong></td>
      <td>0.1377</td>
      <td>0.1260</td>
      <td>0.1137</td>
    </tr>
    <tr>
      <td>+ ATNC & CBAM</td>
      <td>8.7101</td>
      <td>9.5872</td>
      <td>10.4739</td>
      <td>11.4075</td>
      <td>12.1743</td>
      <td>0.1778</td>
      <td>0.1592</td>
      <td>0.1433</td>
      <td>0.1285</td>
      <td>0.1173</td>
    </tr>
    <tr>
      <td>+ ATNC & ECA</td>
      <td>8.8771</td>
      <td>9.7266</td>
      <td>10.5383</td>
      <td>11.4005</td>
      <td>12.2690</td>
      <td>0.1745</td>
      <td>0.1572</td>
      <td>0.1429</td>
      <td>0.1284</td>
      <td>0.1163</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>+ ATNC & ALEM (DANCE)</strong></td>
      <td><strong>9.1636</strong></td>
      <td><strong>10.0173</strong></td>
      <td><strong>10.9101</strong></td>
      <td><strong>11.7258</strong></td>
      <td><strong>12.5284</strong></td>
      <td><strong>0.1692</strong></td>
      <td>0.1526</td>
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
      <td rowspan="5">bw</td>
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
    <tr>
      <td>+ ALEM</td>
      <td>10.5171</td>
      <td>11.0747</td>
      <td>11.8112</td>
      <td>12.7277</td>
      <td>13.4625</td>
      <td>0.1465</td>
      <td>0.1373</td>
      <td>0.1256</td>
      <td>0.1143</td>
      <td>0.1066</td>
    </tr>
    <tr>
      <td style="background-color: #afeef7ff;"><strong>+ ATNC & ALEM</strong></td>
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
      <td rowspan="5">ma</td>
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
    <tr>
      <td>+ ALEM</td>
      <td>8.1146</td>
      <td>8.6506</td>
      <td>9.3925</td>
      <td>10.0354</td>
      <td>10.9139</td>
      <td>0.1924</td>
      <td>0.1797</td>
      <td>0.1645</td>
      <td>0.1515</td>
      <td>0.1355</td>
    </tr>
    <tr>
      <td style="background-color: #afeef7ff;"><strong>+ ATNC & ALEM</strong></td>
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
      <td rowspan="5">em</td>
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
    <tr>
      <td>+ ALEM</td>
      <td>9.2341</td>
      <td>9.9263</td>
      <td>10.7868</td>
      <td>11.5670</td>
      <td>12.5154</td>
      <td>0.1628</td>
      <td>0.1510</td>
      <td>0.1364</td>
      <td>0.1243</td>
      <td>0.1116</td>
    </tr>
    <tr>
      <td style="background-color: #afeef7ff;"><strong>+ ATNC & ALEM</strong></td>
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
      <td rowspan="5">emb</td>
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
    <tr>
      <td>+ ALEM</td>
      <td>8.1553</td>
      <td>8.8737</td>
      <td>9.6007</td>
      <td>10.5840</td>
      <td>11.4104</td>
      <td>0.1876</td>
      <td>0.1713</td>
      <td>0.1576</td>
      <td>0.1397</td>
      <td>0.1272</td>
    </tr>
    <tr>
      <td style="background-color: #afeef7ff;"><strong>+ ATNC & ALEM</strong></td>
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
- **有效的模块协同**：消融实验证明ATNC与ALEM的协同作用，两者结合可获得最大性能增益
- **鲁棒的特征保持**：在有效抑制噪声的同时，能够保持ECG信号的临床关键特征

---

**by Dan Liu, Tianhai Xie @IIP-2025**