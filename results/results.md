## 📊 模型性能对比

在**基线漂移(bw)、肌电噪声(ma)、电极运动(em)、复合干扰(emb)**四类噪声、**-4dB**信噪比水平下的综合性能对比：


<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="4">SNR(dB)</th>
      <th colspan="4">RMSE</th>
    </tr>
    <tr>
      <th>bw</th>
      <th>em</th>
      <th>ma</th>
      <th>emb</th>
      <th>bw</th>
      <th>em</th>
      <th>ma</th>
      <th>emb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DWT</td>
      <td>1.82</td>
      <td>1.32</td>
      <td>1.69</td>
      <td>1.24</td>
      <td>0.4189</td>
      <td>0.4383</td>
      <td>0.4124</td>
      <td>0.4406</td>
    </tr>
    <tr>
      <td>EMD</td>
      <td>2.95</td>
      <td>1.14</td>
      <td>0.49</td>
      <td>0.94</td>
      <td>0.3680</td>
      <td>0.4454</td>
      <td>0.4757</td>
      <td>0.4519</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>6.89</td>
      <td>6.33</td>
      <td>5.73</td>
      <td>5.33</td>
      <td>0.2393</td>
      <td>0.2518</td>
      <td>0.2698</td>
      <td>0.2845</td>
    </tr>
    <tr>
      <td>DRSN</td>
      <td>7.77</td>
      <td>7.10</td>
      <td>6.47</td>
      <td>6.07</td>
      <td>0.2214</td>
      <td>0.2337</td>
      <td>0.2500</td>
      <td>0.2612</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>8.23</td>
      <td>7.20</td>
      <td><strong>7.08</strong></td>
      <td>6.39</td>
      <td>0.2147</td>
      <td>0.2305</td>
      <td><strong>0.2364</strong></td>
      <td>0.2521</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>7.89</td>
      <td>6.91</td>
      <td>6.65</td>
      <td>6.09</td>
      <td>0.2172</td>
      <td>0.2371</td>
      <td>0.2470</td>
      <td>0.2604</td>
    </tr>
    <tr>
      <td>RALENet</td>
      <td>7.84</td>
      <td>5.16</td>
      <td>5.25</td>
      <td>4.58</td>
      <td>0.2187</td>
      <td>0.2797</td>
      <td>0.2838</td>
      <td>0.2995</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>DANCE (ours)</strong></td>
      <td><strong>8.52</strong></td>
      <td><strong>7.43</strong></td>
      <td>6.96</td>
      <td><strong>6.56</strong></td>
      <td><strong>0.2049</strong></td>
      <td><strong>0.2263</strong></td>
      <td>0.2380</td>
      <td><strong>0.2464</strong></td>
    </tr>
  </tbody>
</table>

---

在**复合干扰(emb)**四类噪声、**-4dB至+4dB**五个信噪比水平下的综合性能对比：

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
      <td>DWT</td>
      <td>1.24</td>
      <td>1.64</td>
      <td>2.57</td>
      <td>3.97</td>
      <td>5.33</td>
      <td>0.4406</td>
      <td>0.4241</td>
      <td>0.3810</td>
      <td>0.3208</td>
      <td>0.2756</td>
    </tr>
    <tr>
      <td>EMD</td>
      <td>0.94</td>
      <td>1.15</td>
      <td>1.53</td>
      <td>2.14</td>
      <td>4.00</td>
      <td>0.4519</td>
      <td>0.4460</td>
      <td>0.4277</td>
      <td>0.4011</td>
      <td>0.3210</td>
    </tr>
    <tr>
      <td>U-Net</td>
      <td>5.33</td>
      <td>6.15</td>
      <td>6.69</td>
      <td>7.75</td>
      <td>8.68</td>
      <td>0.2845</td>
      <td>0.2585</td>
      <td>0.2437</td>
      <td>0.2164</td>
      <td>0.1949</td>
    </tr>
    <tr>
      <tr>
      <td>DRSN</td>
      <td>6.07</td>
      <td>6.85</td>
      <td>7.53</td>
      <td>8.29</td>
      <td>9.06</td>
      <td>0.2612</td>
      <td>0.2468</td>
      <td>0.2236</td>
      <td>0.2028</td>
      <td>0.1883</td>
    </tr>
    <tr>
      <td>DACNN</td>
      <td>6.39</td>
      <td>7.16</td>
      <td>7.66</td>
      <td>8.64</td>
      <td>9.29</td>
      <td>0.2521</td>
      <td>0.2321</td>
      <td>0.2214</td>
      <td>0.1954</td>
      <td>0.1858</td>
    </tr>
    <tr>
      <td>ACDAE</td>
      <td>6.09</td>
      <td>6.77</td>
      <td>7.36</td>
      <td>8.33</td>
      <td>8.87</td>
      <td>0.2604</td>
      <td>0.2433</td>
      <td>0.2266</td>
      <td>0.2030</td>
      <td>0.1941</td>
    </tr>
    <tr>
      <td>RALENet</td>
      <td>4.58</td>
      <td>5.47</td>
      <td>6.48</td>
      <td>7.62</td>
      <td>8.69</td>
      <td>0.2995</td>
      <td>0.2709</td>
      <td>0.2431</td>
      <td>0.2151</td>
      <td>0.1917</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>DANCE (ours)</strong></td>
      <td><strong>6.56</strong></td>
      <td><strong>7.41</strong></td>
      <td><strong>7.94</strong></td>
      <td><strong>8.87</strong></td>
      <td><strong>9.64</strong></td>
      <td><strong>0.2464</strong></td>
      <td><strong>0.2263</strong></td>
      <td><strong>0.2127</strong></td>
      <td><strong>0.1907</strong></td>
      <td><strong>0.1765</strong></td>
    </tr>
  </tbody>
</table>

---


## 🧩 组件分析

### 实验设计
在相同的U-Net基线架构上，对比与不同的注意力机制，评估在emb噪声类型下的去噪性能：

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
      <td>5.33</td>
      <td>6.15</td>
      <td>6.69</td>
      <td>7.75</td>
      <td>8.68</td>
      <td>0.2845</td>
      <td>0.2585</td>
      <td>0.2437</td>
      <td>0.2164</td>
      <td>0.1949</td>
    </tr>
    <tr>
      <td>+ ATNC & SE</td>
      <td>6.50</td>
      <td>7.33</td>
      <td>8.05</td>
      <td>8.86</td>
      <td>9.70</td>
      <td>0.2481</td>
      <td>0.2283</td>
      <td>0.2101</td>
      <td>0.1905</td>
      <td>0.1741</td>
    </tr>
    <tr>
      <td>+ ATNC & CBAM</td>
      <td>6.37</td>
      <td>7.28</td>
      <td>7.80</td>
      <td>8.77</td>
      <td>9.60</td>
      <td>0.2517</td>
      <td>0.2301</td>
      <td>0.2158</td>
      <td>0.1922</td>
      <td>0.1758</td>
    </tr>
    <tr>
      <td>+ ATNC & ECA</td>
      <td>6.42</td>
      <td>7.32</td>
      <td>7.95</td>
      <td>8.81</td>
      <td>9.68</td>
      <td>0.2506</td>
      <td>0.2286</td>
      <td>0.2134</td>
      <td>0.1920</td>
      <td>0.1752</td>
    </tr>
    <tr style="background-color: #cdeecdff;">
      <td><strong>+ ATNC & AREM</strong></td>
      <td><strong>6.56</strong></td>
      <td><strong>7.41</strong></td>
      <td><strong>7.94</strong></td>
      <td><strong>8.87</strong></td>
      <td><strong>9.64</strong></td>
      <td><strong>0.2464</strong></td>
      <td><strong>0.2263</strong></td>
      <td><strong>0.2127</strong></td>
      <td><strong>0.1907</strong></td>
      <td><strong>0.1765</strong></td>
    </tr>
  </tbody>
</table>

---

## 🔬 消融实验分析

基于U-Net基线，逐步引入DANCE子模块的性能提升：

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
      <td>Baseline (U-Net)</td>
      <td>5.33</td>
      <td>6.15</td>
      <td>6.69</td>
      <td>7.75</td>
      <td>8.68</td>
      <td>0.2845</td>
      <td>0.2585</td>
      <td>0.2437</td>
      <td>0.2164</td>
      <td>0.1949</td>
    </tr>
    <tr>
      <td>+ AREM</td>
      <td>5.49</td>
      <td>6.26</td>
      <td>6.88</td>
      <td>7.82</td>
      <td>8.70</td>
      <td>0.2779</td>
      <td>0.2538</td>
      <td>0.2373</td>
      <td>0.2126</td>
      <td>0.1921</td>
    </tr>
    <tr>
      <td>+ ATNC</td>
      <td>6.44</td>
      <td>7.33</td>
      <td>7.90</td>
      <td>8.73</td>
      <td>9.57</td>
      <td>0.2495</td>
      <td>0.2272</td>
      <td>0.2130</td>
      <td>0.1923</td>
      <td>0.1760</td>
    </tr>
    <tr>
      <td style="background-color: #afeef7ff;"><strong>+ ATNC & AREM</strong></td>
      <td><strong>6.56</strong></td>
      <td><strong>7.41</strong></td>
      <td><strong>7.94</strong></td>
      <td><strong>8.87</strong></td>
      <td><strong>9.64</strong></td>
      <td><strong>0.2464</strong></td>
      <td><strong>0.2263</strong></td>
      <td><strong>0.2127</strong></td>
      <td><strong>0.1907</strong></td>
      <td><strong>0.1765</strong></td>
    </tr>
  </tbody>
</table>

---


## 🎨 去噪效果可视化

![Denoising Comparison](./ecg_denoising_comparison.png)

*图示：不同去噪方法在-4dB噪声水平下对双通道ECG信号的去噪效果对比*

