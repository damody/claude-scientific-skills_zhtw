# 呼吸訊號處理

## 概述

NeuroKit2 的呼吸訊號處理支援呼吸模式、呼吸率、振幅和變異性分析。呼吸與心臟活動（呼吸性竇性心律不齊）、情緒狀態和認知過程密切相關。

## 主要處理管線

### rsp_process()

自動化呼吸訊號處理，包含波峰/波谷偵測和特徵提取。

```python
signals, info = nk.rsp_process(rsp_signal, sampling_rate=100, method='khodadad2018')
```

**管線步驟：**
1. 訊號清理（去噪、濾波）
2. 波峰（呼氣）和波谷（吸氣）偵測
3. 呼吸率計算
4. 振幅計算
5. 相位確定（吸氣/呼氣）
6. 每時間呼吸量（RVT）

**回傳：**
- `signals`：DataFrame 包含：
  - `RSP_Clean`：濾波後的呼吸訊號
  - `RSP_Peaks`、`RSP_Troughs`：極值標記
  - `RSP_Rate`：瞬時呼吸率（次/分鐘）
  - `RSP_Amplitude`：呼吸對呼吸振幅
  - `RSP_Phase`：吸氣（0）vs. 呼氣（1）
  - `RSP_Phase_Completion`：相位完成百分比（0-1）
  - `RSP_RVT`：每時間呼吸量
- `info`：包含波峰/波谷索引的字典

**方法：**
- `'khodadad2018'`：Khodadad et al. 演算法（預設，穩健）
- `'biosppy'`：基於 BioSPPy 的處理（替代）

## 預處理函數

### rsp_clean()

移除噪音並平滑呼吸訊號。

```python
cleaned_rsp = nk.rsp_clean(rsp_signal, sampling_rate=100, method='khodadad2018')
```

**方法：**

**1. Khodadad2018（預設）：**
- Butterworth 低通濾波器
- 移除高頻噪音
- 保留呼吸波形

**2. BioSPPy：**
- 替代濾波方法
- 與 Khodadad 相似的性能

**3. Hampel 濾波器：**
```python
cleaned_rsp = nk.rsp_clean(rsp_signal, sampling_rate=100, method='hampel')
```
- 基於中位數的異常值移除
- 對偽跡和尖峰穩健
- 保留銳利轉換

**典型呼吸頻率：**
- 靜息成人：12-20 次/分鐘（0.2-0.33 Hz）
- 兒童：更快的速率
- 運動期間：高達 40-60 次/分鐘

### rsp_peaks()

識別呼吸訊號中的吸氣波谷和呼氣波峰。

```python
peaks, info = nk.rsp_peaks(cleaned_rsp, sampling_rate=100, method='khodadad2018')
```

**偵測方法：**
- `'khodadad2018'`：針對乾淨訊號最佳化
- `'biosppy'`：替代方法
- `'scipy'`：簡單的基於 scipy 的偵測

**回傳：**
- 字典包含：
  - `RSP_Peaks`：呼氣波峰索引（最大點）
  - `RSP_Troughs`：吸氣波谷索引（最小點）

**呼吸週期定義：**
- **吸氣**：波谷 → 波峰（空氣流入，胸部/腹部擴張）
- **呼氣**：波峰 → 波谷（空氣流出，胸部/腹部收縮）

### rsp_findpeaks()

帶多種演算法選項的低階波峰偵測。

```python
peaks_dict = nk.rsp_findpeaks(cleaned_rsp, sampling_rate=100, method='scipy')
```

**方法：**
- `'scipy'`：Scipy 的 find_peaks
- 自訂基於閾值的演算法

**應用場景：**
- 精細波峰偵測
- 自訂參數調整
- 演算法比較

### rsp_fixpeaks()

校正偵測到的波峰/波谷異常（例如，遺漏或錯誤偵測）。

```python
corrected_peaks = nk.rsp_fixpeaks(peaks, sampling_rate=100)
```

**校正：**
- 移除生理上不合理的間隔
- 內插遺漏的波峰
- 移除與偽跡相關的錯誤波峰

## 特徵提取函數

### rsp_rate()

計算瞬時呼吸率（每分鐘呼吸次數）。

```python
rate = nk.rsp_rate(peaks, sampling_rate=100, desired_length=None)
```

**方法：**
- 從波峰/波谷時間計算呼吸間隔
- 轉換為每分鐘呼吸次數（BPM）
- 內插以匹配訊號長度

**典型值：**
- 靜息成人：12-20 BPM
- 慢速呼吸：<10 BPM（冥想、放鬆）
- 快速呼吸：>25 BPM（運動、焦慮）

### rsp_amplitude()

計算呼吸對呼吸振幅（波峰到波谷差異）。

```python
amplitude = nk.rsp_amplitude(cleaned_rsp, peaks)
```

**解讀：**
- 較大振幅：較深的呼吸（潮氣量增加）
- 較小振幅：淺呼吸
- 可變振幅：不規則呼吸模式

**臨床相關性：**
- 振幅減少：限制性肺病、胸壁僵硬
- 振幅增加：代償性過度換氣

### rsp_phase()

確定吸氣/呼氣相位和完成百分比。

```python
phase, completion = nk.rsp_phase(cleaned_rsp, peaks, sampling_rate=100)
```

**回傳：**
- `RSP_Phase`：二元（0 = 吸氣，1 = 呼氣）
- `RSP_Phase_Completion`：連續 0-1 表示相位進度

**應用場景：**
- 呼吸門控刺激呈現
- 相位鎖定平均
- 呼吸-心臟耦合分析

### rsp_symmetry()

分析呼吸對稱性模式（波峰-波谷平衡、上升-下降時間）。

```python
symmetry = nk.rsp_symmetry(cleaned_rsp, peaks)
```

**指標：**
- 波峰-波谷對稱性：波峰和波谷是否等距？
- 上升-下降對稱性：吸氣時間是否等於呼氣時間？

**解讀：**
- 對稱：正常、放鬆的呼吸
- 不對稱：費力呼吸、呼吸道阻塞

## 進階分析函數

### rsp_rrv()

呼吸率變異性 - 類似於心率變異性。

```python
rrv_indices = nk.rsp_rrv(peaks, sampling_rate=100)
```

**時域指標：**
- `RRV_SDBB`：呼吸間隔標準差
- `RRV_RMSSD`：連續差異的均方根
- `RRV_MeanBB`：平均呼吸間隔

**頻域指標：**
- 頻帶功率（如適用）

**解讀：**
- 較高 RRV：靈活、適應性的呼吸控制
- 較低 RRV：僵硬、受限的呼吸
- RRV 改變：焦慮、呼吸疾病、自主神經功能障礙

**記錄時長：**
- 最低：2-3 分鐘
- 最佳：5-10 分鐘以獲得穩定估計

### rsp_rvt()

每時間呼吸量 - fMRI 混淆因子迴歸器。

```python
rvt = nk.rsp_rvt(cleaned_rsp, peaks, sampling_rate=100)
```

**計算：**
- 呼吸訊號的導數
- 捕捉體積變化率
- 與 BOLD 訊號波動相關

**應用場景：**
- fMRI 偽跡校正
- 神經影像預處理
- 呼吸混淆因子迴歸

**參考：**
- Birn, R. M., et al. (2008). Separating respiratory-variation-related fluctuations from neuronal-activity-related fluctuations in fMRI. NeuroImage, 31(4), 1536-1548.

### rsp_rav()

呼吸振幅變異性指標。

```python
rav = nk.rsp_rav(amplitude, sampling_rate=100)
```

**指標：**
- 振幅標準差
- 變異係數
- 振幅範圍

**解讀：**
- 高 RAV：不規則深度（嘆息、喚起變化）
- 低 RAV：穩定、受控的呼吸

## 分析函數

### rsp_analyze()

自動選擇事件相關或區間相關分析。

```python
analysis = nk.rsp_analyze(signals, sampling_rate=100)
```

**模式選擇：**
- 時長 < 10 秒 → 事件相關
- 時長 ≥ 10 秒 → 區間相關

### rsp_eventrelated()

分析對特定事件/刺激的呼吸反應。

```python
results = nk.rsp_eventrelated(epochs)
```

**計算的指標（每時段）：**
- `RSP_Rate_Mean`：時段期間的平均呼吸率
- `RSP_Rate_Min/Max`：最小/最大呼吸率
- `RSP_Amplitude_Mean`：平均呼吸深度
- `RSP_Phase`：事件開始時的呼吸相位
- 跨時段的呼吸率和振幅動態

**應用場景：**
- 情緒刺激期間的呼吸變化
- 任務事件前的預期性呼吸
- 屏息或過度換氣範式

### rsp_intervalrelated()

分析延長的呼吸記錄。

```python
results = nk.rsp_intervalrelated(signals, sampling_rate=100)
```

**計算的指標：**
- `RSP_Rate_Mean`：平均呼吸率
- `RSP_Rate_SD`：呼吸率變異性
- `RSP_Amplitude_Mean`：平均呼吸深度
- RRV 指標（如有足夠資料）
- RAV 指標

**記錄時長：**
- 最低：60 秒
- 最佳：5-10 分鐘

**應用場景：**
- 靜息狀態呼吸模式
- 基線呼吸評估
- 壓力或放鬆監測

## 模擬和視覺化

### rsp_simulate()

生成用於測試的合成呼吸訊號。

```python
synthetic_rsp = nk.rsp_simulate(duration=60, sampling_rate=100, respiratory_rate=15,
                                method='sinusoidal', noise=0.1, random_state=42)
```

**方法：**
- `'sinusoidal'`：簡單的正弦振盪（快速）
- `'breathmetrics'`：進階真實呼吸模型（較慢、更準確）

**參數：**
- `respiratory_rate`：每分鐘呼吸次數（預設：15）
- `noise`：高斯噪音水平
- `random_state`：可重現性種子

**應用場景：**
- 演算法驗證
- 參數調整
- 教育示範

### rsp_plot()

視覺化處理後的呼吸訊號。

```python
nk.rsp_plot(signals, info, static=True)
```

**顯示：**
- 原始和清理後的呼吸訊號
- 偵測到的波峰和波谷
- 瞬時呼吸率
- 相位標記

**互動模式：** 設定 `static=False` 使用 Plotly 視覺化

## 實務考量

### 取樣率建議
- **最低**：10 Hz（足夠進行呼吸率估計）
- **標準**：50-100 Hz（研究級）
- **高解析度**：1000 Hz（通常不必要，過度取樣）

### 記錄時長
- **呼吸率估計**：≥10 秒（幾次呼吸）
- **RRV 分析**：≥2-3 分鐘
- **靜息狀態**：5-10 分鐘
- **晝夜模式**：數小時至數天

### 訊號採集方法

**應變規/壓電帶：**
- 胸部或腹部擴張
- 最常見
- 舒適、非侵入性

**熱敏電阻/熱電偶：**
- 鼻/口氣流溫度
- 直接氣流測量
- 可能具侵入性

**呼氣末 CO₂ 測量：**
- 呼氣末 CO₂ 測量
- 生理學黃金標準
- 昂貴，臨床環境

**阻抗描記法：**
- 從 ECG 電極衍生
- 方便進行多模態記錄
- 不如專用感測器準確

### 常見問題和解決方案

**不規則呼吸：**
- 清醒休息狀態下正常
- 嘆息、打哈欠、說話、吞嚥造成變異
- 排除偽跡或作為事件建模

**淺呼吸：**
- 低訊號振幅
- 檢查感測器放置和緊度
- 如可用，增加增益

**動作偽跡：**
- 尖峰或不連續
- 最小化參與者運動
- 使用穩健的波峰偵測（Hampel 濾波器）

**說話/咳嗽：**
- 干擾自然呼吸模式
- 註記並從分析中排除
- 或作為單獨事件類型建模

### 最佳實踐

**標準工作流程：**
```python
# 1. 清理訊號
cleaned = nk.rsp_clean(rsp_raw, sampling_rate=100, method='khodadad2018')

# 2. 偵測波峰/波谷
peaks, info = nk.rsp_peaks(cleaned, sampling_rate=100)

# 3. 提取特徵
rate = nk.rsp_rate(peaks, sampling_rate=100, desired_length=len(cleaned))
amplitude = nk.rsp_amplitude(cleaned, peaks)
phase = nk.rsp_phase(cleaned, peaks, sampling_rate=100)

# 4. 完整處理（替代方案）
signals, info = nk.rsp_process(rsp_raw, sampling_rate=100)

# 5. 分析
analysis = nk.rsp_analyze(signals, sampling_rate=100)
```

**呼吸-心臟整合：**
```python
# 處理兩個訊號
ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=1000)
rsp_signals, rsp_info = nk.rsp_process(rsp, sampling_rate=100)

# 呼吸性竇性心律不齊（RSA）
rsa = nk.hrv_rsa(ecg_info['ECG_R_Peaks'], rsp_signals['RSP_Clean'], sampling_rate=1000)

# 或使用 bio_process 進行多訊號整合
bio_signals, bio_info = nk.bio_process(ecg=ecg, rsp=rsp, sampling_rate=1000)
```

## 臨床和研究應用

**心理生理學：**
- 情緒和喚起（壓力期間快速、淺呼吸）
- 放鬆介入（慢速、深呼吸）
- 呼吸生物回饋

**焦慮和恐慌症：**
- 恐慌發作期間的過度換氣
- 改變的呼吸模式
- 呼吸再訓練治療效果

**睡眠醫學：**
- 睡眠呼吸暫停偵測
- 呼吸模式異常
- 睡眠階段相關性

**心肺耦合：**
- 呼吸性竇性心律不齊（呼吸對 HRV 的調節）
- 心肺互動
- 自主神經系統評估

**神經影像：**
- fMRI 偽跡校正（RVT 迴歸器）
- BOLD 訊號混淆因子移除
- 與呼吸相關的大腦活動

**冥想和正念：**
- 呼吸覺察訓練
- 慢速呼吸練習（共振頻率 ~6 次/分鐘）
- 放鬆的生理標記

**運動表現：**
- 呼吸效率
- 訓練適應
- 恢復監測

## 解讀指南

**呼吸率：**
- **正常**：12-20 BPM（靜息成人）
- **慢速**：<10 BPM（放鬆、冥想、睡眠）
- **快速**：>25 BPM（運動、焦慮、疼痛、發燒）

**呼吸振幅：**
- 靜息時潮氣量通常 400-600 mL
- 深呼吸：2-3 L
- 淺呼吸：<300 mL

**呼吸模式：**
- **正常**：平滑、規則的正弦波
- **Cheyne-Stokes**：漸強漸弱伴呼吸暫停（臨床病理）
- **失調**：完全不規則（腦幹病變）

## 參考文獻

- Khodadad, D., Nordebo, S., Müller, B., Waldmann, A., Yerworth, R., Becher, T., ... & Bayford, R. (2018). A review of tissue substitutes for ultrasound imaging. Ultrasound in medicine & biology, 44(9), 1807-1823.
- Grossman, P., & Taylor, E. W. (2007). Toward understanding respiratory sinus arrhythmia: Relations to cardiac vagal tone, evolution and biobehavioral functions. Biological psychology, 74(2), 263-285.
- Birn, R. M., Diamond, J. B., Smith, M. A., & Bandettini, P. A. (2006). Separating respiratory-variation-related fluctuations from neuronal-activity-related fluctuations in fMRI. NeuroImage, 31(4), 1536-1548.
