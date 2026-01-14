# 心率變異性（HRV）分析

## 概述

心率變異性（Heart Rate Variability，HRV）反映連續心跳之間時間間隔的變化，提供自主神經系統調節、心血管健康和心理狀態的洞察。NeuroKit2 提供時域、頻域和非線性域的全面 HRV 分析。

## 主要函數

### hrv()

一次計算所有可用的 HRV 指標，涵蓋所有域。

```python
hrv_indices = nk.hrv(peaks, sampling_rate=1000, show=False)
```

**輸入：**
- `peaks`：包含 `'ECG_R_Peaks'` 鍵的字典或 R 波峰索引陣列
- `sampling_rate`：訊號取樣率（Hz）

**回傳：**
- DataFrame，包含所有域的 HRV 指標：
  - 時域指標
  - 頻域功率譜
  - 非線性複雜度測量

**這是一個便利包裝器**，結合了：
- `hrv_time()`
- `hrv_frequency()`
- `hrv_nonlinear()`

## 時域分析

### hrv_time()

基於心跳間隔（IBI）計算時域 HRV 指標。

```python
hrv_time = nk.hrv_time(peaks, sampling_rate=1000)
```

### 關鍵指標

**基本間隔統計：**
- `HRV_MeanNN`：NN 間隔平均值（毫秒）
- `HRV_SDNN`：NN 間隔標準差（毫秒）
  - 反映總 HRV，捕捉所有週期性成分
  - 短期需要 ≥5 分鐘，長期需要 ≥24 小時
- `HRV_RMSSD`：連續差異的均方根（毫秒）
  - 高頻變異，反映副交感神經活動
  - 較短記錄時更穩定

**連續差異測量：**
- `HRV_SDSD`：連續差異的標準差（毫秒）
  - 與 RMSSD 相似，與副交感神經活動相關
- `HRV_pNN50`：連續 NN 間隔差異 >50 毫秒的百分比
  - 副交感神經指標，在某些人群中可能不敏感
- `HRV_pNN20`：連續 NN 間隔差異 >20 毫秒的百分比
  - pNN50 的更敏感替代方案

**範圍測量：**
- `HRV_MinNN`、`HRV_MaxNN`：最小和最大 NN 間隔（毫秒）
- `HRV_CVNN`：變異係數（SDNN/MeanNN）
  - 標準化測量，適用於跨受試者比較
- `HRV_CVSD`：連續差異的變異係數（RMSSD/MeanNN）

**基於中位數的統計：**
- `HRV_MedianNN`：NN 間隔中位數（毫秒）
  - 對異常值穩健
- `HRV_MadNN`：NN 間隔的中位絕對偏差
  - 穩健的離散度測量
- `HRV_MCVNN`：基於中位數的變異係數

**進階時域：**
- `HRV_IQRNN`：NN 間隔的四分位距
- `HRV_pNN10`、`HRV_pNN25`、`HRV_pNN40`：額外的百分位閾值
- `HRV_TINN`：NN 間隔直方圖的三角內插
- `HRV_HTI`：HRV 三角指數（總 NN 間隔 / 直方圖高度）

### 記錄時長要求
- **超短期（< 5 分鐘）**：RMSSD、pNN50 最可靠
- **短期（5 分鐘）**：臨床使用標準，所有時域有效
- **長期（24 小時）**：SDNN 解釋所需，捕捉晝夜節律

## 頻域分析

### hrv_frequency()

使用頻譜分析分析各頻帶的 HRV 功率。

```python
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, ulf=(0, 0.0033), vlf=(0.0033, 0.04),
                            lf=(0.04, 0.15), hf=(0.15, 0.4), vhf=(0.4, 0.5),
                            psd_method='welch', normalize=True)
```

### 頻帶

**超低頻（ULF）：0-0.0033 Hz**
- 需要 ≥24 小時記錄
- 晝夜節律、體溫調節
- 緩慢代謝過程

**極低頻（VLF）：0.0033-0.04 Hz**
- 需要 ≥5 分鐘記錄
- 體溫調節、激素波動
- 腎素-血管緊張素系統、周邊血管運動活動

**低頻（LF）：0.04-0.15 Hz**
- 交感和副交感神經的混合影響
- 壓力感受器反射活動
- 血壓調節（10 秒節律）

**高頻（HF）：0.15-0.4 Hz**
- 副交感（迷走）神經活動
- 呼吸性竇性心律不齊
- 與呼吸同步（呼吸率範圍）

**極高頻（VHF）：0.4-0.5 Hz**
- 很少使用，可能反映測量噪音
- 需要謹慎解釋

### 關鍵指標

**絕對功率（ms²）：**
- `HRV_ULF`、`HRV_VLF`、`HRV_LF`、`HRV_HF`、`HRV_VHF`：各頻帶的功率
- `HRV_TP`：總功率（NN 間隔的變異數）
- `HRV_LFHF`：LF/HF 比率（交感迷走平衡）

**標準化功率：**
- `HRV_LFn`：LF 功率 / (LF + HF) - 標準化 LF
- `HRV_HFn`：HF 功率 / (LF + HF) - 標準化 HF
- `HRV_LnHF`：HF 的自然對數（對數常態分布）

**峰值頻率：**
- `HRV_LFpeak`、`HRV_HFpeak`：各頻帶最大功率的頻率
- 用於識別主導振盪

### 功率譜密度方法

**Welch 方法（預設）：**
```python
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, psd_method='welch')
```
- 帶重疊的窗口 FFT
- 更平滑的頻譜，減少變異
- 適合標準 HRV 分析

**Lomb-Scargle 週期圖：**
```python
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, psd_method='lomb')
```
- 處理不均勻取樣資料
- 不需要內插
- 更適合有噪音或含偽跡的資料

**多錐方法：**
```python
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, psd_method='multitapers')
```
- 優越的頻譜估計
- 以最小偏差減少變異
- 計算密集

**Burg 自迴歸：**
```python
hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, psd_method='burg', order=16)
```
- 參數方法
- 具有明確定義峰值的平滑頻譜
- 需要階數選擇

### 解釋指南

**LF/HF 比率：**
- 傳統上解釋為交感迷走平衡
- **注意**：近期證據質疑此解釋
- LF 反映交感和副交感神經的影響
- 情境依賴：控制呼吸會影響 HF

**HF 功率：**
- 可靠的副交感神經指標
- 增加：休息、放鬆、深呼吸
- 減少：壓力、焦慮、交感神經激活

**記錄要求：**
- **最低**：60 秒用於 LF/HF 估計
- **建議**：2-5 分鐘用於短期 HRV
- **最佳**：5 分鐘按任務小組標準
- **長期**：24 小時用於 ULF 分析

## 非線性域分析

### hrv_nonlinear()

計算反映自主神經動態的複雜度、熵和碎形測量。

```python
hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=1000)
```

### 龐加萊圖指標

**龐加萊圖（Poincare plot）**：NN(i+1) vs NN(i) 散佈圖幾何

- `HRV_SD1`：垂直於等同線的標準差（毫秒）
  - 短期 HRV，快速的逐拍變異
  - 反映副交感神經活動
  - 數學上與 RMSSD 相關：SD1 ≈ RMSSD/√2

- `HRV_SD2`：沿等同線的標準差（毫秒）
  - 長期 HRV，緩慢變異
  - 反映交感和副交感神經活動
  - 與 SDNN 相關

- `HRV_SD1SD2`：比率 SD1/SD2
  - 短期和長期變異之間的平衡
  - <1：主要是長期變異

- `HRV_SD2SD1`：比率 SD2/SD1
  - SD1SD2 的倒數

- `HRV_S`：橢圓面積（π × SD1 × SD2）
  - 總 HRV 大小

- `HRV_CSI`：心臟交感神經指數（SD2/SD1）
  - 提議的交感神經指標

- `HRV_CVI`：心臟迷走神經指數（log10(SD1 × SD2)）
  - 提議的副交感神經指標

- `HRV_CSI_Modified`：修改的 CSI（SD2²/(SD1 × SD2)）

### 心率不對稱性

分析心率加速和減速是否對 HRV 有不同的貢獻。

- `HRV_GI`：Guzik 指數 - 短期變異的不對稱性
- `HRV_SI`：斜率指數 - 長期變異的不對稱性
- `HRV_AI`：面積指數 - 整體不對稱性
- `HRV_PI`：Porta 指數 - 減速的百分比
- `HRV_C1d`、`HRV_C2d`：減速貢獻
- `HRV_C1a`、`HRV_C2a`：加速貢獻
- `HRV_SD1d`、`HRV_SD1a`：減速/加速的龐加萊 SD1
- `HRV_SD2d`、`HRV_SD2a`：減速/加速的龐加萊 SD2

**解釋：**
- 健康個體：存在不對稱性（更多/更大的減速）
- 臨床人群：不對稱性減少
- 反映加速 vs. 減速的差異性自主神經控制

### 熵測量

**近似熵（ApEn）：**
- `HRV_ApEn`：規律性測量，越低 = 越規律/可預測
- 對資料長度、階數 m、容差 r 敏感

**樣本熵（SampEn）：**
- `HRV_SampEn`：改進的 ApEn，較少依賴資料長度
- 短記錄時更一致
- 較低值 = 更規律的模式

**多尺度熵（MSE）：**
- `HRV_MSE`：跨多個時間尺度的複雜度
- 區分真正的複雜度和隨機性

**模糊熵：**
- `HRV_FuzzyEn`：用於模式匹配的模糊隸屬函數
- 短資料時更穩定

**Shannon 熵：**
- `HRV_ShanEn`：資訊論隨機性測量

### 碎形測量

**去趨勢波動分析（DFA）：**
- `HRV_DFA_alpha1`：短期碎形縮放指數（4-11 拍）
  - α1 > 1：相關性，心臟病時減少
  - α1 ≈ 1：粉紅噪音，健康
  - α1 < 0.5：反相關

- `HRV_DFA_alpha2`：長期碎形縮放指數（>11 拍）
  - 反映長程相關性

- `HRV_DFA_alpha1alpha2`：比率 α1/α2

**相關維度：**
- `HRV_CorDim`：相空間中吸引子的維度
- 表示系統複雜度

**Higuchi 碎形維度：**
- `HRV_HFD`：複雜度和自相似性
- 較高值 = 更複雜、不規則

**Petrosian 碎形維度：**
- `HRV_PFD`：替代複雜度測量
- 計算效率高

**Katz 碎形維度：**
- `HRV_KFD`：波形複雜度

### 心率碎片化

量化反映自主神經失調的異常短期波動。

- `HRV_PIP`：轉折點百分比
  - 正常：~50%，碎片化：>70%
- `HRV_IALS`：加速/減速片段平均長度的倒數
- `HRV_PSS`：短片段（<3 拍）的百分比
- `HRV_PAS`：交替片段中 NN 間隔的百分比

**臨床相關性：**
- 碎片化增加與心血管風險相關
- 超越傳統 HRV 指標的獨立預測因子

### 其他非線性指標

- `HRV_Hurst`：Hurst 指數（長程依賴性）
- `HRV_LZC`：Lempel-Ziv 複雜度（演算法複雜度）
- `HRV_MFDFA`：多重碎形 DFA 指標

## 專門 HRV 函數

### hrv_rsa()

呼吸性竇性心律不齊 - 呼吸對心率的調節。

```python
rsa = nk.hrv_rsa(peaks, rsp_signal, sampling_rate=1000, method='porges1980')
```

**方法：**
- `'porges1980'`：Porges-Bohrer 方法（圍繞呼吸頻率的帶通濾波 HR）
- `'harrison2021'`：峰谷 RSA（每個呼吸週期的最大-最小 HR）

**需求：**
- ECG 和呼吸訊號
- 同步時間
- 至少幾個呼吸週期

**回傳：**
- `RSA`：RSA 大小（次/分鐘或類似單位，取決於方法）

### hrv_rqa()

遞歸量化分析 - 來自相空間重建的非線性動態。

```python
rqa = nk.hrv_rqa(peaks, sampling_rate=1000)
```

**指標：**
- `RQA_RR`：遞歸率 - 系統可預測性
- `RQA_DET`：確定性 - 形成線的遞歸點百分比
- `RQA_LMean`、`RQA_LMax`：平均和最大對角線長度
- `RQA_ENTR`：線長度的 Shannon 熵 - 複雜度
- `RQA_LAM`：層流性 - 系統困在特定狀態
- `RQA_TT`：捕獲時間 - 層流狀態的持續時間

**應用場景：**
- 偵測生理狀態的轉換
- 評估系統確定性 vs. 隨機性

## 間隔處理

### intervals_process()

HRV 分析前預處理 RR 間隔。

```python
processed_intervals = nk.intervals_process(rr_intervals, interpolate=False,
                                           interpolate_sampling_rate=1000)
```

**操作：**
- 移除生理上不合理的間隔
- 選用：內插至規則取樣
- 選用：去趨勢以移除緩慢趨勢

**應用場景：**
- 處理預先提取的 RR 間隔
- 清理來自外部設備的間隔
- 為頻域分析準備資料

### intervals_to_peaks()

將間隔資料（RR、NN）轉換為 HRV 分析的峰值索引。

```python
peaks_dict = nk.intervals_to_peaks(rr_intervals, sampling_rate=1000)
```

**應用場景：**
- 從外部 HRV 設備匯入資料
- 處理來自商業系統的間隔資料
- 在間隔和峰值表示之間轉換

## 實務考量

### 最低記錄時長

| 分析 | 最低時長 | 最佳時長 |
|------|---------|---------|
| RMSSD、pNN50 | 30 秒 | 5 分鐘 |
| SDNN | 5 分鐘 | 5 分鐘（短期），24 小時（長期）|
| LF、HF 功率 | 2 分鐘 | 5 分鐘 |
| VLF 功率 | 5 分鐘 | 10+ 分鐘 |
| ULF 功率 | 24 小時 | 24 小時 |
| 非線性（ApEn、SampEn）| 100-300 拍 | 500+ 拍 |
| DFA | 300 拍 | 1000+ 拍 |

### 偽跡管理

**預處理：**
```python
# 使用偽跡校正偵測 R 波峰
peaks, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=1000, correct_artifacts=True)

# 或手動處理間隔
processed = nk.intervals_process(rr_intervals, interpolate=False)
```

**品質檢查：**
- 視覺檢視心搏圖（NN 間隔隨時間）
- 識別生理上不合理的間隔（<300 毫秒或 >2000 毫秒）
- 檢查突然跳躍或遺漏的心跳
- 分析前評估訊號品質

### 標準化和比較

**任務小組標準（1996）：**
- 短期使用 5 分鐘記錄
- 建議仰臥、控制呼吸
- 長期評估使用 24 小時

**標準化：**
- 考慮年齡、性別、體適能水平效應
- 一天中的時間和晝夜效應
- 身體姿勢（仰臥 vs. 站立）
- 呼吸率和深度

**個體間變異性：**
- HRV 在受試者之間有很大的變異
- 受試者內變化更可解釋
- 優先使用基線比較

## 臨床和研究應用

**心血管健康：**
- 降低的 HRV：心臟事件的風險因子
- SDNN、DFA alpha1：預後指標
- 心肌梗塞後監測

**心理狀態：**
- 焦慮/壓力：降低的 HRV（特別是 RMSSD、HF）
- 憂鬱：改變的自主神經平衡
- PTSD：碎片化指標

**運動表現：**
- 通過每日 RMSSD 監測訓練負荷
- 過度訓練：降低的 HRV
- 恢復評估

**神經科學：**
- 情緒調節研究
- 認知負荷評估
- 腦心軸研究

**老化：**
- HRV 隨年齡下降
- 複雜度測量下降
- 需要基線參考

## 參考文獻

- Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.
- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in public health, 5, 258.
- Peng, C. K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. Chaos, 5(1), 82-87.
- Guzik, P., Piskorski, J., Krauze, T., Wykretowicz, A., & Wysocki, H. (2006). Heart rate asymmetry by Poincaré plots of RR intervals. Biomedizinische Technik/Biomedical Engineering, 51(4), 272-275.
- Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy analysis of biological signals. Physical review E, 71(2), 021906.
