# 眼電圖（EOG）分析

## 概述

眼電圖（EOG）透過檢測眼位變化產生的電位差來測量眼動和眨眼。EOG 用於睡眠研究、注意力研究、閱讀分析和 EEG 偽影校正。

## 主要處理管線

### eog_process()

自動化 EOG 訊號處理管線。

```python
signals, info = nk.eog_process(eog_signal, sampling_rate=500, method='neurokit')
```

**管線步驟：**
1. 訊號清理（濾波）
2. 眨眼檢測
3. 眨眼頻率計算

**返回：**
- `signals`：包含以下內容的 DataFrame：
  - `EOG_Clean`：濾波後的 EOG 訊號
  - `EOG_Blinks`：二元眨眼標記（0/1）
  - `EOG_Rate`：瞬時眨眼頻率（次/分鐘）
- `info`：包含眨眼索引和參數的字典

**方法：**
- `'neurokit'`：NeuroKit2 最佳化方法（預設）
- `'agarwal2019'`：Agarwal et al. (2019) 演算法
- `'mne'`：MNE-Python 方法
- `'brainstorm'`：Brainstorm 工具箱方法
- `'kong1998'`：Kong et al. (1998) 方法

## 預處理函數

### eog_clean()

準備原始 EOG 訊號用於眨眼檢測。

```python
cleaned_eog = nk.eog_clean(eog_signal, sampling_rate=500, method='neurokit')
```

**方法：**
- `'neurokit'`：針對 EOG 最佳化的 Butterworth 濾波
- `'agarwal2019'`：替代濾波
- `'mne'`：MNE-Python 預處理
- `'brainstorm'`：Brainstorm 方法
- `'kong1998'`：Kong 方法

**典型濾波：**
- 低通：10-20 Hz（去除高頻雜訊）
- 高通：0.1-1 Hz（去除直流漂移）
- 保留眨眼波形（典型持續時間 100-400 ms）

**EOG 訊號特徵：**
- **眨眼**：大振幅、定型波形（200-400 ms）
- **眼跳**：快速階梯狀偏轉（20-80 ms）
- **平滑追蹤**：緩慢斜坡狀變化
- **基線**：眼睛固定時穩定

## 眨眼檢測

### eog_peaks()

檢測 EOG 訊號中的眨眼。

```python
blinks, info = nk.eog_peaks(cleaned_eog, sampling_rate=500, method='neurokit',
                            threshold=0.33)
```

**方法：**
- `'neurokit'`：振幅和持續時間準則（預設）
- `'mne'`：MNE-Python 眨眼檢測
- `'brainstorm'`：Brainstorm 方法
- `'blinker'`：BLINKER 演算法（Kleifges et al., 2017）

**關鍵參數：**
- `threshold`：振幅閾值（最大振幅的分數）
  - 典型：0.2-0.5
  - 較低：更敏感（可能包含偽陽性）
  - 較高：更保守（可能遺漏小眨眼）

**返回：**
- 包含 `'EOG_Blinks'` 鍵的字典，包含眨眼峰值索引

**眨眼特徵：**
- **頻率**：15-20 次/分鐘（靜息、舒適）
- **持續時間**：100-400 ms（平均 ~200 ms）
- **振幅**：隨電極放置和個體因素變化
- **波形**：雙相或三相

### eog_findpeaks()

多演算法的低階眨眼檢測。

```python
blinks_dict = nk.eog_findpeaks(cleaned_eog, sampling_rate=500, method='neurokit')
```

**使用情境：**
- 自訂參數調整
- 演算法比較
- 研究方法開發

## 特徵提取

### eog_features()

提取單個眨眼的特徵。

```python
features = nk.eog_features(signals, sampling_rate=500)
```

**計算的特徵：**
- **振幅速度比（AVR）**：峰值速度 / 振幅
  - 區分眨眼和偽影
- **眨眼振幅比**：眨眼振幅的一致性
- **持續時間指標**：眨眼持續時間統計（平均值、SD）
- **峰值振幅**：最大偏轉
- **峰值速度**：最大變化率

**使用情境：**
- 眨眼品質評估
- 嗜睡檢測（嗜睡時眨眼持續時間增加）
- 神經系統評估（疾病中眨眼動力學改變）

### eog_rate()

計算眨眼頻率（每分鐘眨眼次數）。

```python
blink_rate = nk.eog_rate(blinks, sampling_rate=500, desired_length=None)
```

**方法：**
- 計算眨眼間隔
- 轉換為每分鐘眨眼次數
- 插值以匹配訊號長度

**典型眨眼頻率：**
- **靜息**：15-20 次/分鐘
- **閱讀/視覺任務**：5-10 次/分鐘（被抑制）
- **對話**：20-30 次/分鐘
- **壓力/乾眼**：>30 次/分鐘
- **嗜睡**：變異性高、眨眼較長

## 分析函數

### eog_analyze()

自動選擇事件相關或區間相關分析。

```python
analysis = nk.eog_analyze(signals, sampling_rate=500)
```

**模式選擇：**
- 持續時間 < 10 秒 → 事件相關
- 持續時間 ≥ 10 秒 → 區間相關

### eog_eventrelated()

分析相對於特定事件的眨眼模式。

```python
results = nk.eog_eventrelated(epochs)
```

**計算的指標（每個分段）：**
- `EOG_Blinks_N`：分段期間的眨眼次數
- `EOG_Rate_Mean`：平均眨眼頻率
- `EOG_Blink_Presence`：二元（是否發生任何眨眼）
- 跨分段的眨眼時間分佈

**使用情境：**
- 眨眼鎖定 ERP 污染評估
- 刺激期間的注意力和投入度
- 視覺任務難度（要求高的任務期間眨眼被抑制）
- 刺激結束後的自發眨眼

### eog_intervalrelated()

分析延長時期的眨眼模式。

```python
results = nk.eog_intervalrelated(signals, sampling_rate=500)
```

**計算的指標：**
- `EOG_Blinks_N`：眨眼總數
- `EOG_Rate_Mean`：平均眨眼頻率（次/分鐘）
- `EOG_Rate_SD`：眨眼頻率變異性
- `EOG_Duration_Mean`：平均眨眼持續時間（如可用）
- `EOG_Amplitude_Mean`：平均眨眼振幅（如可用）

**使用情境：**
- 靜息狀態眨眼模式
- 嗜睡或疲勞監測（持續時間增加）
- 持續注意力任務（頻率被抑制）
- 乾眼症評估（頻率增加、眨眼不完整）

## 模擬和視覺化

### eog_plot()

視覺化處理過的 EOG 訊號和檢測到的眨眼。

```python
nk.eog_plot(signals, info)
```

**顯示：**
- 原始和清理後的 EOG 訊號
- 檢測到的眨眼標記
- 眨眼頻率時間過程

## 實務考量

### 取樣率建議
- **最低**：100 Hz（基本眨眼檢測）
- **標準**：250-500 Hz（研究應用）
- **高解析度**：1000 Hz（詳細波形分析、眼跳）
- **睡眠研究**：典型 200-250 Hz

### 記錄持續時間
- **眨眼檢測**：任何持續時間（≥1 次眨眼）
- **眨眼頻率估計**：≥60 秒以獲得穩定估計
- **事件相關**：依範式而定（每試驗秒數）
- **睡眠 EOG**：小時（整夜）

### 電極放置

**標準配置：**

**水平 EOG（HEOG）：**
- 兩個電極：左右眼外眥（外角）
- 測量水平眼動（眼跳、平滑追蹤）
- 雙極記錄（左 - 右）

**垂直 EOG（VEOG）：**
- 兩個電極：一眼上方和下方（通常右眼）
- 測量垂直眼動和眨眼
- 雙極記錄（上 - 下）

**睡眠 EOG：**
- 常使用稍不同的放置（太陽穴區域）
- E1：左眼外眥外側 1 cm 且下方 1 cm
- E2：右眼外眥外側 1 cm 且上方 1 cm
- 捕捉水平和垂直運動

**EEG 污染去除：**
- 額葉電極（Fp1、Fp2）可作為 EOG 代理
- 基於 ICA 的 EOG 偽影去除在 EEG 預處理中很常見

### 常見問題和解決方案

**電極問題：**
- 接觸不良：低振幅、雜訊
- 皮膚準備：清潔、輕微磨皮
- 導電凝膠：確保良好接觸

**偽影：**
- 肌肉活動（尤其是額肌）：高頻雜訊
- 運動：電纜偽影、頭部運動
- 電力雜訊：50/60 Hz 哼聲（正確接地）

**飽和：**
- 大眼跳可能使放大器飽和
- 調整增益或電壓範圍
- 低解析度系統更常見

### 最佳實踐

**標準工作流程：**
```python
# 1. 清理訊號
cleaned = nk.eog_clean(eog_raw, sampling_rate=500, method='neurokit')

# 2. 檢測眨眼
blinks, info = nk.eog_peaks(cleaned, sampling_rate=500, method='neurokit')

# 3. 提取特徵
features = nk.eog_features(signals, sampling_rate=500)

# 4. 全面處理（替代方案）
signals, info = nk.eog_process(eog_raw, sampling_rate=500)

# 5. 分析
analysis = nk.eog_analyze(signals, sampling_rate=500)
```

**EEG 偽影校正工作流程：**
```python
# 選項 1：迴歸法去除
# 從清理的 EOG 訊號識別 EOG 成分
# 從 EEG 通道迴歸去除 EOG

# 選項 2：ICA 法去除（首選）
# 1. 對包含 EOG 通道的 EEG 資料運行 ICA
# 2. 識別與 EOG 相關的 ICA 成分
# 3. 從 EEG 資料去除 EOG 成分
# NeuroKit2 與 MNE 整合用於此工作流程
```

## 臨床和研究應用

**EEG 偽影校正：**
- 眨眼污染額葉 EEG 通道
- ICA 或迴歸方法去除 EOG 偽影
- 對 ERP 研究至關重要

**睡眠分期：**
- REM 睡眠期間的快速眼動（REMs）
- 嗜睡期間的緩慢滾動眼動
- 睡眠開始和階段轉換

**注意力和認知負荷：**
- 要求高的任務期間眨眼頻率被抑制
- 眨眼在任務邊界聚集（自然斷點）
- 自發眨眼作為注意力轉移的指標

**疲勞和嗜睡監測：**
- 嗜睡時眨眼持續時間增加
- 眼瞼閉合變慢
- 部分或不完整眨眼
- 駕駛員監測應用

**閱讀和視覺處理：**
- 閱讀期間眨眼被抑制
- 眼跳期間的眼動（換行）
- 疲勞對閱讀效率的影響

**神經系統疾病：**
- **帕金森氏症**：自發眨眼頻率降低
- **思覺失調症**：眨眼頻率增加
- **妥瑞症**：過度眨眼（抽動）
- **乾眼症候群**：眨眼增加、不完整

**情感和社會認知：**
- 社交互動中的眨眼同步
- 眨眼頻率的情緒調節
- ERP 中的眨眼相關電位

**人機互動：**
- 凝視追蹤預處理
- 注意力監測
- 使用者投入度評估

## EOG 可檢測的眼動類型

**眨眼：**
- 大振幅、短持續時間（100-400 ms）
- NeuroKit2 主要焦點
- 垂直 EOG 最敏感

**眼跳：**
- 快速、彈道式眼動（20-80 ms）
- 階梯狀電壓偏轉
- 水平或垂直
- 需要較高取樣率進行詳細分析

**平滑追蹤：**
- 緩慢追蹤移動物體
- 斜坡狀電壓變化
- 振幅比眼跳低

**注視：**
- 穩定凝視
- 帶小振盪的基線 EOG
- 持續時間變化（閱讀中典型 200-600 ms）

**注意：**詳細的眼跳/注視分析通常需要眼動追蹤（紅外線、視訊式）。EOG 對眨眼和粗眼動很有用。

## 解讀指南

**眨眼頻率：**
- **正常靜息**：15-20 次/分鐘
- **<10 次/分鐘**：視覺任務投入、專注
- **>30 次/分鐘**：壓力、乾眼、疲勞
- **依情境而定**：任務要求、照明、螢幕使用

**眨眼持續時間：**
- **正常**：100-400 ms（平均 ~200 ms）
- **延長**：嗜睡、疲勞（>500 ms）
- **短暫**：正常警覺

**眨眼振幅：**
- 隨電極放置和個體變化
- 受試者內比較最可靠
- 不完整眨眼：振幅降低（乾眼、疲勞）

**時間模式：**
- **聚集眨眼**：任務或認知狀態之間的轉換
- **抑制眨眼**：活躍視覺處理、持續注意力
- **刺激後眨眼**：視覺處理完成後

## 參考文獻

- Kleifges, K., Bigdely-Shamlo, N., Kerick, S. E., & Robbins, K. A. (2017). BLINKER: Automated extraction of ocular indices from EEG enabling large-scale analysis. Frontiers in Neuroscience, 11, 12.
- Agarwal, M., & Sivakumar, R. (2019). Blink: A fully automated unsupervised algorithm for eye-blink detection in EEG signals. In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (pp. 1113-1121). IEEE.
- Kong, X., & Wilson, G. F. (1998). A new EOG-based eyeblink detection algorithm. Behavior Research Methods, Instruments, & Computers, 30(4), 713-719.
- Schleicher, R., Galley, N., Briest, S., & Galley, L. (2008). Blinks and saccades as indicators of fatigue in sleepiness warnings: Looking tired? Ergonomics, 51(7), 982-1010.
