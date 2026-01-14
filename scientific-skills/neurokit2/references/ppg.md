# 光體積變化描記法（PPG）分析

## 概述

光體積變化描記法（Photoplethysmography，PPG）使用光學感測器測量微血管組織中的血容量變化。PPG 廣泛應用於穿戴式裝置、脈搏血氧計和臨床監測器，用於心率、脈搏特徵和心血管評估。

## 主要處理管線

### ppg_process()

自動化 PPG 訊號處理管線。

```python
signals, info = nk.ppg_process(ppg_signal, sampling_rate=100, method='elgendi')
```

**管線步驟：**
1. 訊號清理（濾波）
2. 收縮期波峰偵測
3. 心率計算
4. 訊號品質評估

**回傳：**
- `signals`：DataFrame 包含：
  - `PPG_Clean`：濾波後的 PPG 訊號
  - `PPG_Peaks`：收縮期波峰標記
  - `PPG_Rate`：瞬時心率（BPM）
  - `PPG_Quality`：訊號品質指標
- `info`：包含波峰索引和參數的字典

**方法：**
- `'elgendi'`：Elgendi et al. (2013) 演算法（預設，穩健）
- `'nabian2018'`：Nabian et al. (2018) 方法

## 預處理函數

### ppg_clean()

準備原始 PPG 訊號以進行波峰偵測。

```python
cleaned_ppg = nk.ppg_clean(ppg_signal, sampling_rate=100, method='elgendi')
```

**方法：**

**1. Elgendi（預設）：**
- Butterworth 帶通濾波器（0.5-8 Hz）
- 移除基線漂移和高頻噪音
- 針對波峰偵測可靠性最佳化

**2. Nabian2018：**
- 替代濾波方法
- 不同的頻率特性

**PPG 訊號特徵：**
- **收縮期波峰**：快速上升、尖銳波峰（心臟射血）
- **重搏切跡**：次級波峰（主動脈瓣關閉）
- **基線**：因呼吸、運動、灌注而緩慢漂移

### ppg_peaks()

在 PPG 訊號中偵測收縮期波峰。

```python
peaks, info = nk.ppg_peaks(cleaned_ppg, sampling_rate=100, method='elgendi',
                           correct_artifacts=False)
```

**方法：**
- `'elgendi'`：帶動態閾值的雙移動平均
- `'bishop'`：Bishop 演算法
- `'nabian2018'`：Nabian 方法
- `'scipy'`：簡單的 scipy 波峰偵測

**偽跡校正：**
- 設定 `correct_artifacts=True` 進行生理合理性檢查
- 根據心跳間隔異常值移除虛假波峰

**回傳：**
- 包含 `'PPG_Peaks'` 鍵的字典，含波峰索引

**典型心跳間隔：**
- 靜息成人：600-1200 毫秒（50-100 BPM）
- 運動員：可能更長（心搏過緩）
- 壓力/運動中：較短（<600 毫秒，>100 BPM）

### ppg_findpeaks()

帶演算法比較的低階波峰偵測。

```python
peaks_dict = nk.ppg_findpeaks(cleaned_ppg, sampling_rate=100, method='elgendi')
```

**應用場景：**
- 自訂參數調整
- 演算法測試
- 研究方法開發

## 分析函數

### ppg_analyze()

自動選擇事件相關或區間相關分析。

```python
analysis = nk.ppg_analyze(signals, sampling_rate=100)
```

**模式選擇：**
- 時長 < 10 秒 → 事件相關
- 時長 ≥ 10 秒 → 區間相關

### ppg_eventrelated()

分析 PPG 對離散事件/刺激的反應。

```python
results = nk.ppg_eventrelated(epochs)
```

**計算的指標（每時段）：**
- `PPG_Rate_Baseline`：事件前的心率
- `PPG_Rate_Min/Max`：時段期間的最小/最大心率
- 跨時段時間視窗的心率動態

**應用場景：**
- 對情緒刺激的心血管反應
- 認知負荷評估
- 壓力反應性範式

### ppg_intervalrelated()

分析延長的 PPG 記錄。

```python
results = nk.ppg_intervalrelated(signals, sampling_rate=100)
```

**計算的指標：**
- `PPG_Rate_Mean`：平均心率
- 心率變異性（HRV）指標
  - 委派給 `hrv()` 函數
  - 時域、頻域和非線性域

**記錄時長：**
- 最低：60 秒用於基本心率
- HRV 分析：建議 2-5 分鐘

**應用場景：**
- 靜息狀態心血管評估
- 穿戴式裝置資料分析
- 長期心率監測

## 品質評估

### ppg_quality()

評估訊號品質和可靠性。

```python
quality = nk.ppg_quality(ppg_signal, sampling_rate=100, method='averageQRS')
```

**方法：**

**1. averageQRS（預設）：**
- 模板匹配方法
- 將每個脈搏與平均模板相關
- 回傳每拍 0-1 的品質分數
- 閾值：>0.6 = 可接受品質

**2. dissimilarity：**
- 地形差異性測量
- 偵測形態變化

**應用場景：**
- 識別損壞的片段
- 分析前過濾低品質資料
- 驗證波峰偵測準確性

**常見品質問題：**
- 動作偽跡：突然的訊號變化
- 感測器接觸不良：低振幅、噪音
- 血管收縮：訊號振幅降低（寒冷、壓力）

## 工具函數

### ppg_segment()

提取個別脈搏用於形態分析。

```python
pulses = nk.ppg_segment(cleaned_ppg, peaks, sampling_rate=100)
```

**回傳：**
- 脈搏時段字典，每個以收縮期波峰為中心
- 支援脈搏對脈搏比較
- 跨條件的形態分析

**應用場景：**
- 脈搏波分析
- 動脈硬度代理指標
- 血管老化評估

### ppg_methods()

記錄分析中使用的預處理方法。

```python
methods_info = nk.ppg_methods(method='elgendi')
```

**回傳：**
- 記錄處理管線的字串
- 適用於出版物的方法章節

## 模擬和視覺化

### ppg_simulate()

生成用於測試的合成 PPG 訊號。

```python
synthetic_ppg = nk.ppg_simulate(duration=60, sampling_rate=100, heart_rate=70,
                                noise=0.1, random_state=42)
```

**參數：**
- `heart_rate`：平均 BPM（預設：70）
- `heart_rate_std`：HRV 大小
- `noise`：高斯噪音水平
- `random_state`：可重現性種子

**應用場景：**
- 演算法驗證
- 參數最佳化
- 教育示範

### ppg_plot()

視覺化處理後的 PPG 訊號。

```python
nk.ppg_plot(signals, info, static=True)
```

**顯示：**
- 原始和清理後的 PPG 訊號
- 偵測到的收縮期波峰
- 瞬時心率軌跡
- 訊號品質指標

## 實務考量

### 取樣率建議
- **最低**：20 Hz（基本心率）
- **標準**：50-100 Hz（大多數穿戴式裝置）
- **高解析度**：200-500 Hz（研究、脈搏波分析）
- **過高**：>1000 Hz（對 PPG 不必要）

### 記錄時長
- **心率**：≥10 秒（幾拍）
- **HRV 分析**：最少 2-5 分鐘
- **長期監測**：數小時至數天（穿戴式裝置）

### 感測器放置

**常見部位：**
- **指尖**：最高訊號品質，最常見
- **耳垂**：動作偽跡較少，臨床使用
- **手腕**：穿戴式裝置（智慧手錶）
- **前額**：反射模式，醫療監測

**透射 vs. 反射：**
- **透射**：光通過組織（指尖、耳垂）
  - 較高訊號品質
  - 較少動作偽跡
- **反射**：光從組織反射（手腕、前額）
  - 對噪音更敏感
  - 適合穿戴式裝置

### 常見問題和解決方案

**低訊號振幅：**
- 灌注不良：溫暖雙手，增加血流
- 感測器接觸：調整放置，清潔皮膚
- 血管收縮：環境溫度、壓力

**動作偽跡：**
- 穿戴式裝置的主要問題
- 自適應濾波、基於加速度計的校正
- 模板匹配、異常值拒絕

**基線漂移：**
- 呼吸調節（正常）
- 運動或壓力變化
- 高通濾波或去趨勢

**遺漏波峰：**
- 低品質訊號：檢查感測器接觸
- 演算法參數：調整閾值
- 嘗試替代偵測方法

### 最佳實踐

**標準工作流程：**
```python
# 1. 清理訊號
cleaned = nk.ppg_clean(ppg_raw, sampling_rate=100, method='elgendi')

# 2. 使用偽跡校正偵測波峰
peaks, info = nk.ppg_peaks(cleaned, sampling_rate=100, correct_artifacts=True)

# 3. 評估品質
quality = nk.ppg_quality(cleaned, sampling_rate=100)

# 4. 完整處理（替代方案）
signals, info = nk.ppg_process(ppg_raw, sampling_rate=100)

# 5. 分析
analysis = nk.ppg_analyze(signals, sampling_rate=100)
```

**從 PPG 計算 HRV：**
```python
# 處理 PPG 訊號
signals, info = nk.ppg_process(ppg_raw, sampling_rate=100)

# 提取波峰並計算 HRV
hrv_indices = nk.hrv(info['PPG_Peaks'], sampling_rate=100)

# PPG 衍生的 HRV 有效但可能與 ECG 衍生的 HRV 略有不同
# 差異來自脈搏到達時間、血管特性
```

## 臨床和研究應用

**穿戴式健康監測：**
- 消費者智慧手錶和健身追蹤器
- 連續心率監測
- 睡眠追蹤和活動評估

**臨床監測：**
- 脈搏血氧計（SpO₂ + 心率）
- 圍手術期監測
- 重症監護心率評估

**心血管評估：**
- 脈搏波分析
- 動脈硬度代理指標（脈搏到達時間）
- 血管老化指數

**自主神經功能：**
- 來自 PPG 的 HRV（PPG-HRV）
- 壓力和恢復監測
- 心理工作負荷評估

**遠端病患監測：**
- 遠距醫療應用
- 家庭健康追蹤
- 慢性疾病管理

**情感運算：**
- 從生理訊號識別情緒
- 使用者體驗研究
- 人機互動

## PPG vs. ECG

**PPG 的優點：**
- 非侵入性，無電極
- 適合長期監測
- 低成本，可小型化
- 適合穿戴式裝置

**PPG 的缺點：**
- 對動作偽跡更敏感
- 灌注不良時訊號品質較低
- 與心臟的脈搏到達時間延遲
- 無法評估心臟電活動

**HRV 比較：**
- PPG-HRV 在時域/頻域通常有效
- 可能因脈搏傳遞時間變異而略有不同
- 可用時優先使用 ECG 進行臨床 HRV
- PPG 可用於研究和消費者應用

## 解讀指南

**來自 PPG 的心率：**
- 與 ECG 衍生心率的解讀相同
- 輕微延遲（脈搏到達時間）對心率計算可忽略
- 動作偽跡更常見：用訊號品質驗證

**脈搏振幅：**
- 反映周邊灌注
- 增加：血管舒張、溫暖
- 減少：血管收縮、寒冷、壓力、接觸不良

**脈搏形態：**
- 收縮期波峰：心臟射血
- 重搏切跡：主動脈瓣關閉、動脈順應性
- 老化/硬化：較早、更明顯的重搏切跡

## 參考文獻

- Elgendi, M. (2012). On the analysis of fingertip photoplethysmogram signals. Current cardiology reviews, 8(1), 14-25.
- Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D. (2013). Systolic peak detection in acceleration photoplethysmograms measured from emergency responders in tropical conditions. PloS one, 8(10), e76585.
- Allen, J. (2007). Photoplethysmography and its application in clinical physiological measurement. Physiological measurement, 28(3), R1.
- Tamura, T., Maeda, Y., Sekine, M., & Yoshida, M. (2014). Wearable photoplethysmographic sensors—past and present. Electronics, 3(2), 282-302.
