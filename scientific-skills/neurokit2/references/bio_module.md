# 多訊號整合（Bio 模組）

## 概述

Bio 模組提供統一的函數，用於同時處理和分析多個生理訊號。它作為一個包裝器，協調特定訊號的處理函數，並實現整合的多模態分析。

## 多訊號處理

### bio_process()

使用單一函數調用同時處理多個生理訊號。

```python
bio_signals, bio_info = nk.bio_process(ecg=None, rsp=None, eda=None, emg=None,
                                       ppg=None, eog=None, sampling_rate=1000)
```

**參數：**
- `ecg`：ECG 訊號陣列（可選）
- `rsp`：呼吸訊號陣列（可選）
- `eda`：EDA 訊號陣列（可選）
- `emg`：EMG 訊號陣列（可選）
- `ppg`：PPG 訊號陣列（可選）
- `eog`：EOG 訊號陣列（可選）
- `sampling_rate`：取樣率（Hz）（必須在各訊號間保持一致或按訊號指定）

**返回：**
- `bio_signals`：包含所有處理過訊號的統一 DataFrame，欄位包括：
  - 特定訊號特徵（例如 `ECG_Clean`、`ECG_Rate`、`EDA_Phasic`、`RSP_Rate`）
  - 所有檢測到的事件/波峰
  - 衍生測量
- `bio_info`：包含特定訊號資訊（波峰位置、參數）的字典

**範例：**
```python
# 同時處理 ECG、呼吸和 EDA
bio_signals, bio_info = nk.bio_process(
    ecg=ecg_signal,
    rsp=rsp_signal,
    eda=eda_signal,
    sampling_rate=1000
)

# 存取處理過的訊號
ecg_clean = bio_signals['ECG_Clean']
rsp_rate = bio_signals['RSP_Rate']
eda_phasic = bio_signals['EDA_Phasic']

# 存取檢測到的波峰
ecg_peaks = bio_info['ECG']['ECG_R_Peaks']
rsp_peaks = bio_info['RSP']['RSP_Peaks']
```

**內部工作流程：**
1. 每個訊號由其專用處理函數處理：
   - `ecg_process()` 用於 ECG
   - `rsp_process()` 用於呼吸
   - `eda_process()` 用於 EDA
   - `emg_process()` 用於 EMG
   - `ppg_process()` 用於 PPG
   - `eog_process()` 用於 EOG
2. 結果合併到統一的 DataFrame
3. 計算跨訊號特徵（例如，如果 ECG 和 RSP 都存在則計算 RSA）

**優點：**
- 簡化多模態記錄的 API
- 所有訊號的統一時間基準
- 自動跨訊號特徵計算
- 一致的輸出格式

## 多訊號分析

### bio_analyze()

對處理過的多模態訊號執行全面分析。

```python
bio_results = nk.bio_analyze(bio_signals, sampling_rate=1000)
```

**參數：**
- `bio_signals`：來自 `bio_process()` 的 DataFrame 或自訂處理的訊號
- `sampling_rate`：取樣率（Hz）

**返回：**
- 包含所有檢測到訊號類型分析結果的 DataFrame：
  - 如果持續時間 ≥ 10 秒則為區間相關指標
  - 如果持續時間 < 10 秒則為事件相關指標
  - 跨訊號指標（例如，如果 ECG + RSP 可用則計算 RSA）

**按訊號計算的指標：**
- **ECG**：心率統計、HRV 指標（時域、頻域、非線性域）
- **RSP**：呼吸頻率統計、RRV、振幅測量
- **EDA**：SCR 計數、振幅、緊張性水準、交感神經指標
- **EMG**：激活計數、振幅統計
- **PPG**：類似 ECG（心率、HRV）
- **EOG**：眨眼計數、眨眼頻率

**跨訊號指標：**
- **RSA（呼吸性竇性心律不整）**：如果 ECG + RSP 存在
- **心肺耦合**：相位同步指標
- **多模態喚醒**：組合自主神經指標

**範例：**
```python
# 分析處理過的訊號
results = nk.bio_analyze(bio_signals, sampling_rate=1000)

# 存取結果
heart_rate_mean = results['ECG_Rate_Mean']
hrv_rmssd = results['HRV_RMSSD']
breathing_rate = results['RSP_Rate_Mean']
scr_count = results['SCR_Peaks_N']
rsa_value = results['RSA']  # 如果 ECG 和 RSP 都存在
```

## 跨訊號特徵

當多個訊號一起處理時，NeuroKit2 可以計算整合特徵：

### 呼吸性竇性心律不整（RSA）

當 ECG 和呼吸訊號都存在時自動計算。

```python
bio_signals, bio_info = nk.bio_process(ecg=ecg, rsp=rsp, sampling_rate=1000)
results = nk.bio_analyze(bio_signals, sampling_rate=1000)

rsa = results['RSA']  # 自動包含
```

**計算：**
- 呼吸對高頻 HRV 的調節
- 需要同步的 ECG R 波峰和呼吸訊號
- 方法：Porges-Bohrer 或峰谷法

**解讀：**
- 較高的 RSA：較大的副交感神經（迷走神經）影響
- 心肺耦合的標記
- 健康指標和情緒調節能力

### ECG 衍生呼吸（EDR）

如果呼吸訊號不可用，NeuroKit2 可以從 ECG 估計：

```python
ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=1000)

# 提取 EDR
edr = nk.ecg_rsp(ecg_signals['ECG_Clean'], sampling_rate=1000)
```

**使用情境：**
- 當直接測量不可用時估計呼吸
- 交叉驗證呼吸測量

### 心臟-EDA 整合

同步的心臟和皮膚電活動：

```python
bio_signals, bio_info = nk.bio_process(ecg=ecg, eda=eda, sampling_rate=1000)

# 兩個訊號都可用於整合分析
ecg_rate = bio_signals['ECG_Rate']
eda_phasic = bio_signals['EDA_Phasic']

# 計算相關性或耦合指標
correlation = ecg_rate.corr(eda_phasic)
```

## 實用工作流程

### 完整多模態記錄分析

```python
import neurokit2 as nk
import pandas as pd

# 1. 載入多模態生理資料
ecg = load_ecg()        # 您的資料載入函數
rsp = load_rsp()
eda = load_eda()
emg = load_emg()

# 2. 同時處理所有訊號
bio_signals, bio_info = nk.bio_process(
    ecg=ecg,
    rsp=rsp,
    eda=eda,
    emg=emg,
    sampling_rate=1000
)

# 3. 視覺化處理過的訊號
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# ECG
axes[0].plot(bio_signals.index / 1000, bio_signals['ECG_Clean'])
axes[0].set_ylabel('ECG')
axes[0].set_title('Multi-Modal Physiological Recording')

# 呼吸
axes[1].plot(bio_signals.index / 1000, bio_signals['RSP_Clean'])
axes[1].set_ylabel('Respiration')

# EDA
axes[2].plot(bio_signals.index / 1000, bio_signals['EDA_Phasic'])
axes[2].set_ylabel('EDA (Phasic)')

# EMG
axes[3].plot(bio_signals.index / 1000, bio_signals['EMG_Amplitude'])
axes[3].set_ylabel('EMG Amplitude')
axes[3].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()

# 4. 分析所有訊號
results = nk.bio_analyze(bio_signals, sampling_rate=1000)

# 5. 提取關鍵指標
print("Heart Rate (mean):", results['ECG_Rate_Mean'])
print("HRV (RMSSD):", results['HRV_RMSSD'])
print("Breathing Rate:", results['RSP_Rate_Mean'])
print("SCRs (count):", results['SCR_Peaks_N'])
print("RSA:", results['RSA'])
```

### 事件相關多模態分析

```python
# 1. 處理訊號
bio_signals, bio_info = nk.bio_process(ecg=ecg, rsp=rsp, eda=eda, sampling_rate=1000)

# 2. 檢測事件
events = nk.events_find(trigger_channel, threshold=0.5)

# 3. 為所有訊號建立分段
epochs = nk.epochs_create(bio_signals, events, sampling_rate=1000,
                          epochs_start=-1.0, epochs_end=10.0,
                          event_labels=event_labels,
                          event_conditions=event_conditions)

# 4. 特定訊號的事件相關分析
ecg_eventrelated = nk.ecg_eventrelated(epochs)
rsp_eventrelated = nk.rsp_eventrelated(epochs)
eda_eventrelated = nk.eda_eventrelated(epochs)

# 5. 合併結果
all_results = pd.merge(ecg_eventrelated, rsp_eventrelated,
                       left_index=True, right_index=True)
all_results = pd.merge(all_results, eda_eventrelated,
                       left_index=True, right_index=True)

# 6. 按條件統計比較
all_results['Condition'] = event_conditions
condition_means = all_results.groupby('Condition').mean()
```

### 不同取樣率

處理具有不同原生取樣率的訊號：

```python
# ECG 為 1000 Hz，EDA 為 100 Hz
bio_signals, bio_info = nk.bio_process(
    ecg=ecg_1000hz,
    eda=eda_100hz,
    sampling_rate=1000  # 目標取樣率
)
# EDA 將在內部自動重取樣到 1000 Hz
```

或分別處理後合併：

```python
# 以原生取樣率處理
ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=1000)
eda_signals, eda_info = nk.eda_process(eda, sampling_rate=100)

# 重取樣到共同速率
eda_resampled = nk.signal_resample(eda_signals, sampling_rate=100,
                                   desired_sampling_rate=1000)

# 手動合併
bio_signals = pd.concat([ecg_signals, eda_resampled], axis=1)
```

## 使用情境和應用

### 全面心理生理學研究

捕捉生理喚醒的多個維度：

- **心臟**：定向、注意力、情緒效價
- **呼吸**：喚醒、壓力、情緒調節
- **EDA**：交感神經喚醒、情緒強度
- **EMG**：肌肉緊張、面部表情、驚嚇

**範例：情緒圖片觀看**
- ECG：圖片觀看期間心率減速（注意力）
- EDA：SCR 反映情緒喚醒強度
- RSP：屏息或變化反映情緒投入
- 面部 EMG：皺眉肌（皺眉）、顴大肌（微笑）用於效價

### 壓力和放鬆評估

多模態標記提供收斂證據：

- **壓力增加**：↑ HR、↓ HRV、↑ EDA、↑ 呼吸頻率、↑ 肌肉緊張
- **放鬆**：↓ HR、↑ HRV、↓ EDA、↓ 呼吸頻率、慢呼吸、↓ 肌肉緊張

**介入效果：**
- 比較介入前後的多模態指標
- 識別哪些模態對特定技術有反應

### 臨床評估

**焦慮症：**
- 基線 EDA、HR 升高
- 對壓力源的誇大反應
- HRV、呼吸變異性降低

**憂鬱症：**
- 自主神經平衡改變（↓ HRV）
- EDA 反應遲鈍
- 呼吸模式不規則

**PTSD：**
- 過度警覺（↑ HR、↑ EDA 基線）
- 誇大的驚嚇反應（EMG）
- RSA 改變

### 人機互動

非侵入式使用者狀態評估：

- **認知負荷**：↓ HRV、↑ EDA、眨眼抑制
- **挫折**：↑ HR、↑ EDA、↑ 肌肉緊張
- **投入**：適度喚醒、同步反應
- **無聊**：低喚醒、不規則模式

### 運動表現和恢復

監測訓練負荷和恢復：

- **靜息 HRV**：每日監測過度訓練
- **EDA**：交感神經激活和壓力
- **呼吸**：運動/恢復期間的呼吸模式
- **多模態整合**：全面的恢復評估

## 多模態記錄的優點

**收斂效度：**
- 多個指標收斂到同一構念（例如喚醒）
- 比單一測量更穩健

**區辨效度：**
- 不同訊號在某些條件下分離
- ECG 反映交感神經和副交感神經
- EDA 主要反映交感神經

**系統整合：**
- 了解全身生理協調
- 跨訊號耦合指標（RSA、相干性）

**冗餘和穩健性：**
- 如果一個訊號品質差，其他訊號可用
- 跨模態交叉驗證發現

**更豐富的解讀：**
- HR 減速 + SCR 增加 = 伴隨喚醒的定向
- HR 加速 + 無 SCR = 無交感神經喚醒的心臟反應

## 注意事項

### 硬體和同步

- **同一設備**：訊號固有同步
- **不同設備**：需要共同觸發/時間戳
  - 使用硬體觸發標記同時事件
  - 基於事件標記的軟體對齊
  - 驗證同步品質（交叉相關冗餘訊號）

### 跨模態訊號品質

- 並非所有訊號都可能具有相同品質
- 根據研究問題優先排序
- 記錄每個訊號的品質問題

### 計算成本

- 處理多個訊號會增加計算時間
- 考慮對大型資料集進行批次處理
- 適當降取樣以減少負荷

### 分析複雜度

- 更多訊號 = 更多變數 = 更多統計比較
- 沒有校正時存在第一類錯誤（偽陽性）風險
- 使用多變量方法或預註冊分析

### 解讀

- 避免過度解讀複雜的多模態模式
- 以生理理論為基礎
- 在做出強烈主張之前複製發現

## 參考文獻

- Berntson, G. G., Cacioppo, J. T., & Quigley, K. S. (1993). Respiratory sinus arrhythmia: autonomic origins, physiological mechanisms, and psychophysiological implications. Psychophysiology, 30(2), 183-196.
- Cacioppo, J. T., Tassinary, L. G., & Berntson, G. (Eds.). (2017). Handbook of psychophysiology (4th ed.). Cambridge University Press.
- Kreibig, S. D. (2010). Autonomic nervous system activity in emotion: A review. Biological psychology, 84(3), 394-421.
- Laborde, S., Mosley, E., & Thayer, J. F. (2017). Heart rate variability and cardiac vagal tone in psychophysiological research–recommendations for experiment planning, data analysis, and data reporting. Frontiers in psychology, 8, 213.
