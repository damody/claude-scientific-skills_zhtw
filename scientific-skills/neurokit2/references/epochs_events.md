# 時段與事件相關分析（Epochs and Event-Related Analysis）

## 概述

事件相關分析檢視與特定刺激或事件時間鎖定的生理反應。NeuroKit2 提供用於事件偵測、時段（epoch）建立、平均和事件相關特徵提取的工具，適用於所有訊號類型。

## 事件偵測

### events_find()

根據閾值交叉或變化自動偵測訊號中的事件/觸發。

```python
events = nk.events_find(event_channel, threshold=0.5, threshold_keep='above',
                        duration_min=1, inter_min=0)
```

**參數：**
- `threshold`：偵測閾值
- `threshold_keep`：`'above'` 或 `'below'` 閾值
- `duration_min`：保留的最小事件持續時間（樣本數）
- `inter_min`：事件之間的最小間隔（樣本數）

**回傳：**
- 字典包含：
  - `'onset'`：事件開始索引
  - `'offset'`：事件結束索引（如適用）
  - `'duration'`：事件持續時間
  - `'label'`：事件標籤（如有多種事件類型）

**常見應用場景：**

**實驗中的 TTL 觸發：**
```python
# 觸發通道：基線 0V，事件期間 5V 脈衝
events = nk.events_find(trigger_channel, threshold=2.5, threshold_keep='above')
```

**按鈕按下：**
```python
# 偵測按鈕訊號變高的時刻
button_events = nk.events_find(button_signal, threshold=0.5, threshold_keep='above',
                               duration_min=10)  # 消彈跳
```

**狀態變化：**
```python
# 偵測高於/低於閾值的時期
high_arousal = nk.events_find(eda_signal, threshold='auto', duration_min=100)
```

### events_plot()

視覺化事件時間相對於訊號。

```python
nk.events_plot(events, signal)
```

**顯示：**
- 訊號軌跡
- 事件標記（垂直線或陰影區域）
- 事件標籤

**應用場景：**
- 驗證事件偵測準確性
- 檢視事件的時間分布
- 時段化前的品質控制

## 時段建立

### epochs_create()

圍繞事件建立資料時段（片段）用於事件相關分析。

```python
epochs = nk.epochs_create(data, events, sampling_rate=1000,
                          epochs_start=-0.5, epochs_end=2.0,
                          event_labels=None, event_conditions=None,
                          baseline_correction=False)
```

**參數：**
- `data`：包含訊號的 DataFrame 或單一訊號
- `events`：事件索引或來自 `events_find()` 的字典
- `sampling_rate`：訊號取樣率（Hz）
- `epochs_start`：相對於事件的開始時間（秒，負數 = 之前）
- `epochs_end`：相對於事件的結束時間（秒，正數 = 之後）
- `event_labels`：每個事件的標籤列表（選用）
- `event_conditions`：每個事件的條件名稱列表（選用）
- `baseline_correction`：如為 True，從每個時段減去基線平均值

**回傳：**
- DataFrame 字典，每個時段一個
- 每個 DataFrame 包含訊號資料，時間相對於事件（Index=0 為事件開始）
- 如有提供，包含 `'Label'` 和 `'Condition'` 欄位

**典型時段視窗：**
- **視覺 ERP**：-0.2 到 1.0 秒（200 毫秒基線，1 秒刺激後）
- **心臟定向**：-1.0 到 10 秒（捕捉預期和反應）
- **EMG 驚嚇**：-0.1 到 0.5 秒（短暫反應）
- **EDA SCR**：-1.0 到 10 秒（1-3 秒潛伏期，緩慢恢復）

### 事件標籤和條件

按類型和實驗條件組織事件：

```python
# 範例：情緒圖片實驗
event_times = [1000, 2500, 4200, 5800]  # 事件開始（樣本數）
event_labels = ['trial1', 'trial2', 'trial3', 'trial4']
event_conditions = ['positive', 'negative', 'positive', 'neutral']

epochs = nk.epochs_create(signals, events=event_times, sampling_rate=1000,
                          epochs_start=-1, epochs_end=5,
                          event_labels=event_labels,
                          event_conditions=event_conditions)
```

**存取時段：**
```python
# 按編號存取時段
epoch_1 = epochs['1']

# 按條件篩選
positive_epochs = {k: v for k, v in epochs.items() if v['Condition'][0] == 'positive'}
```

### 基線校正

從時段移除刺激前基線以隔離事件相關變化：

**自動（在時段建立期間）：**
```python
epochs = nk.epochs_create(data, events, sampling_rate=1000,
                          epochs_start=-0.5, epochs_end=2.0,
                          baseline_correction=True)  # 減去整個基線的平均值
```

**手動（在時段建立後）：**
```python
# 減去基線時期平均值
baseline_start = -0.5  # 秒
baseline_end = 0.0     # 秒

for key, epoch in epochs.items():
    baseline_mask = (epoch.index >= baseline_start) & (epoch.index < baseline_end)
    baseline_mean = epoch[baseline_mask].mean()
    epochs[key] = epoch - baseline_mean
```

**何時進行基線校正：**
- **ERP**：總是（隔離事件相關變化）
- **心臟/EDA**：通常（移除個體間基線差異）
- **絕對測量**：有時不需要（例如，分析絕對振幅）

## 時段分析和視覺化

### epochs_plot()

視覺化個別或平均時段。

```python
nk.epochs_plot(epochs, column='ECG_Rate', condition=None, show=True)
```

**參數：**
- `column`：要繪製的訊號欄位
- `condition`：僅繪製特定條件（選用）

**顯示：**
- 個別時段軌跡（半透明）
- 跨時段平均（粗線）
- 選用：誤差陰影（SEM 或 SD）

**應用場景：**
- 視覺化事件相關反應
- 比較條件
- 識別異常時段

### epochs_average()

計算跨時段的總平均和統計。

```python
average_epochs = nk.epochs_average(epochs, output='dict')
```

**參數：**
- `output`：`'dict'`（預設）或 `'df'`（DataFrame）

**回傳：**
- 字典或 DataFrame 包含：
  - `'Mean'`：每個時間點的跨時段平均
  - `'SD'`：標準差
  - `'SE'`：平均標準誤
  - `'CI_lower'`、`'CI_upper'`：95% 信賴區間

**應用場景：**
- 計算事件相關電位（ERP）
- 心臟/EDA/EMG 反應的總平均
- 群組層級分析

**按條件平均：**
```python
# 按條件分開平均
positive_epochs = {k: v for k, v in epochs.items() if v['Condition'][0] == 'positive'}
negative_epochs = {k: v for k, v in epochs.items() if v['Condition'][0] == 'negative'}

avg_positive = nk.epochs_average(positive_epochs)
avg_negative = nk.epochs_average(negative_epochs)
```

### epochs_to_df()

將時段字典轉換為統一的 DataFrame。

```python
epochs_df = nk.epochs_to_df(epochs)
```

**回傳：**
- 所有時段堆疊的單一 DataFrame
- 包含 `'Epoch'`、`'Time'`、`'Label'`、`'Condition'` 欄位
- 便於使用 pandas/seaborn 進行統計分析和繪圖

**應用場景：**
- 為混合效果模型準備資料
- 使用 seaborn/plotly 繪圖
- 匯出至 R 或統計軟體

### epochs_to_array()

將時段轉換為 3D NumPy 陣列。

```python
epochs_array = nk.epochs_to_array(epochs, column='ECG_Rate')
```

**回傳：**
- 3D 陣列：(n_epochs, n_timepoints, n_columns)

**應用場景：**
- 機器學習輸入（時段化特徵）
- 自訂基於陣列的分析
- 陣列資料的統計檢定

## 訊號特定事件相關分析

NeuroKit2 為每種訊號類型提供專門的事件相關分析：

### ECG 事件相關
```python
ecg_epochs = nk.epochs_create(ecg_signals, events, sampling_rate=1000,
                              epochs_start=-1, epochs_end=10)
ecg_results = nk.ecg_eventrelated(ecg_epochs)
```

**計算的指標：**
- `ECG_Rate_Baseline`：事件前的心率
- `ECG_Rate_Min/Max`：時段期間的最小/最大心率
- `ECG_Phase_*`：事件開始時的心臟相位
- 跨時間視窗的心率動態

### EDA 事件相關
```python
eda_epochs = nk.epochs_create(eda_signals, events, sampling_rate=100,
                              epochs_start=-1, epochs_end=10)
eda_results = nk.eda_eventrelated(eda_epochs)
```

**計算的指標：**
- `EDA_SCR`：SCR 的存在（二元）
- `SCR_Amplitude`：最大 SCR 振幅
- `SCR_Latency`：到 SCR 開始的時間
- `SCR_RiseTime`、`SCR_RecoveryTime`
- `EDA_Tonic`：平均緊張水平

### RSP 事件相關
```python
rsp_epochs = nk.epochs_create(rsp_signals, events, sampling_rate=100,
                              epochs_start=-0.5, epochs_end=5)
rsp_results = nk.rsp_eventrelated(rsp_epochs)
```

**計算的指標：**
- `RSP_Rate_Mean`：平均呼吸率
- `RSP_Amplitude_Mean`：平均呼吸深度
- `RSP_Phase`：事件時的呼吸相位
- 呼吸率/振幅動態

### EMG 事件相關
```python
emg_epochs = nk.epochs_create(emg_signals, events, sampling_rate=1000,
                              epochs_start=-0.1, epochs_end=1.0)
emg_results = nk.emg_eventrelated(emg_epochs)
```

**計算的指標：**
- `EMG_Activation`：激活的存在
- `EMG_Amplitude_Mean/Max`：振幅統計
- `EMG_Onset_Latency`：到激活開始的時間
- `EMG_Bursts`：激活爆發次數

### EOG 事件相關
```python
eog_epochs = nk.epochs_create(eog_signals, events, sampling_rate=500,
                              epochs_start=-0.5, epochs_end=2.0)
eog_results = nk.eog_eventrelated(eog_epochs)
```

**計算的指標：**
- `EOG_Blinks_N`：時段期間的眨眼次數
- `EOG_Rate_Mean`：眨眼率
- 眨眼的時間分布

### PPG 事件相關
```python
ppg_epochs = nk.epochs_create(ppg_signals, events, sampling_rate=100,
                              epochs_start=-1, epochs_end=10)
ppg_results = nk.ppg_eventrelated(ppg_epochs)
```

**計算的指標：**
- 與 ECG 類似：心率動態、相位資訊

## 實務工作流程

### 完整事件相關分析管線

```python
import neurokit2 as nk

# 1. 處理生理訊號
ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=1000)
eda_signals, eda_info = nk.eda_process(eda, sampling_rate=100)

# 2. 如需要，對齊取樣率
eda_signals_resampled = nk.signal_resample(eda_signals, sampling_rate=100,
                                           desired_sampling_rate=1000)

# 3. 合併訊號到單一 DataFrame
signals = pd.concat([ecg_signals, eda_signals_resampled], axis=1)

# 4. 偵測事件
events = nk.events_find(trigger_channel, threshold=0.5)

# 5. 添加事件標籤和條件
event_labels = ['trial1', 'trial2', 'trial3', ...]
event_conditions = ['condition_A', 'condition_B', 'condition_A', ...]

# 6. 建立時段
epochs = nk.epochs_create(signals, events, sampling_rate=1000,
                          epochs_start=-1.0, epochs_end=5.0,
                          event_labels=event_labels,
                          event_conditions=event_conditions,
                          baseline_correction=True)

# 7. 訊號特定事件相關分析
ecg_results = nk.ecg_eventrelated(epochs)
eda_results = nk.eda_eventrelated(epochs)

# 8. 合併結果
results = pd.merge(ecg_results, eda_results, left_index=True, right_index=True)

# 9. 按條件進行統計分析
results['Condition'] = event_conditions
condition_comparison = results.groupby('Condition').mean()
```

### 處理多種事件類型

```python
# 不同標記的不同事件類型
event_type1 = nk.events_find(trigger_ch1, threshold=0.5)
event_type2 = nk.events_find(trigger_ch2, threshold=0.5)

# 合併帶標籤的事件
all_events = np.concatenate([event_type1['onset'], event_type2['onset']])
event_labels = ['type1'] * len(event_type1['onset']) + ['type2'] * len(event_type2['onset'])

# 按時間排序
sort_idx = np.argsort(all_events)
all_events = all_events[sort_idx]
event_labels = [event_labels[i] for i in sort_idx]

# 建立時段
epochs = nk.epochs_create(signals, all_events, sampling_rate=1000,
                          epochs_start=-0.5, epochs_end=3.0,
                          event_labels=event_labels)

# 按類型分開
type1_epochs = {k: v for k, v in epochs.items() if v['Label'][0] == 'type1'}
type2_epochs = {k: v for k, v in epochs.items() if v['Label'][0] == 'type2'}
```

### 品質控制和偽跡拒絕

```python
# 移除噪音過多或有偽跡的時段
clean_epochs = {}
for key, epoch in epochs.items():
    # 範例：如果 EDA 振幅過高則拒絕（動作偽跡）
    if epoch['EDA_Phasic'].abs().max() < 5.0:  # 閾值
        # 範例：如果心率變化過大則拒絕（無效）
        if epoch['ECG_Rate'].max() - epoch['ECG_Rate'].min() < 50:
            clean_epochs[key] = epoch

print(f"保留 {len(clean_epochs)}/{len(epochs)} 個時段")

# 分析乾淨的時段
results = nk.ecg_eventrelated(clean_epochs)
```

## 統計考量

### 樣本量
- **ERP/平均**：每條件至少 20-30+ 次試驗
- **個別試驗分析**：混合效果模型可處理可變試驗次數
- **群組比較**：使用先導資料進行統計檢定力分析

### 時間視窗選擇
- **先驗假設**：根據文獻預先註冊時間視窗
- **探索性**：使用完整時段，校正多重比較
- **避免**：根據觀察資料選擇視窗（循環）

### 基線時期
- 應無預期效應
- 足夠的持續時間以獲得穩定估計（典型 500-1000 毫秒）
- 快速動態較短（例如，驚嚇：100 毫秒足夠）

### 條件比較
- 受試者內設計使用重複測量 ANOVA
- 不平衡資料使用混合效果模型
- 非參數比較使用置換檢定
- 校正多重比較（時間點/訊號）

## 常見應用

**認知心理學：**
- P300 ERP 分析
- 錯誤相關負波（ERN）
- 注意力瞬盲
- 工作記憶負荷效應

**情感神經科學：**
- 情緒圖片觀看（EDA、HR、臉部 EMG）
- 恐懼制約（HR 減速、SCR）
- 效價/喚起維度

**臨床研究：**
- 驚嚇反應（眼輪匝肌 EMG）
- 定向反應（HR 減速）
- 預期和預測誤差

**心理生理學：**
- 心臟防禦反應
- 定向 vs. 防禦反射
- 情緒期間的呼吸變化

**人機互動：**
- 事件期間的使用者參與
- 驚訝/預期違反
- 任務事件期間的認知負荷

## 參考文獻

- Luck, S. J. (2014). An introduction to the event-related potential technique (2nd ed.). MIT press.
- Bradley, M. M., & Lang, P. J. (2000). Measuring emotion: Behavior, feeling, and physiology. In R. D. Lane & L. Nadel (Eds.), Cognitive neuroscience of emotion (pp. 242-276). Oxford University Press.
- Boucsein, W. (2012). Electrodermal activity (2nd ed.). Springer.
- Gratton, G., Coles, M. G., & Donchin, E. (1983). A new method for off-line removal of ocular artifact. Electroencephalography and clinical neurophysiology, 55(4), 468-484.
