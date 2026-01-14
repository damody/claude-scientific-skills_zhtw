# 肌電圖（EMG）分析

## 概述

肌電圖（EMG）測量骨骼肌收縮時產生的電活動。NeuroKit2 中的 EMG 分析專注於振幅估計、肌肉激活檢測和時間動態，適用於心理生理學和運動控制研究。

## 主要處理管線

### emg_process()

自動化 EMG 訊號處理管線。

```python
signals, info = nk.emg_process(emg_signal, sampling_rate=1000)
```

**管線步驟：**
1. 訊號清理（高通濾波、去趨勢）
2. 振幅包絡提取
3. 肌肉激活檢測
4. 起始和結束識別

**返回：**
- `signals`：包含以下內容的 DataFrame：
  - `EMG_Clean`：濾波後的 EMG 訊號
  - `EMG_Amplitude`：線性包絡（平滑整流訊號）
  - `EMG_Activity`：二元激活指示器（0/1）
  - `EMG_Onsets`：激活起始標記
  - `EMG_Offsets`：激活結束標記
- `info`：包含激活參數的字典

**典型工作流程：**
- 處理原始 EMG → 提取振幅 → 檢測激活 → 分析特徵

## 預處理函數

### emg_clean()

應用濾波以去除雜訊並準備振幅提取。

```python
cleaned_emg = nk.emg_clean(emg_signal, sampling_rate=1000)
```

**濾波方法（BioSPPy 方法）：**
- 四階 Butterworth 高通濾波器（100 Hz）
- 去除低頻運動偽影和基線漂移
- 去除直流偏移
- 訊號去趨勢

**原理：**
- EMG 頻率內容：20-500 Hz（主要：50-150 Hz）
- 100 Hz 高通濾波隔離肌肉活動
- 去除 ECG 污染（尤其在軀幹肌肉）
- 去除運動偽影（<20 Hz）

**EMG 訊號特徵：**
- 收縮期間隨機、零均值振盪
- 較高振幅 = 較強收縮
- 原始 EMG：正負偏轉皆有

## 特徵提取

### emg_amplitude()

計算代表肌肉收縮強度的線性包絡。

```python
amplitude = nk.emg_amplitude(cleaned_emg, sampling_rate=1000)
```

**方法：**
1. 全波整流（絕對值）
2. 低通濾波（平滑包絡）
3. 降取樣（可選）

**線性包絡：**
- 跟隨 EMG 振幅調變的平滑曲線
- 代表肌力/激活程度
- 適合進一步分析（激活檢測、積分）

**典型平滑：**
- 低通濾波器：10-20 Hz 截止
- 移動平均：50-200 ms 視窗
- 平衡：響應性與平滑度

## 激活檢測

### emg_activation()

檢測肌肉激活時期（起始和結束）。

```python
activity, info = nk.emg_activation(emg_amplitude, sampling_rate=1000, method='threshold',
                                   threshold='auto', duration_min=0.05)
```

**方法：**

**1. 閾值法（預設）：**
```python
activity = nk.emg_activation(amplitude, method='threshold', threshold='auto')
```
- 將振幅與閾值比較
- `threshold='auto'`：基於訊號統計自動（例如平均值 + 1 SD）
- `threshold=0.1`：手動絕對閾值
- 簡單、快速、廣泛使用

**2. 高斯混合模型（GMM）：**
```python
activity = nk.emg_activation(amplitude, method='mixture', n_clusters=2)
```
- 無監督聚類：活動 vs. 休息
- 適應訊號特徵
- 對變化的基線更穩健

**3. 變化點檢測：**
```python
activity = nk.emg_activation(amplitude, method='changepoint')
```
- 檢測訊號屬性的突然轉變
- 識別激活/去激活點
- 適用於複雜時間模式

**4. 雙峰性（Silva et al., 2013）：**
```python
activity = nk.emg_activation(amplitude, method='bimodal')
```
- 測試雙峰分佈（活動 vs. 休息）
- 確定最佳分離閾值
- 統計原則性

**關鍵參數：**
- `duration_min`：最小激活持續時間（秒）
  - 過濾短暫虛假激活
  - 典型：50-100 ms
- `threshold`：激活閾值（依方法而定）

**返回：**
- `activity`：二元陣列（0 = 休息，1 = 活動）
- `info`：包含起始/結束索引的字典

**激活指標：**
- **起始**：從休息到活動的轉變
- **結束**：從活動到休息的轉變
- **持續時間**：起始和結束之間的時間
- **爆發**：單一連續激活時期

## 分析函數

### emg_analyze()

自動選擇事件相關或區間相關分析。

```python
analysis = nk.emg_analyze(signals, sampling_rate=1000)
```

**模式選擇：**
- 持續時間 < 10 秒 → 事件相關
- 持續時間 ≥ 10 秒 → 區間相關

### emg_eventrelated()

分析離散事件/刺激的 EMG 反應。

```python
results = nk.emg_eventrelated(epochs)
```

**計算的指標（每個分段）：**
- `EMG_Activation`：激活存在（二元）
- `EMG_Amplitude_Mean`：分段期間的平均振幅
- `EMG_Amplitude_Max`：峰值振幅
- `EMG_Bursts`：激活爆發次數
- `EMG_Onset_Latency`：從事件到首次激活的時間（如適用）

**使用情境：**
- 驚嚇反應（眼輪匝肌 EMG）
- 情緒刺激期間的面部 EMG（皺眉肌、顴大肌）
- 運動反應潛時
- 肌肉反應性範式

### emg_intervalrelated()

分析延長的 EMG 記錄。

```python
results = nk.emg_intervalrelated(signals, sampling_rate=1000)
```

**計算的指標：**
- `EMG_Bursts_N`：激活爆發總數
- `EMG_Amplitude_Mean`：整個區間的平均振幅
- `EMG_Activation_Duration`：處於活動狀態的總時間
- `EMG_Rest_Duration`：處於休息狀態的總時間

**使用情境：**
- 靜息肌肉張力評估
- 慢性疼痛或壓力相關的肌肉活動
- 持續任務期間的疲勞監測
- 姿勢肌肉評估

## 模擬和視覺化

### emg_simulate()

生成合成 EMG 訊號用於測試。

```python
synthetic_emg = nk.emg_simulate(duration=10, sampling_rate=1000, burst_number=3,
                                noise=0.1, random_state=42)
```

**參數：**
- `burst_number`：包含的激活爆發數量
- `noise`：背景雜訊水準
- `random_state`：可重現性種子

**生成的特徵：**
- 爆發期間隨機類 EMG 振盪
- 真實的頻率內容
- 可變的爆發時間和振幅

**使用情境：**
- 演算法驗證
- 檢測參數調整
- 教育演示

### emg_plot()

視覺化處理過的 EMG 訊號。

```python
nk.emg_plot(signals, info, static=True)
```

**顯示：**
- 原始和清理後的 EMG 訊號
- 振幅包絡
- 檢測到的激活時期
- 起始/結束標記

**互動模式：**設定 `static=False` 進行 Plotly 視覺化

## 實務考量

### 取樣率建議
- **最低**：500 Hz（250 Hz 上限頻率的奈奎斯特）
- **標準**：1000 Hz（大多數研究應用）
- **高解析度**：2000-4000 Hz（詳細運動單元研究）
- **表面 EMG**：典型 1000-2000 Hz
- **肌肉內 EMG**：單一運動單元需 10,000+ Hz

### 記錄持續時間
- **事件相關**：依範式而定（例如每次試驗 2-5 秒）
- **持續收縮**：秒到分鐘
- **疲勞研究**：分鐘到小時
- **長期監測**：天（穿戴式 EMG）

### 電極放置

**表面 EMG（最常見）：**
- 雙極配置（兩個電極在肌腹上）
- 參考/接地電極在電中性位置（骨骼）
- 皮膚準備：清潔、磨皮、降低阻抗
- 電極間距：10-20 mm（SENIAM 標準）

**肌肉特定指南：**
- 遵循 SENIAM（非侵入性肌肉評估的表面 EMG）建議
- 收縮時觸診肌肉以定位肌腹
- 電極與肌纖維方向對齊

**心理生理學中的常見肌肉：**
- **皺眉肌**：皺眉、負向情感（眉毛上方）
- **顴大肌**：微笑、正向情感（臉頰）
- **眼輪匝肌**：驚嚇反應、恐懼（眼周）
- **咬肌**：咬緊下巴、壓力（下巴肌肉）
- **斜方肌**：肩膀緊張、壓力（上背）
- **額肌**：額頭緊張、驚訝

### 訊號品質問題

**ECG 污染：**
- 常見於軀幹和近端肌肉
- 高通濾波（>100 Hz）通常足夠
- 如持續存在：模板減法、ICA

**運動偽影：**
- 低頻干擾
- 電極電纜移動
- 固定電極、減少電纜運動

**電極問題：**
- 接觸不良：高阻抗、低振幅
- 出汗：振幅逐漸增加、不穩定
- 毛髮：清潔或剃除區域

**串擾：**
- 相鄰肌肉活動滲入記錄
- 仔細的電極放置
- 小電極間距

### 最佳實踐

**標準工作流程：**
```python
# 1. 清理訊號（高通濾波、去趨勢）
cleaned = nk.emg_clean(emg_raw, sampling_rate=1000)

# 2. 提取振幅包絡
amplitude = nk.emg_amplitude(cleaned, sampling_rate=1000)

# 3. 檢測激活時期
activity, info = nk.emg_activation(amplitude, sampling_rate=1000,
                                   method='threshold', threshold='auto')

# 4. 全面處理（替代方案）
signals, info = nk.emg_process(emg_raw, sampling_rate=1000)

# 5. 分析
analysis = nk.emg_analyze(signals, sampling_rate=1000)
```

**正規化：**
```python
# 最大自主收縮（MVC）正規化
mvc_amplitude = np.max(mvc_emg_amplitude)  # 來自單獨的 MVC 試驗
normalized_emg = (amplitude / mvc_amplitude) * 100  # 表達為 % MVC

# 常用於人因工程、運動生理學
# 允許跨個體和會話比較
```

## 臨床和研究應用

**心理生理學：**
- **面部 EMG**：情緒效價（微笑 vs. 皺眉）
- **驚嚇反應**：恐懼、驚訝、防禦反應
- **壓力**：慢性肌肉緊張（斜方肌、咬肌）

**運動控制和復健：**
- 步態分析
- 運動障礙（震顫、肌張力障礙）
- 中風復健（肌肉再激活）
- 義肢控制（肌電式）

**人因工程和職業健康：**
- 工作相關肌肉骨骼疾病
- 姿勢評估
- 重複性勞損風險

**運動科學：**
- 運動期間的肌肉激活模式
- 疲勞評估（中位頻率偏移）
- 訓練最佳化

**生物回饋：**
- 放鬆訓練（降低肌肉緊張）
- 神經肌肉再教育
- 慢性疼痛管理

**睡眠醫學：**
- 下巴 EMG 用於 REM 睡眠弛緩
- 週期性肢體運動
- 磨牙症（磨牙）

## 進階 EMG 分析（超越 NeuroKit2 基本函數）

**頻域：**
- 疲勞期間中位頻率偏移
- 功率譜分析
- 需要較長區段（每分析視窗 ≥1 秒）

**運動單元識別：**
- 肌肉內 EMG
- 尖峰檢測和分類
- 發放頻率分析
- 需要高取樣率（10+ kHz）

**肌肉協調：**
- 共同收縮指數
- 協同分析
- 多肌肉整合

## 解讀指南

**振幅（線性包絡）：**
- 較高振幅 ≈ 較強收縮（非完美線性）
- 與力量的關係：S 型曲線，受多因素影響
- 受試者內比較最可靠

**激活閾值：**
- 自動閾值：方便但需視覺驗證
- 手動閾值：非標準肌肉可能需要
- 靜息基線：應接近零（如不是，檢查電極）

**爆發特徵：**
- **相位性**：短暫爆發（驚嚇、快速動作）
- **緊張性**：持續激活（姿勢、持續抓握）
- **節律性**：重複爆發（震顫、行走）

## 參考文獻

- Fridlund, A. J., & Cacioppo, J. T. (1986). Guidelines for human electromyographic research. Psychophysiology, 23(5), 567-589.
- Hermens, H. J., Freriks, B., Disselhorst-Klug, C., & Rau, G. (2000). Development of recommendations for SEMG sensors and sensor placement procedures. Journal of electromyography and Kinesiology, 10(5), 361-374.
- Silva, H., Scherer, R., Sousa, J., & Londral, A. (2013). Towards improving the ssability of electromyographic interfaces. Journal of Oral Rehabilitation, 40(6), 456-465.
- Tassinary, L. G., Cacioppo, J. T., & Vanman, E. J. (2017). The skeletomotor system: Surface electromyography. In Handbook of psychophysiology (pp. 267-299). Cambridge University Press.
