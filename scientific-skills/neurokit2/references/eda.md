# 皮膚電活動（EDA）分析

## 概述

皮膚電活動（EDA，Electrodermal Activity），也稱為皮膚電反應（GSR，Galvanic Skin Response）或皮膚電導（SC，Skin Conductance），測量皮膚的電導率，反映交感神經系統激發和汗腺活動。EDA 廣泛應用於心理生理學、情感計算和測謊。

## 主要處理管線

### eda_process()

自動處理原始 EDA 訊號，回傳張力/相位分解和 SCR 特徵。

```python
signals, info = nk.eda_process(eda_signal, sampling_rate=100, method='neurokit')
```

**管線步驟：**
1. 訊號清理（低通濾波）
2. 張力-相位分解
3. 皮膚電導反應（SCR）檢測
4. SCR 特徵提取（起始點、波峰、振幅、上升/恢復時間）

**回傳值：**
- `signals`：DataFrame 包含：
  - `EDA_Clean`：濾波後的訊號
  - `EDA_Tonic`：緩慢變化的基線
  - `EDA_Phasic`：快速變化的反應
  - `SCR_Onsets`、`SCR_Peaks`、`SCR_Height`：反應標記
  - `SCR_Amplitude`、`SCR_RiseTime`、`SCR_RecoveryTime`：反應特徵
- `info`：包含處理參數的字典

**方法：**
- `'neurokit'`：cvxEDA 分解 + neurokit 波峰檢測
- `'biosppy'`：中位數平滑 + biosppy 方法

## 預處理函數

### eda_clean()

透過低通濾波去除噪聲。

```python
cleaned_eda = nk.eda_clean(eda_signal, sampling_rate=100, method='neurokit')
```

**方法：**
- `'neurokit'`：低通 Butterworth 濾波器（3 Hz 截止頻率）
- `'biosppy'`：低通 Butterworth 濾波器（5 Hz 截止頻率）

**自動跳過：**
- 如果取樣率 < 7 Hz，則跳過清理（已為低通）

**原理：**
- EDA 頻率內容通常為 0-3 Hz
- 移除高頻噪聲和運動偽影
- 保留緩慢的 SCR（典型上升時間 1-3 秒）

### eda_phasic()

將 EDA 分解為張力（緩慢基線）和相位（快速反應）成分。

```python
tonic, phasic = nk.eda_phasic(eda_cleaned, sampling_rate=100, method='cvxeda')
```

**方法：**

**1. cvxEDA（預設，建議使用）：**
```python
tonic, phasic = nk.eda_phasic(eda_cleaned, sampling_rate=100, method='cvxeda')
```
- 凸優化方法（Greco 等人，2016）
- 稀疏相位驅動模型
- 最符合生理學
- 計算密集但分解效果優越

**2. 中位數平滑：**
```python
tonic, phasic = nk.eda_phasic(eda_cleaned, sampling_rate=100, method='smoothmedian')
```
- 具有可配置窗口的中位數濾波器
- 快速、簡單
- 準確性不如 cvxEDA

**3. 高通濾波（Biopac 的 Acqknowledge）：**
```python
tonic, phasic = nk.eda_phasic(eda_cleaned, sampling_rate=100, method='highpass')
```
- 高通濾波器（0.05 Hz）提取相位
- 快速計算
- 張力透過減法獲得

**4. SparsEDA：**
```python
tonic, phasic = nk.eda_phasic(eda_cleaned, sampling_rate=100, method='sparseda')
```
- 稀疏反卷積方法
- 替代優化方法

**回傳值：**
- `tonic`：緩慢變化的皮膚電導水平（SCL）
- `phasic`：快速的皮膚電導反應（SCRs）

**生理學解釋：**
- **張力（SCL）**：基線激發、一般活化、水合
- **相位（SCR）**：事件相關反應、定向、情緒反應

### eda_peaks()

在相位成分中檢測皮膚電導反應（SCRs）。

```python
peaks, info = nk.eda_peaks(eda_phasic, sampling_rate=100, method='neurokit',
                           amplitude_min=0.1)
```

**方法：**
- `'neurokit'`：針對可靠性優化，可配置閾值
- `'gamboa2008'`：Gamboa 的演算法
- `'kim2004'`：Kim 的方法
- `'vanhalem2020'`：Van Halem 的方法
- `'nabian2018'`：Nabian 的演算法

**主要參數：**
- `amplitude_min`：最小 SCR 振幅（預設：0.1 µS）
  - 太低：噪聲造成的假陽性
  - 太高：遺漏小但有效的反應
- `rise_time_max`：最大上升時間（預設：2 秒）
- `rise_time_min`：最小上升時間（預設：0.01 秒）

**回傳值：**
- 字典包含：
  - `SCR_Onsets`：SCR 開始的索引
  - `SCR_Peaks`：峰值振幅的索引
  - `SCR_Height`：高於基線的峰值高度
  - `SCR_Amplitude`：起始點到峰值的振幅
  - `SCR_RiseTime`：起始點到峰值的持續時間
  - `SCR_RecoveryTime`：峰值到恢復的持續時間（50% 衰減）

**SCR 時間慣例：**
- **潛伏期**：刺激後 1-3 秒（典型）
- **上升時間**：0.5-3 秒
- **恢復時間**：2-10 秒（到 50% 恢復）
- **最小振幅**：0.01-0.05 µS（檢測閾值）

### eda_fixpeaks()

校正檢測到的 SCR 波峰（目前為 EDA 的佔位符）。

```python
corrected_peaks = nk.eda_fixpeaks(peaks)
```

**注意：** 由於 EDA 動態較慢，對 EDA 而言不如心臟訊號那麼關鍵。

## 分析函數

### eda_analyze()

根據資料持續時間自動選擇適當的分析類型。

```python
analysis = nk.eda_analyze(signals, sampling_rate=100)
```

**模式選擇：**
- 持續時間 < 10 秒 → `eda_eventrelated()`
- 持續時間 ≥ 10 秒 → `eda_intervalrelated()`

**回傳值：**
- 包含適合分析模式的 EDA 指標的 DataFrame

### eda_eventrelated()

分析刺激鎖定的 EDA 時期以進行事件相關反應分析。

```python
results = nk.eda_eventrelated(epochs)
```

**計算的指標（每個時期）：**
- `EDA_SCR`：SCR 存在（二元：0 或 1）
- `SCR_Amplitude`：時期內的最大 SCR 振幅
- `SCR_Magnitude`：平均相位活動
- `SCR_Peak_Amplitude`：起始點到峰值的振幅
- `SCR_RiseTime`：從起始點到峰值的時間
- `SCR_RecoveryTime`：到 50% 恢復的時間
- `SCR_Latency`：從刺激到 SCR 起始點的延遲
- `EDA_Tonic`：時期內的平均張力水平

**典型參數：**
- 時期持續時間：刺激後 0-10 秒
- 基線：刺激前 -1 到 0 秒
- 預期 SCR 潛伏期：1-3 秒

**使用情境：**
- 情緒刺激處理（圖像、聲音）
- 認知負荷評估（心算）
- 預期和預測誤差
- 定向反應

### eda_intervalrelated()

分析延長的 EDA 記錄以了解整體激發和活化模式。

```python
results = nk.eda_intervalrelated(signals, sampling_rate=100)
```

**計算的指標：**
- `SCR_Peaks_N`：檢測到的 SCR 數量
- `SCR_Peaks_Amplitude_Mean`：平均 SCR 振幅
- `EDA_Tonic_Mean`、`EDA_Tonic_SD`：張力水平統計
- `EDA_Sympathetic`：交感神經系統指數
- `EDA_SympatheticN`：標準化交感神經指數
- `EDA_Autocorrelation`：時間結構（滯後 4 秒）
- `EDA_Phasic_*`：相位成分的平均值、標準差、最小值、最大值

**記錄持續時間：**
- **最低**：10 秒
- **建議**：60+ 秒以獲得穩定的 SCR 率
- **交感神經指數**：需要 ≥64 秒

**使用情境：**
- 靜息狀態激發評估
- 壓力水平監測
- 基線交感神經活動
- 長期情感狀態

## 專門分析函數

### eda_sympathetic()

從頻帶（0.045-0.25 Hz）衍生交感神經系統活動。

```python
sympathetic = nk.eda_sympathetic(signals, sampling_rate=100, method='posada',
                                  show=False)
```

**方法：**
- `'posada'`：Posada-Quintero 方法（2016）
  - 0.045-0.25 Hz 頻帶的頻譜功率
  - 已針對其他自律神經測量進行驗證
- `'ghiasi'`：Ghiasi 方法（2018）
  - 替代的基於頻率的方法

**要求：**
- **最小持續時間**：64 秒
- 足以在目標頻帶中獲得頻率解析度

**回傳值：**
- `EDA_Sympathetic`：交感神經指數（絕對值）
- `EDA_SympatheticN`：標準化交感神經指數（0-1）

**解釋：**
- 較高值：交感神經激發增加
- 反映張力交感神經活動，而非相位反應
- 補充 SCR 分析

**使用情境：**
- 壓力評估
- 隨時間推移的激發監測
- 認知負荷測量
- 補充 HRV 以了解自律神經平衡

### eda_autocor()

計算自相關以評估 EDA 訊號的時間結構。

```python
autocorr = nk.eda_autocor(eda_phasic, sampling_rate=100, lag=4)
```

**參數：**
- `lag`：時間滯後（秒）（預設：4 秒）

**解釋：**
- 高自相關：持久、緩慢變化的訊號
- 低自相關：快速、不相關的波動
- 反映 SCR 的時間規律性

**使用情境：**
- 評估訊號品質
- 表徵反應模式
- 區分持續性 vs. 瞬態激發

### eda_changepoints()

檢測 EDA 訊號平均值和方差的突然變化。

```python
changepoints = nk.eda_changepoints(eda_phasic, penalty=10000, show=False)
```

**方法：**
- 基於懲罰的分割
- 識別狀態之間的轉換

**參數：**
- `penalty`：控制敏感度（預設：10,000）
  - 較高懲罰：較少、更穩健的變化點
  - 較低懲罰：對小變化更敏感

**回傳值：**
- 檢測到的變化點索引
- 可選的區段視覺化

**使用情境：**
- 識別連續監測中的狀態轉換
- 按激發水平分割資料
- 檢測實驗中的相位變化
- 自動時期定義

## 視覺化

### eda_plot()

建立處理後 EDA 的靜態或互動視覺化。

```python
nk.eda_plot(signals, info, static=True)
```

**顯示內容：**
- 原始和清理後的 EDA 訊號
- 張力和相位成分
- 檢測到的 SCR 起始點、峰值和恢復
- 交感神經指數時間過程（如果已計算）

**互動模式（`static=False`）：**
- 基於 Plotly 的互動探索
- 縮放、平移、懸停查看詳情
- 匯出為圖像格式

## 模擬與測試

### eda_simulate()

生成具有可配置參數的合成 EDA 訊號。

```python
synthetic_eda = nk.eda_simulate(duration=10, sampling_rate=100, scr_number=3,
                                noise=0.01, drift=0.01)
```

**參數：**
- `duration`：訊號長度（秒）
- `sampling_rate`：取樣頻率（Hz）
- `scr_number`：要包含的 SCR 數量
- `noise`：高斯噪聲等級
- `drift`：緩慢基線漂移幅度
- `random_state`：可重現性的種子

**回傳值：**
- 具有逼真 SCR 形態的合成 EDA 訊號

**使用情境：**
- 演算法測試和驗證
- 教育演示
- 方法比較

## 實際考量

### 取樣率建議
- **最低**：10 Hz（足以應付緩慢的 SCR）
- **標準**：20-100 Hz（大多數商業系統）
- **高解析度**：1000 Hz（研究級，過度取樣）

### 記錄持續時間
- **SCR 檢測**：≥10 秒（取決於刺激）
- **事件相關**：每次試驗通常 10-20 秒
- **區間相關**：≥60 秒以獲得穩定估計
- **交感神經指數**：≥64 秒（頻率解析度）

### 電極放置
- **標準位置**：
  - 手掌：遠端/中間指骨（手指）
  - 腳掌：腳底
- **高密度**：大魚際/小魚際隆起
- **避免**：多毛皮膚、低汗腺密度區域
- **雙側**：左手 vs. 右手（通常相似）

### 訊號品質問題

**平坦訊號（無變化）：**
- 檢查電極接觸和凝膠
- 驗證是否正確放置在汗腺豐富區域
- 允許 5-10 分鐘適應期

**過度噪聲：**
- 運動偽影：最小化參與者運動
- 電氣干擾：檢查接地、屏蔽
- 熱效應：控制室溫

**基線漂移：**
- 正常：幾分鐘內的緩慢變化
- 過度：電極極化、接觸不良
- 解決方案：使用 `eda_phasic()` 分離張力漂移

**無反應者：**
- 約 5-10% 的人群 EDA 極小
- 遺傳/生理變異
- 不表示設備故障

### 最佳實踐

**預處理工作流程：**
```python
# 1. 清理訊號
cleaned = nk.eda_clean(eda_raw, sampling_rate=100, method='neurokit')

# 2. 分解張力/相位
tonic, phasic = nk.eda_phasic(cleaned, sampling_rate=100, method='cvxeda')

# 3. 檢測 SCR
signals, info = nk.eda_peaks(phasic, sampling_rate=100, amplitude_min=0.05)

# 4. 分析
analysis = nk.eda_analyze(signals, sampling_rate=100)
```

**事件相關工作流程：**
```python
# 1. 處理訊號
signals, info = nk.eda_process(eda_raw, sampling_rate=100)

# 2. 尋找事件
events = nk.events_find(trigger_channel, threshold=0.5)

# 3. 建立時期（刺激周圍 -1 到 10 秒）
epochs = nk.epochs_create(signals, events, sampling_rate=100,
                          epochs_start=-1, epochs_end=10)

# 4. 事件相關分析
results = nk.eda_eventrelated(epochs)

# 5. 統計分析
# 比較不同條件下的 SCR 振幅
```

## 臨床與研究應用

**情緒與情感科學：**
- 情緒的激發維度（非效價）
- 情緒圖片觀看
- 音樂誘發的情緒
- 恐懼制約

**認知過程：**
- 心智工作負荷和努力
- 注意力和警覺
- 決策和不確定性
- 錯誤處理

**臨床族群：**
- 焦慮症：基線升高、反應誇大
- PTSD：恐懼制約、消退缺陷
- 自閉症：非典型激發模式
- 心理變態：恐懼反應減少

**應用場景：**
- 測謊（測謊儀）
- 使用者體驗研究
- 駕駛監測
- 真實世界環境中的壓力評估

**神經影像整合：**
- fMRI：EDA 與杏仁核、島葉活動相關
- 腦部影像期間的同步記錄
- 自律神經-大腦耦合

## 解釋指南

**SCR 振幅：**
- **0.01-0.05 µS**：小但可檢測
- **0.05-0.2 µS**：中等反應
- **>0.2 µS**：大反應
- **情境依賴**：受試者內標準化

**SCR 頻率：**
- **靜息**：每分鐘 1-3 個 SCR（典型）
- **壓力下**：每分鐘 >5 個 SCR
- **非特異性 SCR**：自發性（無可識別的刺激）

**張力 SCL：**
- **範圍**：2-20 µS（個體間變異很大）
- **受試者內變化**比絕對水平更具解釋性
- **增加**：激發、壓力、認知負荷
- **減少**：放鬆、習慣化

## 參考文獻

- Boucsein, W. (2012). Electrodermal activity (2nd ed.). Springer Science & Business Media.
- Greco, A., Valenza, G., & Scilingo, E. P. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. IEEE Transactions on Biomedical Engineering, 63(4), 797-804.
- Posada-Quintero, H. F., Florian, J. P., Orjuela-Cañón, A. D., Aljama-Corrales, T., Charleston-Villalobos, S., & Chon, K. H. (2016). Power spectral density analysis of electrodermal activity for sympathetic function assessment. Annals of biomedical engineering, 44(10), 3124-3135.
- Dawson, M. E., Schell, A. M., & Filion, D. L. (2017). The electrodermal system. In Handbook of psychophysiology (pp. 217-243). Cambridge University Press.
