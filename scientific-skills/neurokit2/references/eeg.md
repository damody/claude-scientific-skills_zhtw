# EEG 分析與微狀態（Microstates）

## 概述

分析腦電圖（Electroencephalography，EEG）訊號以進行頻帶功率、通道品質評估、源定位和微狀態（microstate）識別。NeuroKit2 與 MNE-Python 整合，提供完整的 EEG 處理工作流程。

## 核心 EEG 函數

### eeg_power()

計算指定通道在標準頻帶上的功率。

```python
power = nk.eeg_power(eeg_data, sampling_rate=250, channels=['Fz', 'Cz', 'Pz'],
                     frequency_bands={'Delta': (0.5, 4),
                                     'Theta': (4, 8),
                                     'Alpha': (8, 13),
                                     'Beta': (13, 30),
                                     'Gamma': (30, 45)})
```

**標準頻帶：**
- **Delta（0.5-4 Hz）**：深度睡眠、無意識過程
- **Theta（4-8 Hz）**：嗜睡、冥想、記憶編碼
- **Alpha（8-13 Hz）**：放鬆清醒、閉眼
- **Beta（13-30 Hz）**：主動思考、專注、焦慮
- **Gamma（30-45 Hz）**：認知處理、綁定

**回傳：**
- DataFrame，包含每個通道 × 頻帶組合的功率值
- 欄位名稱：`Channel_Band`（例如 'Fz_Alpha'、'Cz_Beta'）

**應用場景：**
- 靜息狀態分析
- 認知狀態分類
- 睡眠分期
- 冥想或神經回饋監測

### eeg_badchannels()

使用統計異常值偵測識別問題通道。

```python
bad_channels = nk.eeg_badchannels(eeg_data, sampling_rate=250, bad_threshold=2)
```

**偵測方法：**
- 跨通道的標準差異常值
- 與其他通道的相關性
- 平坦或無訊號通道
- 噪音過多的通道

**參數：**
- `bad_threshold`：異常值偵測的 Z 分數閾值（預設：2）

**回傳：**
- 被識別為有問題的通道名稱列表

**應用場景：**
- 分析前的品質控制
- 自動壞通道拒絕
- 內插或排除決策

### eeg_rereference()

將電壓測量值重新表達為相對於不同參考點。

```python
rereferenced = nk.eeg_rereference(eeg_data, reference='average', robust=False)
```

**參考類型：**
- `'average'`：平均參考（所有電極的平均值）
- `'REST'`：參考電極標準化技術（Reference Electrode Standardization Technique）
- `'bipolar'`：電極對之間的差分記錄
- 特定通道名稱：使用單一電極作為參考

**常見參考：**
- **平均參考**：高密度 EEG 最常用
- **連接乳突**：傳統臨床 EEG
- **頂點（Cz）**：有時用於 ERP 研究
- **REST**：近似無窮遠參考

**回傳：**
- 重新參考的 EEG 資料

### eeg_gfp()

計算全域場功率（Global Field Power）- 每個時間點所有電極的標準差。

```python
gfp = nk.eeg_gfp(eeg_data)
```

**解讀：**
- 高 GFP：跨區域強烈、同步的大腦活動
- 低 GFP：弱或不同步的活動
- GFP 峰值：穩定地形的時間點，用於微狀態偵測

**應用場景：**
- 識別穩定地形模式的時期
- 選擇微狀態分析的時間點
- 事件相關電位（ERP）視覺化

### eeg_diss()

測量電場配置之間的地形差異性。

```python
dissimilarity = nk.eeg_diss(eeg_data1, eeg_data2, method='gfp')
```

**方法：**
- 基於 GFP：標準化差異
- 空間相關
- 餘弦距離

**應用場景：**
- 比較不同條件之間的地形
- 微狀態轉換分析
- 模板匹配

## 源定位（Source Localization）

### eeg_source()

執行源重建以從頭皮記錄估計大腦層級活動。

```python
sources = nk.eeg_source(eeg_data, method='sLORETA')
```

**方法：**
- `'sLORETA'`：標準化低解析度電磁斷層成像（Standardized Low-Resolution Electromagnetic Tomography）
  - 點源的零定位誤差
  - 良好的空間解析度
- `'MNE'`：最小範數估計（Minimum Norm Estimate）
  - 快速、成熟
  - 傾向淺層源
- `'dSPM'`：動態統計參數映射（Dynamic Statistical Parametric Mapping）
  - 標準化的 MNE
- `'eLORETA'`：精確 LORETA
  - 改進的定位準確度

**需求：**
- 前向模型（導聯場矩陣）
- 共配準的電極位置
- 頭部模型（邊界元素或球形）

**回傳：**
- 源空間活動估計

### eeg_source_extract()

從特定解剖大腦區域提取活動。

```python
regional_activity = nk.eeg_source_extract(sources, regions=['PFC', 'MTL', 'Parietal'])
```

**區域選項：**
- 標準圖譜：Desikan-Killiany、Destrieux、AAL
- 自訂 ROI
- Brodmann 區域

**回傳：**
- 每個區域的時間序列
- 跨體素的平均或主成分

**應用場景：**
- 感興趣區域分析
- 功能連接
- 源層級統計

## 微狀態分析（Microstate Analysis）

微狀態是短暫（80-120 毫秒）的穩定大腦地形時期，代表協調的神經網路。通常有 4-7 個微狀態類別（通常標記為 A、B、C、D），具有不同的功能。

### microstates_segment()

使用聚類演算法識別和提取微狀態。

```python
microstates = nk.microstates_segment(eeg_data, n_microstates=4, sampling_rate=250,
                                      method='kmod', normalize=True)
```

**方法：**
- `'kmod'`（預設）：為 EEG 地形最佳化的修改版 k-means
  - 極性不變聚類
  - 微狀態文獻中最常見
- `'kmeans'`：標準 k-means 聚類
- `'kmedoids'`：K-medoids（對異常值更穩健）
- `'pca'`：主成分分析
- `'ica'`：獨立成分分析
- `'aahc'`：原子化和聚集層次聚類

**參數：**
- `n_microstates`：微狀態類別數（通常 4-7）
- `normalize`：標準化地形（建議：True）
- `n_inits`：隨機初始化次數（增加以提高穩定性）

**回傳：**
- 字典包含：
  - `'maps'`：微狀態模板地形
  - `'labels'`：每個時間點的微狀態標籤
  - `'gfp'`：全域場功率
  - `'gev'`：全域解釋變異

### microstates_findnumber()

估計最佳微狀態數量。

```python
optimal_k = nk.microstates_findnumber(eeg_data, show=True)
```

**標準：**
- **全域解釋變異（GEV）**：解釋的變異百分比
  - 肘部法：在 GEV 曲線中找到「膝蓋」
  - 通常達到 70-80% GEV
- **Krzanowski-Lai（KL）標準**：平衡擬合和簡約性的統計量度
  - 最大 KL 表示最佳 k

**典型範圍：** 4-7 個微狀態
- 4 個微狀態：經典的 A、B、C、D 狀態
- 5-7 個微狀態：更精細的分解

### microstates_classify()

根據前後和左右通道值重新排序微狀態。

```python
classified = nk.microstates_classify(microstates)
```

**目的：**
- 跨受試者標準化微狀態標籤
- 匹配傳統的 A、B、C、D 地形：
  - **A**：左右方向，頂枕
  - **B**：右左方向，額顳
  - **C**：前後方向，額中央
  - **D**：額中央，前後（C 的反向）

**回傳：**
- 重新排序的微狀態映射和標籤

### microstates_clean()

為微狀態提取預處理 EEG 資料。

```python
cleaned_eeg = nk.microstates_clean(eeg_data, sampling_rate=250)
```

**預處理步驟：**
- 帶通濾波（通常 2-20 Hz）
- 偽跡拒絕
- 壞通道內插
- 重新參考到平均

**原理：**
- 微狀態反映大規模網路活動
- 高頻和低頻偽跡可能扭曲地形

### microstates_peaks()

識別微狀態分析的 GFP 峰值。

```python
peak_indices = nk.microstates_peaks(eeg_data, sampling_rate=250)
```

**目的：**
- 微狀態通常在 GFP 峰值處分析
- 峰值代表最大、穩定地形活動的時刻
- 減少計算負擔和噪音敏感性

**回傳：**
- GFP 局部最大值的索引

### microstates_static()

計算個別微狀態的時間特性。

```python
static_metrics = nk.microstates_static(microstates)
```

**指標：**
- **持續時間（ms）**：在每個微狀態中花費的平均時間
  - 典型：80-120 毫秒
  - 反映穩定性和持久性
- **發生次數（每秒）**：微狀態出現的頻率
  - 每個狀態被進入的頻率
- **覆蓋率（%）**：在每個微狀態中的總時間百分比
  - 相對主導性
- **全域解釋變異（GEV）**：每個類別解釋的變異
  - 模板擬合的品質

**回傳：**
- 包含每個微狀態類別指標的 DataFrame

**解讀：**
- 持續時間變化：改變的網路穩定性
- 發生次數變化：移動的狀態動態
- 覆蓋率變化：特定網路的主導性

### microstates_dynamic()

分析微狀態之間的轉換模式。

```python
dynamic_metrics = nk.microstates_dynamic(microstates)
```

**指標：**
- **轉換矩陣**：從狀態 i 轉換到狀態 j 的機率
  - 揭示優先序列
- **轉換率**：整體轉換頻率
  - 較高的率：更快速的切換
- **熵**：轉換的隨機性
  - 高熵：不可預測的切換
  - 低熵：刻板的序列
- **馬可夫檢驗**：轉換是否依賴歷史？

**回傳：**
- 包含轉換統計的字典

**應用場景：**
- 識別臨床人群中異常的微狀態序列
- 網路動態和靈活性
- 狀態依賴的資訊處理

### microstates_plot()

視覺化微狀態地形和時間過程。

```python
nk.microstates_plot(microstates, eeg_data)
```

**顯示：**
- 每個微狀態類別的地形圖
- 帶有微狀態標籤的 GFP 軌跡
- 顯示狀態序列的轉換圖
- 統計摘要

## MNE 整合工具

### mne_data()

從 MNE-Python 存取範例資料集。

```python
raw = nk.mne_data(dataset='sample', directory=None)
```

**可用資料集：**
- `'sample'`：多模態（MEG/EEG）範例
- `'ssvep'`：穩態視覺誘發電位
- `'eegbci'`：運動想像 BCI 資料集

### mne_to_df() / mne_to_dict()

將 MNE 物件轉換為 NeuroKit 相容格式。

```python
df = nk.mne_to_df(raw)
data_dict = nk.mne_to_dict(epochs)
```

**應用場景：**
- 在 NeuroKit2 中處理 MNE 處理過的資料
- 在格式之間轉換以進行分析

### mne_channel_add() / mne_channel_extract()

管理 MNE 物件中的個別通道。

```python
# 提取特定通道
subset = nk.mne_channel_extract(raw, ['Fz', 'Cz', 'Pz'])

# 添加衍生通道
raw_with_eog = nk.mne_channel_add(raw, new_channel_data, ch_name='EOG')
```

### mne_crop()

按時間或樣本修剪記錄。

```python
cropped = nk.mne_crop(raw, tmin=10, tmax=100)
```

### mne_templateMRI()

為源定位提供模板解剖。

```python
subjects_dir = nk.mne_templateMRI()
```

**應用場景：**
- 沒有個人 MRI 的源分析
- 群組層級源定位
- fsaverage 模板腦

### eeg_simulate()

生成用於測試的合成 EEG 訊號。

```python
synthetic_eeg = nk.eeg_simulate(duration=60, sampling_rate=250, n_channels=32)
```

## 實務考量

### 取樣率建議
- **最低**：100 Hz 用於基本功率分析
- **標準**：250-500 Hz 用於大多數應用
- **高解析度**：1000+ Hz 用於詳細的時間動態

### 記錄時長
- **功率分析**：≥2 分鐘以獲得穩定估計
- **微狀態**：≥2-5 分鐘，越長越好
- **靜息狀態**：典型 3-10 分鐘
- **事件相關**：取決於試驗次數（每條件 ≥30 次試驗）

### 偽跡管理
- **眨眼**：使用 ICA 或迴歸移除
- **肌肉偽跡**：高通濾波（≥1 Hz）或手動拒絕
- **壞通道**：分析前偵測並內插
- **市電噪音**：在 50/60 Hz 使用陷波濾波器

### 最佳實踐

**功率分析：**
```python
# 1. 清理資料
cleaned = nk.signal_filter(eeg_data, sampling_rate=250, lowcut=0.5, highcut=45)

# 2. 識別並內插壞通道
bad = nk.eeg_badchannels(cleaned, sampling_rate=250)
# 使用 MNE 內插壞通道

# 3. 重新參考
rereferenced = nk.eeg_rereference(cleaned, reference='average')

# 4. 計算功率
power = nk.eeg_power(rereferenced, sampling_rate=250, channels=channel_list)
```

**微狀態工作流程：**
```python
# 1. 預處理
cleaned = nk.microstates_clean(eeg_data, sampling_rate=250)

# 2. 確定最佳狀態數
optimal_k = nk.microstates_findnumber(cleaned, show=True)

# 3. 分割微狀態
microstates = nk.microstates_segment(cleaned, n_microstates=optimal_k,
                                     sampling_rate=250, method='kmod')

# 4. 分類為標準標籤
microstates = nk.microstates_classify(microstates)

# 5. 計算時間指標
static = nk.microstates_static(microstates)
dynamic = nk.microstates_dynamic(microstates)

# 6. 視覺化
nk.microstates_plot(microstates, cleaned)
```

## 臨床和研究應用

**認知神經科學：**
- 注意力、工作記憶、執行功能
- 語言處理
- 感覺知覺

**臨床人群：**
- 癲癇：癲癇偵測、定位
- 阿茲海默症：EEG 減慢、微狀態改變
- 思覺失調症：改變的微狀態，特別是狀態 C
- 注意力不足過動症：增加的 theta/beta 比率
- 憂鬱症：額葉 alpha 不對稱

**意識研究：**
- 麻醉監測
- 意識障礙
- 睡眠分期

**神經回饋：**
- 即時頻帶訓練
- Alpha 增強以放鬆
- Beta 增強以專注

## 參考文獻

- Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review. Neuroimage, 180, 577-593.
- Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). Segmentation of brain electrical activity into microstates: model estimation and validation. IEEE Transactions on Biomedical Engineering, 42(7), 658-665.
- Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in neuroscience, 7, 267.
