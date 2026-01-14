# PyHealth 資料預處理與處理器

## 概述

PyHealth 提供全面的資料處理工具，將原始醫療資料轉換為模型可用的格式。處理器負責特徵提取、序列處理、訊號轉換和標籤準備。

## 處理器基礎類別

所有處理器繼承自 `Processor`，具有標準介面：

**關鍵方法：**
- `__call__()`：轉換輸入資料
- `get_input_info()`：回傳處理後的輸入 schema
- `get_output_info()`：回傳處理後的輸出 schema

## 核心處理器類型

### 特徵處理器

**FeatureProcessor**（`FeatureProcessor`）
- 特徵提取的基礎類別
- 處理詞彙表建構
- 嵌入準備
- 特徵編碼

**常見操作：**
- 醫療代碼分詞
- 類別編碼
- 特徵標準化
- 缺失值處理

**使用方式：**
```python
from pyhealth.data import FeatureProcessor

processor = FeatureProcessor(
    vocabulary="diagnoses",
    min_freq=5,  # 最小代碼頻率
    max_vocab_size=10000
)

processed_features = processor(raw_features)
```

### 序列處理器

**SequenceProcessor**（`SequenceProcessor`）
- 處理序列臨床事件
- 保留時間順序
- 序列填充/截斷
- 時間間隔編碼

**關鍵特徵：**
- 變長序列處理
- 時間特徵提取
- 序列統計計算

**參數：**
- `max_seq_length`：最大序列長度（超過則截斷）
- `padding`：填充策略（"pre" 或 "post"）
- `truncating`：截斷策略（"pre" 或 "post"）

**使用方式：**
```python
from pyhealth.data import SequenceProcessor

processor = SequenceProcessor(
    max_seq_length=100,
    padding="post",
    truncating="post"
)

# 處理診斷序列
processed_seq = processor(diagnosis_sequences)
```

**NestedSequenceProcessor**（`NestedSequenceProcessor`）
- 處理階層序列（例如包含事件的就診）
- 兩層處理（就診層級和事件層級）
- 保留巢狀結構

**使用情境：**
- 包含多個事件的 EHR 就診
- 多層時間建模
- 階層注意力模型

**結構：**
```python
# 輸入：[[visit1_events], [visit2_events], ...]
# 輸出：經適當填充的處理後巢狀序列
```

### 數值資料處理器

**NestedFloatsProcessor**（`NestedFloatsProcessor`）
- 處理巢狀數值陣列
- 檢驗值、生命徵象、測量值
- 多層數值特徵

**操作：**
- 標準化
- 正規化
- 缺失值填補
- 離群值處理

**使用方式：**
```python
from pyhealth.data import NestedFloatsProcessor

processor = NestedFloatsProcessor(
    normalization="z-score",  # 或 "min-max"
    fill_missing="mean"  # 填補策略
)

processed_labs = processor(lab_values)
```

**TensorProcessor**（`TensorProcessor`）
- 將資料轉換為 PyTorch 張量
- 類型處理（long、float 等）
- 裝置配置（CPU/GPU）

**參數：**
- `dtype`：張量資料類型
- `device`：運算裝置

### 時間序列處理器

**TimeseriesProcessor**（`TimeseriesProcessor`）
- 處理帶時間戳記的時間資料
- 時間間隔計算
- 時間特徵工程
- 不規則採樣處理

**提取的特徵：**
- 距前一事件的時間
- 距下一事件的時間
- 事件頻率
- 時間模式

**使用方式：**
```python
from pyhealth.data import TimeseriesProcessor

processor = TimeseriesProcessor(
    time_unit="hour",  # "day"、"hour"、"minute"
    compute_gaps=True,
    compute_frequency=True
)

processed_ts = processor(timestamps, events)
```

**SignalProcessor**（`SignalProcessor`）
- 生理訊號處理
- EEG、ECG、PPG 訊號
- 濾波與預處理

**操作：**
- 帶通濾波
- 偽影移除
- 分段
- 特徵提取（頻率、振幅）

**使用方式：**
```python
from pyhealth.data import SignalProcessor

processor = SignalProcessor(
    sampling_rate=256,  # Hz
    bandpass_filter=(0.5, 50),  # Hz 範圍
    segment_length=30  # 秒
)

processed_signal = processor(raw_eeg_signal)
```

### 影像處理器

**ImageProcessor**（`ImageProcessor`）
- 醫學影像預處理
- 標準化與調整大小
- 資料增強支援
- 格式標準化

**操作：**
- 調整至標準尺寸
- 標準化（均值/標準差）
- 視窗調整（用於 CT/MRI）
- 資料增強

**使用方式：**
```python
from pyhealth.data import ImageProcessor

processor = ImageProcessor(
    image_size=(224, 224),
    normalization="imagenet",  # 或自訂均值/標準差
    augmentation=True
)

processed_image = processor(raw_image)
```

## 標籤處理器

### 二元分類

**BinaryLabelProcessor**（`BinaryLabelProcessor`）
- 二元分類標籤（0/1）
- 處理正/負類別
- 類別加權處理不平衡

**使用方式：**
```python
from pyhealth.data import BinaryLabelProcessor

processor = BinaryLabelProcessor(
    positive_class=1,
    class_weight="balanced"
)

processed_labels = processor(raw_labels)
```

### 多類別分類

**MultiClassLabelProcessor**（`MultiClassLabelProcessor`）
- 多類別分類（互斥類別）
- 標籤編碼
- 類別平衡

**參數：**
- `num_classes`：類別數量
- `class_weight`：加權策略

**使用方式：**
```python
from pyhealth.data import MultiClassLabelProcessor

processor = MultiClassLabelProcessor(
    num_classes=5,  # 例如睡眠階段：W、N1、N2、N3、REM
    class_weight="balanced"
)

processed_labels = processor(raw_labels)
```

### 多標籤分類

**MultiLabelProcessor**（`MultiLabelProcessor`）
- 多標籤分類（每個樣本多個標籤）
- 每個標籤的二元編碼
- 標籤共現處理

**使用情境：**
- 藥物推薦（多種藥物）
- ICD 編碼（多個診斷）
- 共病預測

**使用方式：**
```python
from pyhealth.data import MultiLabelProcessor

processor = MultiLabelProcessor(
    num_labels=100,  # 可能的標籤總數
    threshold=0.5  # 預測閾值
)

processed_labels = processor(raw_label_sets)
```

### 迴歸

**RegressionLabelProcessor**（`RegressionLabelProcessor`）
- 連續值預測
- 目標縮放與標準化
- 離群值處理

**使用情境：**
- 住院天數預測
- 檢驗值預測
- 風險分數估計

**使用方式：**
```python
from pyhealth.data import RegressionLabelProcessor

processor = RegressionLabelProcessor(
    normalization="z-score",  # 或 "min-max"
    clip_outliers=True,
    outlier_std=3  # 在 3 個標準差處截斷
)

processed_targets = processor(raw_values)
```

## 專門處理器

### 文字處理

**TextProcessor**（`TextProcessor`）
- 臨床文字預處理
- 分詞
- 詞彙表建構
- 序列編碼

**操作：**
- 轉小寫
- 標點移除
- 醫療縮寫處理
- 詞頻過濾

**使用方式：**
```python
from pyhealth.data import TextProcessor

processor = TextProcessor(
    tokenizer="word",  # 或 "sentencepiece"、"bpe"
    lowercase=True,
    max_vocab_size=50000,
    min_freq=5
)

processed_text = processor(clinical_notes)
```

### 模型專用處理器

**StageNetProcessor**（`StageNetProcessor`）
- StageNet 模型專用預處理
- 區塊式序列處理
- 階段感知特徵提取

**使用方式：**
```python
from pyhealth.data import StageNetProcessor

processor = StageNetProcessor(
    chunk_size=128,
    num_stages=3
)

processed_data = processor(sequential_data)
```

**StageNetTensorProcessor**（`StageNetTensorProcessor`）
- StageNet 的張量轉換
- 適當的批次處理與填充
- 階段遮罩生成

### 原始資料處理

**RawProcessor**（`RawProcessor`）
- 最小預處理
- 預處理資料的直接傳遞
- 自訂預處理場景

**使用方式：**
```python
from pyhealth.data import RawProcessor

processor = RawProcessor()
processed_data = processor(data)  # 最小轉換
```

## 樣本層級處理

**SampleProcessor**（`SampleProcessor`）
- 處理完整樣本（輸入 + 輸出）
- 協調多個處理器
- 端到端預處理管線

**工作流程：**
1. 對特徵套用輸入處理器
2. 對標籤套用輸出處理器
3. 組合成模型可用的樣本

**使用方式：**
```python
from pyhealth.data import SampleProcessor

processor = SampleProcessor(
    input_processors={
        "diagnoses": SequenceProcessor(max_seq_length=50),
        "medications": SequenceProcessor(max_seq_length=30),
        "labs": NestedFloatsProcessor(normalization="z-score")
    },
    output_processor=BinaryLabelProcessor()
)

processed_sample = processor(raw_sample)
```

## 資料集層級處理

**DatasetProcessor**（`DatasetProcessor`）
- 處理整個資料集
- 批次處理
- 平行處理支援
- 快取提升效率

**操作：**
- 對所有樣本套用處理器
- 從資料集生成詞彙表
- 計算資料集統計
- 儲存處理後的資料

**使用方式：**
```python
from pyhealth.data import DatasetProcessor

processor = DatasetProcessor(
    sample_processor=sample_processor,
    num_workers=4,  # 平行處理
    cache_dir="/path/to/cache"
)

processed_dataset = processor(raw_dataset)
```

## 常見預處理工作流程

### 工作流程 1：EHR 死亡率預測

```python
from pyhealth.data import (
    SequenceProcessor,
    BinaryLabelProcessor,
    SampleProcessor
)

# 定義處理器
input_processors = {
    "diagnoses": SequenceProcessor(max_seq_length=50),
    "medications": SequenceProcessor(max_seq_length=30),
    "procedures": SequenceProcessor(max_seq_length=20)
}

output_processor = BinaryLabelProcessor(class_weight="balanced")

# 組合成樣本處理器
sample_processor = SampleProcessor(
    input_processors=input_processors,
    output_processor=output_processor
)

# 處理資料集
processed_samples = [sample_processor(s) for s in raw_samples]
```

### 工作流程 2：腦電圖睡眠分期

```python
from pyhealth.data import (
    SignalProcessor,
    MultiClassLabelProcessor,
    SampleProcessor
)

# 訊號預處理
signal_processor = SignalProcessor(
    sampling_rate=100,
    bandpass_filter=(0.3, 35),  # EEG 頻率範圍
    segment_length=30  # 30 秒 epoch
)

# 標籤處理
label_processor = MultiClassLabelProcessor(
    num_classes=5,  # W、N1、N2、N3、REM
    class_weight="balanced"
)

# 組合
sample_processor = SampleProcessor(
    input_processors={"signal": signal_processor},
    output_processor=label_processor
)
```

### 工作流程 3：藥物推薦

```python
from pyhealth.data import (
    SequenceProcessor,
    MultiLabelProcessor,
    SampleProcessor
)

# 輸入處理
input_processors = {
    "diagnoses": SequenceProcessor(max_seq_length=50),
    "previous_medications": SequenceProcessor(max_seq_length=40)
}

# 多標籤輸出（多種藥物）
output_processor = MultiLabelProcessor(
    num_labels=150,  # 可能的藥物數量
    threshold=0.5
)

sample_processor = SampleProcessor(
    input_processors=input_processors,
    output_processor=output_processor
)
```

### 工作流程 4：住院天數預測

```python
from pyhealth.data import (
    SequenceProcessor,
    NestedFloatsProcessor,
    RegressionLabelProcessor,
    SampleProcessor
)

# 處理不同特徵類型
input_processors = {
    "diagnoses": SequenceProcessor(max_seq_length=30),
    "procedures": SequenceProcessor(max_seq_length=20),
    "labs": NestedFloatsProcessor(
        normalization="z-score",
        fill_missing="mean"
    )
}

# 迴歸目標
output_processor = RegressionLabelProcessor(
    normalization="log",  # 對住院天數進行對數轉換
    clip_outliers=True
)

sample_processor = SampleProcessor(
    input_processors=input_processors,
    output_processor=output_processor
)
```

## 最佳實務

### 序列處理

1. **選擇適當的 max_seq_length**：平衡上下文與運算
   - 短序列（20-50）：快速，較少上下文
   - 中序列（50-100）：良好平衡
   - 長序列（100+）：更多上下文，較慢

2. **截斷策略**：
   - "post"：保留最近事件（推薦用於臨床預測）
   - "pre"：保留最早事件

3. **填充策略**：
   - "post"：在末尾填充（標準）
   - "pre"：在開頭填充

### 特徵編碼

1. **詞彙表大小**：限制為常見代碼
   - `min_freq=5`：包含出現 ≥5 次的代碼
   - `max_vocab_size=10000`：限制總詞彙表大小

2. **處理罕見代碼**：分組到「未知」類別

3. **缺失值**：
   - 填補（均值、中位數、前向填充）
   - 指標變數
   - 特殊 token

### 標準化

1. **數值特徵**：務必標準化
   - Z-score：標準縮放（均值=0，標準差=1）
   - Min-max：範圍縮放 [0, 1]

2. **僅在訓練集上計算統計量**：防止資料洩漏

3. **對驗證/測試集套用相同標準化**

### 類別不平衡

1. **使用類別加權**：`class_weight="balanced"`

2. **考慮過採樣**：用於極罕見的正例

3. **使用適當的指標評估**：AUROC、AUPRC、F1

### 效能優化

1. **快取處理後的資料**：儲存預處理結果

2. **平行處理**：在 DataLoader 中使用 `num_workers`

3. **批次處理**：一次處理多個樣本

4. **特徵選擇**：移除低資訊特徵

### 驗證

1. **檢查處理後的形狀**：確保正確維度

2. **驗證數值範圍**：標準化後

3. **檢視樣本**：手動審查處理後的資料

4. **監控記憶體使用**：尤其是大型資料集

## 疑難排解

### 常見問題

**記憶體錯誤：**
- 降低 `max_seq_length`
- 使用較小批次
- 分塊處理資料
- 啟用磁碟快取

**處理緩慢：**
- 啟用平行處理（`num_workers`）
- 快取預處理資料
- 降低特徵維度
- 使用更高效的資料類型

**形狀不匹配：**
- 檢查序列長度
- 驗證填充配置
- 確保一致的處理器設定

**NaN 值：**
- 明確處理缺失資料
- 檢查標準化參數
- 驗證填補策略

**類別不平衡：**
- 使用類別加權
- 考慮過採樣
- 調整決策閾值
- 使用適當的評估指標
