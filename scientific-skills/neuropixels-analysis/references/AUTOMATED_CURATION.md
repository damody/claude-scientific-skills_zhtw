# 自動篩選參考

使用 Bombcell、UnitRefine 和其他工具進行自動尖峰分選篩選的指南。

## 為何需要自動篩選？

手動篩選：
- **緩慢**：每個記錄會話需要數小時
- **主觀**：評分者間變異性
- **不可重現**：難以標準化

自動工具提供一致、可重現的品質分類。

## 可用工具

| 工具 | 分類 | 語言 | 整合 |
|------|---------------|----------|-------------|
| **Bombcell** | 4 類（單一/多/雜訊/非軀體） | Python/MATLAB | SpikeInterface、Phy |
| **UnitRefine** | 基於機器學習 | Python | SpikeInterface |
| **SpikeInterface QM** | 基於閾值 | Python | 原生 |
| **UnitMatch** | 跨會話追蹤 | Python/MATLAB | Kilosort、Bombcell |

## Bombcell

### 概述

Bombcell 將單元分為 4 類：
1. **單一軀體單元** - 分離良好的單一神經元
2. **多單元活動（MUA）** - 混合神經元訊號
3. **雜訊** - 非神經偽影
4. **非軀體** - 軸突或樹突訊號

### 安裝

```bash
# Python
pip install bombcell

# 或開發版本
git clone https://github.com/Julie-Fabre/bombcell.git
cd bombcell/py_bombcell
pip install -e .
```

### 基本使用（Python）

```python
import bombcell as bc

# 載入已分選資料（Kilosort 輸出）
kilosort_folder = '/path/to/kilosort/output'
raw_data_path = '/path/to/recording.ap.bin'

# 執行 Bombcell
results = bc.run_bombcell(
    kilosort_folder,
    raw_data_path,
    sample_rate=30000,
    n_channels=384
)

# 取得分類
unit_labels = results['unit_labels']
# 'good' = 單一單元, 'mua' = 多單元, 'noise' = 雜訊
```

### 與 SpikeInterface 整合

```python
import spikeinterface.full as si

# 尖峰分選後
sorting = si.run_sorter('kilosort4', recording, output_folder='ks4/')

# 建立分析器並計算所需擴充功能
analyzer = si.create_sorting_analyzer(sorting, recording, sparse=True)
analyzer.compute('waveforms')
analyzer.compute('templates')
analyzer.compute('spike_amplitudes')

# 匯出至 Phy 格式（Bombcell 可讀取此格式）
si.export_to_phy(analyzer, output_folder='phy_export/')

# 在 Phy 匯出上執行 Bombcell
import bombcell as bc
results = bc.run_bombcell_phy('phy_export/')
```

### Bombcell 指標

Bombcell 計算用於分類的特定指標：

| 指標 | 描述 | 用途 |
|--------|-------------|----------|
| `peak_trough_ratio` | 波形形狀 | 軀體 vs 非軀體 |
| `spatial_decay` | 跨通道振幅 | 雜訊偵測 |
| `refractory_period_violations` | ISI 違規 | 單一 vs 多 |
| `presence_ratio` | 時間穩定性 | 單元品質 |
| `waveform_duration` | 峰谷時間 | 細胞類型 |

### 自訂閾值

```python
# 自訂分類閾值
custom_params = {
    'isi_threshold': 0.01,          # ISI 違規閾值
    'presence_threshold': 0.9,       # 最小存在比率
    'amplitude_threshold': 20,       # 最小振幅（μV）
    'spatial_decay_threshold': 40,   # 空間衰減（μm）
}

results = bc.run_bombcell(
    kilosort_folder,
    raw_data_path,
    **custom_params
)
```

## SpikeInterface 自動篩選

### 基於閾值的篩選

```python
# 計算品質指標
analyzer.compute('quality_metrics')
qm = analyzer.get_extension('quality_metrics').get_data()

# 定義篩選函數
def auto_curate(qm):
    labels = {}
    for unit_id in qm.index:
        row = qm.loc[unit_id]

        # 分類邏輯
        if row['snr'] < 2 or row['presence_ratio'] < 0.5:
            labels[unit_id] = 'noise'
        elif row['isi_violations_ratio'] > 0.1:
            labels[unit_id] = 'mua'
        elif (row['snr'] > 5 and
              row['isi_violations_ratio'] < 0.01 and
              row['presence_ratio'] > 0.9):
            labels[unit_id] = 'good'
        else:
            labels[unit_id] = 'unsorted'

    return labels

unit_labels = auto_curate(qm)

# 按標籤篩選
good_unit_ids = [u for u, l in unit_labels.items() if l == 'good']
sorting_curated = sorting.select_units(good_unit_ids)
```

### 使用 SpikeInterface 篩選模組

```python
from spikeinterface.curation import (
    CurationSorting,
    MergeUnitsSorting,
    SplitUnitSorting
)

# 包裝分選以進行篩選
curation = CurationSorting(sorting)

# 移除雜訊單元
noise_units = qm[qm['snr'] < 2].index.tolist()
curation.remove_units(noise_units)

# 合併相似單元（基於範本相似度）
analyzer.compute('template_similarity')
similarity = analyzer.get_extension('template_similarity').get_data()

# 找到高度相似的配對
import numpy as np
threshold = 0.9
similar_pairs = np.argwhere(similarity > threshold)
# 合併配對（需謹慎 - 需要手動審查）

# 取得篩選後的分選
sorting_curated = curation.to_sorting()
```

## UnitMatch：跨會話追蹤

追蹤跨記錄天數的相同神經元。

### 安裝

```bash
pip install unitmatch
# 或從原始碼
git clone https://github.com/EnnyvanBeest/UnitMatch.git
```

### 使用方式

```python
# 在多個會話上執行 Bombcell 後
session_folders = [
    '/path/to/session1/kilosort/',
    '/path/to/session2/kilosort/',
    '/path/to/session3/kilosort/',
]

from unitmatch import UnitMatch

# 執行 UnitMatch
um = UnitMatch(session_folders)
um.run()

# 取得配對結果
matches = um.get_matches()
# 回傳跨會話配對單元 ID 的 DataFrame

# 分配唯一 ID
unique_ids = um.get_unique_ids()
```

### 與工作流程整合

```python
# 典型工作流程：
# 1. 對每個會話進行尖峰分選
# 2. 執行 Bombcell 進行品質控制
# 3. 執行 UnitMatch 進行跨會話追蹤

# 會話 1
sorting1 = si.run_sorter('kilosort4', rec1, output_folder='session1/ks4/')
# 執行 Bombcell
labels1 = bc.run_bombcell('session1/ks4/', raw1_path)

# 會話 2
sorting2 = si.run_sorter('kilosort4', rec2, output_folder='session2/ks4/')
labels2 = bc.run_bombcell('session2/ks4/', raw2_path)

# 追蹤跨會話的單元
um = UnitMatch(['session1/ks4/', 'session2/ks4/'])
matches = um.get_matches()
```

## 半自動工作流程

結合自動和手動篩選：

```python
# 步驟 1：自動分類
analyzer.compute('quality_metrics')
qm = analyzer.get_extension('quality_metrics').get_data()

# 自動標記明顯案例
auto_labels = {}
for unit_id in qm.index:
    row = qm.loc[unit_id]
    if row['snr'] < 1.5:
        auto_labels[unit_id] = 'noise'
    elif row['snr'] > 8 and row['isi_violations_ratio'] < 0.005:
        auto_labels[unit_id] = 'good'
    else:
        auto_labels[unit_id] = 'needs_review'

# 步驟 2：匯出不確定單元以供手動審查
needs_review = [u for u, l in auto_labels.items() if l == 'needs_review']

# 僅匯出不確定單元至 Phy
sorting_review = sorting.select_units(needs_review)
analyzer_review = si.create_sorting_analyzer(sorting_review, recording)
analyzer_review.compute('waveforms')
analyzer_review.compute('templates')
si.export_to_phy(analyzer_review, output_folder='phy_review/')

# 在 Phy 中手動審查：phy template-gui phy_review/params.py

# 步驟 3：載入手動標籤並合併
manual_labels = si.read_phy('phy_review/').get_property('quality')
# 合併自動 + 手動標籤以取得最終結果
```

## 方法比較

| 方法 | 優點 | 缺點 |
|--------|------|------|
| **手動（Phy）** | 黃金標準、靈活 | 緩慢、主觀 |
| **SpikeInterface QM** | 快速、可重現 | 僅簡單閾值 |
| **Bombcell** | 多類別、已驗證 | 需要波形擷取 |
| **UnitRefine** | 基於 ML、從資料學習 | 需要訓練資料 |

## 最佳實踐

1. **務必視覺化** - 不要盲目信任自動結果
2. **記錄閾值** - 記錄使用的確切參數
3. **驗證** - 在子集上比較自動與手動結果
4. **保守原則** - 有疑慮時排除該單元
5. **報告方法** - 在發表文章中包含篩選標準

## 流程範例

```python
def curate_sorting(sorting, recording, output_dir):
    """完整篩選流程。"""

    # 建立分析器
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=True,
                                          folder=f'{output_dir}/analyzer')

    # 計算所需擴充功能
    analyzer.compute('random_spikes', max_spikes_per_unit=500)
    analyzer.compute('waveforms')
    analyzer.compute('templates')
    analyzer.compute('noise_levels')
    analyzer.compute('spike_amplitudes')
    analyzer.compute('quality_metrics')

    qm = analyzer.get_extension('quality_metrics').get_data()

    # 自動分類
    labels = {}
    for unit_id in qm.index:
        row = qm.loc[unit_id]

        if row['snr'] < 2:
            labels[unit_id] = 'noise'
        elif row['isi_violations_ratio'] > 0.1 or row['presence_ratio'] < 0.8:
            labels[unit_id] = 'mua'
        elif (row['snr'] > 5 and
              row['isi_violations_ratio'] < 0.01 and
              row['presence_ratio'] > 0.9 and
              row['amplitude_cutoff'] < 0.1):
            labels[unit_id] = 'good'
        else:
            labels[unit_id] = 'unsorted'

    # 摘要
    from collections import Counter
    print("Classification summary:")
    print(Counter(labels.values()))

    # 儲存標籤
    import json
    with open(f'{output_dir}/unit_labels.json', 'w') as f:
        json.dump(labels, f)

    # 回傳良好單元
    good_ids = [u for u, l in labels.items() if l == 'good']
    return sorting.select_units(good_ids), labels

# 使用方式
sorting_curated, labels = curate_sorting(sorting, recording, 'output/')
```

## 參考資料

- [Bombcell GitHub](https://github.com/Julie-Fabre/bombcell)
- [UnitMatch GitHub](https://github.com/EnnyvanBeest/UnitMatch)
- [SpikeInterface Curation](https://spikeinterface.readthedocs.io/en/stable/modules/curation.html)
- Fabre et al. (2023) "Bombcell: automated curation and cell classification"
- van Beest et al. (2024) "UnitMatch: tracking neurons across days with high-density probes"
