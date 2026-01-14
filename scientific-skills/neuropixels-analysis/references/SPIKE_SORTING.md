# 尖峰分選參考

Neuropixels 資料尖峰分選的綜合指南。

## 可用分選器

| 分選器 | 需要 GPU | 速度 | 品質 | 最適用於 |
|--------|--------------|-------|---------|----------|
| **Kilosort4** | 是（CUDA） | 快 | 優秀 | 生產使用 |
| **Kilosort3** | 是（CUDA） | 快 | 非常好 | 舊版相容性 |
| **Kilosort2.5** | 是（CUDA） | 快 | 良好 | 舊版流程 |
| **SpykingCircus2** | 否 | 中 | 良好 | 僅 CPU 系統 |
| **Mountainsort5** | 否 | 中 | 良好 | 短記錄 |
| **Tridesclous2** | 否 | 中 | 良好 | 互動式分選 |

## Kilosort4（建議）

### 安裝
```bash
pip install kilosort
```

### 基本使用
```python
import spikeinterface.full as si

# 執行 Kilosort4
sorting = si.run_sorter(
    'kilosort4',
    recording,
    output_folder='ks4_output',
    verbose=True
)

print(f"Found {len(sorting.unit_ids)} units")
```

### 自訂參數
```python
sorting = si.run_sorter(
    'kilosort4',
    recording,
    output_folder='ks4_output',
    # 偵測
    Th_universal=9,        # 尖峰偵測閾值
    Th_learned=8,          # 學習閾值
    # 範本
    dmin=15,               # 範本間最小垂直距離（um）
    dminx=12,              # 最小水平距離（um）
    nblocks=5,             # 非剛性區塊數量
    # 群集
    max_channel_distance=None,  # 範本通道最大距離
    # 輸出
    do_CAR=False,          # 跳過 CAR（在預處理中完成）
    skip_kilosort_preprocessing=True,
    save_extra_kwargs=True
)
```

### Kilosort4 完整參數
```python
# 取得所有可用參數
params = si.get_default_sorter_params('kilosort4')
print(params)

# 關鍵參數：
ks4_params = {
    # 偵測
    'Th_universal': 9,      # 尖峰偵測通用閾值
    'Th_learned': 8,        # 學習範本閾值
    'spkTh': -6,            # 擷取時的尖峰閾值

    # 群集
    'dmin': 15,             # 群集間最小距離（um）
    'dminx': 12,            # 最小水平距離（um）
    'nblocks': 5,           # 非剛性漂移修正區塊

    # 範本
    'n_templates': 6,       # 每組通用範本數量
    'nt': 61,               # 範本時間樣本數

    # 效能
    'batch_size': 60000,    # 批次樣本數
    'nfilt_factor': 8,      # 濾波器數量因子
}
```

## Kilosort3

### 使用方式
```python
sorting = si.run_sorter(
    'kilosort3',
    recording,
    output_folder='ks3_output',
    # 關鍵參數
    detect_threshold=6,
    projection_threshold=[9, 9],
    preclust_threshold=8,
    car=False,  # CAR 在預處理中完成
    freq_min=300,
)
```

## SpykingCircus2（僅 CPU）

### 安裝
```bash
pip install spykingcircus
```

### 使用方式
```python
sorting = si.run_sorter(
    'spykingcircus2',
    recording,
    output_folder='sc2_output',
    # 參數
    detect_threshold=5,
    selection_method='all',
)
```

## Mountainsort5（僅 CPU）

### 安裝
```bash
pip install mountainsort5
```

### 使用方式
```python
sorting = si.run_sorter(
    'mountainsort5',
    recording,
    output_folder='ms5_output',
    # 參數
    detect_threshold=5.0,
    scheme='2',  # '1', '2' 或 '3'
)
```

## 執行多個分選器

### 比較分選器
```python
# 執行多個分選器
sorting_ks4 = si.run_sorter('kilosort4', recording, output_folder='ks4/')
sorting_sc2 = si.run_sorter('spykingcircus2', recording, output_folder='sc2/')
sorting_ms5 = si.run_sorter('mountainsort5', recording, output_folder='ms5/')

# 比較結果
comparison = si.compare_multiple_sorters(
    [sorting_ks4, sorting_sc2, sorting_ms5],
    name_list=['KS4', 'SC2', 'MS5']
)

# 取得一致性分數
agreement = comparison.get_agreement_sorting()
```

### 整合分選
```python
# 建立共識分選
sorting_ensemble = si.create_ensemble_sorting(
    [sorting_ks4, sorting_sc2, sorting_ms5],
    voting_method='agreement',
    min_agreement=2  # 單元必須被至少 2 個分選器找到
)
```

## 在 Docker/Singularity 中分選

### 使用 Docker
```python
sorting = si.run_sorter(
    'kilosort3',
    recording,
    output_folder='ks3_docker/',
    docker_image='spikeinterface/kilosort3-compiled-base:latest',
    verbose=True
)
```

### 使用 Singularity
```python
sorting = si.run_sorter(
    'kilosort3',
    recording,
    output_folder='ks3_singularity/',
    singularity_image='/path/to/kilosort3.sif',
    verbose=True
)
```

## 長時間記錄策略

### 串接記錄
```python
# 多個記錄檔案
recordings = [
    si.read_spikeglx(f'/path/to/recording_{i}', stream_id='imec0.ap')
    for i in range(3)
]

# 串接
recording_concat = si.concatenate_recordings(recordings)

# 分選
sorting = si.run_sorter('kilosort4', recording_concat, output_folder='ks4/')

# 分回原始記錄
sortings_split = si.split_sorting(sorting, recording_concat)
```

### 按片段分選
```python
# 對於非常長的記錄，分別分選各片段
from pathlib import Path

segments_output = Path('sorting_segments')
sortings = []

for i, segment in enumerate(recording.split_by_times([0, 3600, 7200, 10800])):
    sorting_seg = si.run_sorter(
        'kilosort4',
        segment,
        output_folder=segments_output / f'segment_{i}'
    )
    sortings.append(sorting_seg)
```

## 分選後篩選

### 使用 Phy 手動篩選
```python
# 匯出至 Phy 格式
analyzer = si.create_sorting_analyzer(sorting, recording)
analyzer.compute(['random_spikes', 'waveforms', 'templates'])
si.export_to_phy(analyzer, output_folder='phy_export/')

# 開啟 Phy
# 在終端執行：phy template-gui phy_export/params.py
```

### 載入 Phy 篩選
```python
# 手動篩選後
sorting_curated = si.read_phy('phy_export/')

# 或套用 Phy 標籤
sorting_curated = si.apply_phy_curation(sorting, 'phy_export/')
```

### 自動篩選
```python
# 移除低於品質閾值的單元
analyzer = si.create_sorting_analyzer(sorting, recording)
analyzer.compute('quality_metrics')

qm = analyzer.get_extension('quality_metrics').get_data()

# 定義品質標準
query = "(snr > 5) & (isi_violations_ratio < 0.01) & (presence_ratio > 0.9)"
good_unit_ids = qm.query(query).index.tolist()

sorting_clean = sorting.select_units(good_unit_ids)
print(f"Kept {len(good_unit_ids)}/{len(sorting.unit_ids)} units")
```

## 分選指標

### 檢查分選器輸出
```python
# 基本統計
print(f"Units found: {len(sorting.unit_ids)}")
print(f"Total spikes: {sorting.get_total_num_spikes()}")

# 每單元尖峰計數
for unit_id in sorting.unit_ids[:10]:
    n_spikes = len(sorting.get_unit_spike_train(unit_id))
    print(f"Unit {unit_id}: {n_spikes} spikes")
```

### 放電率
```python
# 計算放電率
duration = recording.get_total_duration()
for unit_id in sorting.unit_ids:
    n_spikes = len(sorting.get_unit_spike_train(unit_id))
    fr = n_spikes / duration
    print(f"Unit {unit_id}: {fr:.2f} Hz")
```

## 故障排除

### 常見問題

**GPU 記憶體不足**
```python
# 減少批次大小
sorting = si.run_sorter(
    'kilosort4',
    recording,
    output_folder='ks4/',
    batch_size=30000  # 較小的批次
)
```

**找到的單元太少**
```python
# 降低偵測閾值
sorting = si.run_sorter(
    'kilosort4',
    recording,
    output_folder='ks4/',
    Th_universal=7,  # 從預設 9 降低
    Th_learned=6
)
```

**單元太多（過度分割）**
```python
# 增加範本間最小距離
sorting = si.run_sorter(
    'kilosort4',
    recording,
    output_folder='ks4/',
    dmin=20,   # 從 15 增加
    dminx=16   # 從 12 增加
)
```

**檢查 GPU 可用性**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```
