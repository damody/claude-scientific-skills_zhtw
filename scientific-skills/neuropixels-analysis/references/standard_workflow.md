# 標準 Neuropixels 分析工作流程

從原始資料到篩選單元的完整逐步分析指南。

## 概述

本參考文件記錄完整的分析流程：

```
原始記錄 → 預處理 → 運動修正 → 尖峰分選 →
後處理 → 品質指標 → 篩選 → 匯出
```

## 1. 資料載入

### 支援的格式

```python
import spikeinterface.full as si
import neuropixels_analysis as npa

# SpikeGLX（最常見）
recording = si.read_spikeglx('/path/to/run/', stream_id='imec0.ap')

# Open Ephys
recording = si.read_openephys('/path/to/experiment/')

# NWB 格式
recording = si.read_nwb('/path/to/file.nwb')

# 或使用我們的便利包裝器
recording = npa.load_recording('/path/to/data/', format='spikeglx')
```

### 驗證記錄屬性

```python
# 基本屬性
print(f"Channels: {recording.get_num_channels()}")
print(f"Duration: {recording.get_total_duration():.1f}s")
print(f"Sampling rate: {recording.get_sampling_frequency()}Hz")

# 探針幾何
print(f"Probe: {recording.get_probe().name}")

# 通道位置
locations = recording.get_channel_locations()
```

## 2. 預處理

### 標準預處理鏈

```python
# 選項 1：完整流程（建議）
rec_preprocessed = npa.preprocess(recording)

# 選項 2：逐步控制
rec = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
rec = si.phase_shift(rec)  # 修正 ADC 相位
bad_channels = si.detect_bad_channels(rec)
rec = rec.remove_channels(bad_channels)
rec = si.common_reference(rec, operator='median')
rec_preprocessed = rec
```

### IBL 風格去條紋

對於有強烈偽影的記錄：

```python
from ibldsp.voltage import decompress_destripe_cbin

# IBL 去條紋（非常有效）
rec = si.highpass_filter(recording, freq_min=400)
rec = si.phase_shift(rec)
rec = si.highpass_spatial_filter(rec)  # 去條紋
rec = si.common_reference(rec, reference='global', operator='median')
```

### 儲存預處理資料

```python
# 儲存以供重複使用（加速迭代）
rec_preprocessed.save(folder='preprocessed/', n_jobs=4)
```

## 3. 運動/漂移修正

### 檢查是否需要修正

```python
# 估計運動
motion_info = npa.estimate_motion(rec_preprocessed, preset='kilosort_like')

# 視覺化漂移
npa.plot_drift(rec_preprocessed, motion_info, output='drift_map.png')

# 檢查幅度
if motion_info['motion'].max() > 10:  # 微米
    print("Significant drift detected - correction recommended")
```

### 套用修正

```python
# 基於 DREDge 的修正（預設）
rec_corrected = npa.correct_motion(
    rec_preprocessed,
    preset='nonrigid_accurate',  # 或 'kilosort_like' 以加快速度
)

# 或完全控制
from spikeinterface.preprocessing import correct_motion

rec_corrected = correct_motion(
    rec_preprocessed,
    preset='nonrigid_accurate',
    folder='motion_output/',
    output_motion=True,
)
```

## 4. 尖峰分選

### 建議：Kilosort4

```python
# 執行 Kilosort4（需要 GPU）
sorting = npa.run_sorting(
    rec_corrected,
    sorter='kilosort4',
    output_folder='sorting_KS4/',
)

# 使用自訂參數
sorting = npa.run_sorting(
    rec_corrected,
    sorter='kilosort4',
    output_folder='sorting_KS4/',
    sorter_params={
        'batch_size': 30000,
        'nblocks': 5,  # 用於非剛性漂移
        'Th_learned': 8,  # 偵測閾值
    },
)
```

### 替代分選器

```python
# SpykingCircus2（基於 CPU）
sorting = npa.run_sorting(rec_corrected, sorter='spykingcircus2')

# Mountainsort5（快速，適合短記錄）
sorting = npa.run_sorting(rec_corrected, sorter='mountainsort5')
```

### 比較多個分選器

```python
# 執行多個分選器
sortings = {}
for sorter in ['kilosort4', 'spykingcircus2']:
    sortings[sorter] = npa.run_sorting(rec_corrected, sorter=sorter)

# 比較結果
comparison = npa.compare_sorters(list(sortings.values()))
agreement_matrix = comparison.get_agreement_matrix()
```

## 5. 後處理

### 建立分析器

```python
# 建立分選分析器（所有後處理的核心物件）
analyzer = npa.create_analyzer(
    sorting,
    rec_corrected,
    output_folder='analyzer/',
)

# 計算所有標準擴充功能
analyzer = npa.postprocess(
    sorting,
    rec_corrected,
    output_folder='analyzer/',
    compute_all=True,  # 波形、範本、指標等
)
```

### 計算個別擴充功能

```python
# 波形
analyzer.compute('waveforms', ms_before=1.0, ms_after=2.0, max_spikes_per_unit=500)

# 範本
analyzer.compute('templates', operators=['average', 'std'])

# 尖峰振幅
analyzer.compute('spike_amplitudes')

# 相關圖
analyzer.compute('correlograms', window_ms=50.0, bin_ms=1.0)

# 單元位置
analyzer.compute('unit_locations', method='monopolar_triangulation')

# 尖峰位置
analyzer.compute('spike_locations', method='center_of_mass')
```

## 6. 品質指標

### 計算所有指標

```python
# 計算綜合指標
metrics = npa.compute_quality_metrics(
    analyzer,
    metric_names=[
        'snr',
        'isi_violations_ratio',
        'presence_ratio',
        'amplitude_cutoff',
        'firing_rate',
        'amplitude_cv',
        'sliding_rp_violation',
        'd_prime',
        'nearest_neighbor',
    ],
)

# 查看指標
print(metrics.head())
```

### 關鍵指標說明

| 指標 | 良好值 | 描述 |
|--------|------------|-------------|
| `snr` | > 5 | 訊雜比 |
| `isi_violations_ratio` | < 0.01 | 反應期違規 |
| `presence_ratio` | > 0.9 | 有尖峰的記錄比例 |
| `amplitude_cutoff` | < 0.1 | 估計遺漏的尖峰 |
| `firing_rate` | > 0.1 Hz | 平均放電率 |

## 7. 篩選

### 自動篩選

```python
# Allen Institute 標準
labels = npa.curate(metrics, method='allen')

# IBL 標準
labels = npa.curate(metrics, method='ibl')

# 自訂閾值
labels = npa.curate(
    metrics,
    snr_threshold=5,
    isi_violations_threshold=0.01,
    presence_threshold=0.9,
)
```

### AI 輔助篩選

```python
from anthropic import Anthropic

# 設定 API
client = Anthropic()

# 對不確定單元進行視覺分析
uncertain = metrics.query('snr > 3 and snr < 8').index.tolist()

for unit_id in uncertain:
    result = npa.analyze_unit_visually(analyzer, unit_id, api_client=client)
    labels[unit_id] = result['classification']
```

### 互動式篩選會話

```python
# 建立會話
session = npa.CurationSession.create(analyzer, output_dir='curation/')

# 審查單元
while session.current_unit():
    unit = session.current_unit()
    report = npa.generate_unit_report(analyzer, unit.unit_id)

    # 您的決策
    decision = input(f"Unit {unit.unit_id}: ")
    session.set_decision(unit.unit_id, decision)
    session.next_unit()

# 匯出
labels = session.get_final_labels()
```

## 8. 匯出結果

### 匯出至 Phy

```python
from spikeinterface.exporters import export_to_phy

export_to_phy(
    analyzer,
    output_folder='phy_export/',
    copy_binary=True,
)
```

### 匯出至 NWB

```python
from spikeinterface.exporters import export_to_nwb

export_to_nwb(
    analyzer,
    nwbfile_path='results.nwb',
    metadata={
        'session_description': 'Neuropixels recording',
        'experimenter': 'Lab Name',
    },
)
```

### 儲存品質摘要

```python
# 儲存指標 CSV
metrics.to_csv('quality_metrics.csv')

# 儲存標籤
import json
with open('curation_labels.json', 'w') as f:
    json.dump(labels, f, indent=2)

# 產生摘要報告
npa.plot_quality_metrics(analyzer, metrics, output='quality_summary.png')
```

## 完整流程範例

```python
import neuropixels_analysis as npa

# 載入
recording = npa.load_recording('/data/experiment/', format='spikeglx')

# 預處理
rec = npa.preprocess(recording)

# 運動修正
rec = npa.correct_motion(rec)

# 分選
sorting = npa.run_sorting(rec, sorter='kilosort4')

# 後處理
analyzer, metrics = npa.postprocess(sorting, rec)

# 篩選
labels = npa.curate(metrics, method='allen')

# 匯出良好單元
good_units = [uid for uid, label in labels.items() if label == 'good']
print(f"Good units: {len(good_units)}/{len(labels)}")
```

## 成功秘訣

1. **務必視覺化漂移** - 決定運動修正前先檢查
2. **儲存預處理資料** - 避免重複計算
3. **比較多個分選器** - 對關鍵實驗尤其重要
4. **手動審查不確定單元** - 不要盲目信任自動篩選
5. **記錄您的參數** - 確保可重現性
6. **使用 GPU** 執行 Kilosort4（比 CPU 替代方案快 10-50 倍）
