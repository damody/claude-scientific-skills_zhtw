# 後處理與分析參考

已分選 Neuropixels 資料的品質指標、視覺化和分析綜合指南。

## 分選分析器

`SortingAnalyzer` 是後處理的核心物件。

### 建立分析器
```python
import spikeinterface.full as si

# 建立分析器
analyzer = si.create_sorting_analyzer(
    sorting,
    recording,
    sparse=True,                    # 使用稀疏表示
    format='binary_folder',         # 儲存格式
    folder='analyzer_output'        # 儲存位置
)
```

### 計算擴充功能
```python
# 計算所有標準擴充功能
analyzer.compute('random_spikes')       # 隨機尖峰選取
analyzer.compute('waveforms')           # 擷取波形
analyzer.compute('templates')           # 計算範本
analyzer.compute('noise_levels')        # 雜訊估計
analyzer.compute('principal_components')  # PCA
analyzer.compute('spike_amplitudes')    # 每個尖峰的振幅
analyzer.compute('correlograms')        # 自相關/互相關圖
analyzer.compute('unit_locations')      # 單元位置
analyzer.compute('spike_locations')     # 每個尖峰的位置
analyzer.compute('template_similarity') # 範本相似度矩陣
analyzer.compute('quality_metrics')     # 品質指標

# 或一次計算多個
analyzer.compute([
    'random_spikes', 'waveforms', 'templates', 'noise_levels',
    'principal_components', 'spike_amplitudes', 'correlograms',
    'unit_locations', 'quality_metrics'
])
```

### 儲存與載入
```python
# 儲存
analyzer.save_as(folder='analyzer_saved', format='binary_folder')

# 載入
analyzer = si.load_sorting_analyzer('analyzer_saved')
```

## 品質指標

### 計算指標
```python
analyzer.compute('quality_metrics')
qm = analyzer.get_extension('quality_metrics').get_data()
print(qm)
```

### 可用指標

| 指標 | 描述 | 良好值 |
|--------|-------------|-------------|
| `snr` | 訊雜比 | > 5 |
| `isi_violations_ratio` | ISI 違規比率 | < 0.01 (1%) |
| `isi_violations_count` | ISI 違規計數 | 低 |
| `presence_ratio` | 有尖峰的記錄比例 | > 0.9 |
| `firing_rate` | 每秒尖峰數 | 0.1-50 Hz |
| `amplitude_cutoff` | 估計遺漏的尖峰 | < 0.1 |
| `amplitude_median` | 尖峰振幅中位數 | - |
| `amplitude_cv` | 變異係數 | < 0.5 |
| `drift_ptp` | 峰對峰漂移（um） | < 40 |
| `drift_std` | 漂移標準差 | < 10 |
| `drift_mad` | 中位數絕對離差 | < 10 |
| `sliding_rp_violation` | 滑動反應期違規 | < 0.05 |
| `sync_spike_2` | 與其他單元的同步性 | < 0.5 |
| `isolation_distance` | 馬氏距離 | > 20 |
| `l_ratio` | L 比率（分離度） | < 0.1 |
| `d_prime` | 可辨別性 | > 5 |
| `nn_hit_rate` | 最近鄰命中率 | > 0.9 |
| `nn_miss_rate` | 最近鄰遺漏率 | < 0.1 |
| `silhouette_score` | 群集輪廓係數 | > 0.5 |

### 計算特定指標
```python
analyzer.compute(
    'quality_metrics',
    metric_names=['snr', 'isi_violations_ratio', 'presence_ratio', 'firing_rate']
)
```

### 自訂品質閾值
```python
qm = analyzer.get_extension('quality_metrics').get_data()

# 定義品質標準
quality_criteria = {
    'snr': ('>', 5),
    'isi_violations_ratio': ('<', 0.01),
    'presence_ratio': ('>', 0.9),
    'firing_rate': ('>', 0.1),
    'amplitude_cutoff': ('<', 0.1),
}

# 篩選良好單元
good_units = qm.query(
    "(snr > 5) & (isi_violations_ratio < 0.01) & (presence_ratio > 0.9)"
).index.tolist()

print(f"Good units: {len(good_units)}/{len(qm)}")
```

## 波形與範本

### 擷取波形
```python
analyzer.compute('waveforms', ms_before=1.5, ms_after=2.5, max_spikes_per_unit=500)

# 取得單元的波形
waveforms = analyzer.get_extension('waveforms').get_waveforms(unit_id=0)
print(f"Shape: {waveforms.shape}")  # (n_spikes, n_samples, n_channels)
```

### 計算範本
```python
analyzer.compute('templates', operators=['average', 'std', 'median'])

# 取得範本
templates_ext = analyzer.get_extension('templates')
template = templates_ext.get_unit_template(unit_id=0, operator='average')
```

### 範本相似度
```python
analyzer.compute('template_similarity')
sim = analyzer.get_extension('template_similarity').get_data()
# 範本之間餘弦相似度矩陣
```

## 單元位置

### 計算位置
```python
analyzer.compute('unit_locations', method='monopolar_triangulation')
locations = analyzer.get_extension('unit_locations').get_data()
print(locations)  # 每個單元的 x, y 座標
```

### 尖峰位置
```python
analyzer.compute('spike_locations', method='center_of_mass')
spike_locs = analyzer.get_extension('spike_locations').get_data()
```

### 位置方法
- `'center_of_mass'` - 快速，較不準確
- `'monopolar_triangulation'` - 較準確，較慢
- `'grid_convolution'` - 良好平衡

## 相關圖

### 自相關圖
```python
analyzer.compute('correlograms', window_ms=50, bin_ms=1)
correlograms, bins = analyzer.get_extension('correlograms').get_data()

# correlograms 形狀：(n_units, n_units, n_bins)
# 單元 i 的自相關圖：correlograms[i, i, :]
# 單元 i,j 的互相關圖：correlograms[i, j, :]
```

## 視覺化

### 探針圖
```python
si.plot_probe_map(recording, with_channel_ids=True)
```

### 單元範本
```python
# 所有單元
si.plot_unit_templates(analyzer)

# 特定單元
si.plot_unit_templates(analyzer, unit_ids=[0, 1, 2])
```

### 波形
```python
# 繪製波形與範本
si.plot_unit_waveforms(analyzer, unit_ids=[0])

# 波形密度
si.plot_unit_waveforms_density_map(analyzer, unit_id=0)
```

### 光柵圖
```python
si.plot_rasters(sorting, time_range=(0, 10))  # 前 10 秒
```

### 振幅
```python
analyzer.compute('spike_amplitudes')
si.plot_amplitudes(analyzer)

# 分布
si.plot_all_amplitudes_distributions(analyzer)
```

### 相關圖
```python
# 自相關圖
si.plot_autocorrelograms(analyzer, unit_ids=[0, 1, 2])

# 互相關圖
si.plot_crosscorrelograms(analyzer, unit_ids=[0, 1])
```

### 品質指標
```python
# 摘要圖
si.plot_quality_metrics(analyzer)

# 特定指標分布
import matplotlib.pyplot as plt
qm = analyzer.get_extension('quality_metrics').get_data()
plt.hist(qm['snr'], bins=50)
plt.xlabel('SNR')
plt.ylabel('Count')
```

### 探針上的單元位置
```python
si.plot_unit_locations(analyzer)
```

### 漂移圖
```python
si.plot_drift_raster(sorting, recording)
```

### 摘要圖
```python
# 綜合單元摘要
si.plot_unit_summary(analyzer, unit_id=0)
```

## LFP 分析

### 載入 LFP 資料
```python
lfp = si.read_spikeglx('/path/to/data', stream_id='imec0.lf')
print(f"LFP: {lfp.get_sampling_frequency()} Hz")
```

### 基本 LFP 處理
```python
# 如有需要則降取樣
lfp_ds = si.resample(lfp, resample_rate=1000)

# 共同平均參考
lfp_car = si.common_reference(lfp_ds, reference='global', operator='median')
```

### 擷取 LFP 軌跡
```python
import numpy as np

# 取得軌跡（通道 x 樣本）
traces = lfp.get_traces(start_frame=0, end_frame=30000)

# 特定通道
traces = lfp.get_traces(channel_ids=[0, 1, 2])
```

### 頻譜分析
```python
from scipy import signal
import matplotlib.pyplot as plt

# 取得單一通道
trace = lfp.get_traces(channel_ids=[0]).flatten()
fs = lfp.get_sampling_frequency()

# 功率頻譜
freqs, psd = signal.welch(trace, fs, nperseg=4096)
plt.semilogy(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0, 100)
```

### 頻譜圖
```python
f, t, Sxx = signal.spectrogram(trace, fs, nperseg=2048, noverlap=1024)
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.ylim(0, 100)
plt.colorbar(label='Power (dB)')
```

## 匯出格式

### 匯出至 Phy
```python
si.export_to_phy(
    analyzer,
    output_folder='phy_export',
    compute_pc_features=True,
    compute_amplitudes=True,
    copy_binary=True
)
# 然後：phy template-gui phy_export/params.py
```

### 匯出至 NWB
```python
from spikeinterface.exporters import export_to_nwb

export_to_nwb(
    recording,
    sorting,
    'output.nwb',
    metadata=dict(
        session_description='Neuropixels recording',
        experimenter='Name',
        lab='Lab name',
        institution='Institution'
    )
)
```

### 匯出報告
```python
si.export_report(
    analyzer,
    output_folder='report',
    remove_if_exists=True,
    format='html'
)
```

## 完整分析流程

```python
import spikeinterface.full as si

def analyze_sorting(recording, sorting, output_dir):
    """完整後處理流程。"""

    # 建立分析器
    analyzer = si.create_sorting_analyzer(
        sorting, recording,
        sparse=True,
        folder=f'{output_dir}/analyzer'
    )

    # 計算所有擴充功能
    print("Computing extensions...")
    analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    analyzer.compute(['principal_components', 'spike_amplitudes'])
    analyzer.compute(['correlograms', 'unit_locations', 'template_similarity'])
    analyzer.compute('quality_metrics')

    # 取得品質指標
    qm = analyzer.get_extension('quality_metrics').get_data()

    # 篩選良好單元
    good_units = qm.query(
        "(snr > 5) & (isi_violations_ratio < 0.01) & (presence_ratio > 0.9)"
    ).index.tolist()

    print(f"Quality filtering: {len(good_units)}/{len(qm)} units passed")

    # 匯出
    si.export_to_phy(analyzer, f'{output_dir}/phy')
    si.export_report(analyzer, f'{output_dir}/report')

    # 儲存指標
    qm.to_csv(f'{output_dir}/quality_metrics.csv')

    return analyzer, qm, good_units

# 使用方式
analyzer, qm, good_units = analyze_sorting(recording, sorting, 'output/')
```
