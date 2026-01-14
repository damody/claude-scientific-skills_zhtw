# 繪圖指南

從 Neuropixels 資料建立發表品質視覺化的綜合指南。

## 設定

```python
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
import spikeinterface.widgets as sw
import neuropixels_analysis as npa

# 高品質設定
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
```

## 漂移與運動圖

### 基本漂移圖

```python
# 使用 npa
npa.plot_drift(recording, output='drift_map.png')

# 使用 SpikeInterface widgets
from spikeinterface.preprocessing import detect_peaks, localize_peaks

peaks = detect_peaks(recording, method='locally_exclusive')
peak_locations = localize_peaks(recording, peaks, method='center_of_mass')

sw.plot_drift_raster_map(
    peaks=peaks,
    peak_locations=peak_locations,
    recording=recording,
    clim=(-50, 50),
)
plt.savefig('drift_raster.png', bbox_inches='tight')
```

### 運動估計視覺化

```python
motion_info = npa.estimate_motion(recording)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 運動隨時間變化
ax = axes[0]
for i in range(motion_info['motion'].shape[1]):
    ax.plot(motion_info['temporal_bins'], motion_info['motion'][:, i], alpha=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Motion (um)')
ax.set_title('Estimated Motion')

# 運動直方圖
ax = axes[1]
ax.hist(motion_info['motion'].flatten(), bins=50, edgecolor='black')
ax.set_xlabel('Motion (um)')
ax.set_ylabel('Count')
ax.set_title('Motion Distribution')

plt.tight_layout()
plt.savefig('motion_analysis.png', dpi=300)
```

## 波形圖

### 單一單元波形

```python
unit_id = 0

# 基本波形
sw.plot_unit_waveforms(analyzer, unit_ids=[unit_id])
plt.savefig(f'unit_{unit_id}_waveforms.png')

# 含密度圖
sw.plot_unit_waveform_density_map(analyzer, unit_ids=[unit_id])
plt.savefig(f'unit_{unit_id}_density.png')
```

### 範本比較

```python
# 比較多個單元
unit_ids = [0, 1, 2, 3]
sw.plot_unit_templates(analyzer, unit_ids=unit_ids)
plt.savefig('template_comparison.png')
```

### 探針上的波形

```python
# 在探針上顯示空間波形
sw.plot_unit_waveforms_on_probe(
    analyzer,
    unit_ids=[unit_id],
    plot_channels=True,
)
plt.savefig(f'unit_{unit_id}_probe.png')
```

## 品質指標視覺化

### 指標概覽

```python
npa.plot_quality_metrics(analyzer, metrics, output='quality_overview.png')
```

### 指標分布

```python
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

metric_names = ['snr', 'isi_violations_ratio', 'presence_ratio',
                'amplitude_cutoff', 'firing_rate', 'amplitude_cv']

for ax, metric in zip(axes.flat, metric_names):
    if metric in metrics.columns:
        values = metrics[metric].dropna()
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(values.median(), color='red', linestyle='--', label='median')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.legend()

plt.tight_layout()
plt.savefig('metrics_distribution.png', dpi=300)
```

### 指標散佈矩陣

```python
import pandas as pd

key_metrics = ['snr', 'isi_violations_ratio', 'presence_ratio', 'firing_rate']
pd.plotting.scatter_matrix(
    metrics[key_metrics],
    figsize=(10, 10),
    alpha=0.5,
    diagonal='hist',
)
plt.savefig('metrics_scatter.png', dpi=300)
```

### 指標與標籤

```python
labels_series = pd.Series(labels)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, metric in zip(axes, ['snr', 'isi_violations_ratio', 'presence_ratio']):
    for label in ['good', 'mua', 'noise']:
        mask = labels_series == label
        if mask.any():
            ax.hist(metrics.loc[mask.index[mask], metric],
                   alpha=0.5, label=label, bins=20)
    ax.set_xlabel(metric)
    ax.legend()

plt.tight_layout()
plt.savefig('metrics_by_label.png', dpi=300)
```

## 相關圖

### 自相關圖

```python
sw.plot_autocorrelograms(
    analyzer,
    unit_ids=[unit_id],
    window_ms=50,
    bin_ms=1,
)
plt.savefig(f'unit_{unit_id}_acg.png')
```

### 互相關圖

```python
unit_pairs = [(0, 1), (0, 2), (1, 2)]
sw.plot_crosscorrelograms(
    analyzer,
    unit_pairs=unit_pairs,
    window_ms=50,
    bin_ms=1,
)
plt.savefig('crosscorrelograms.png')
```

### 相關圖矩陣

```python
sw.plot_autocorrelograms(
    analyzer,
    unit_ids=analyzer.sorting.unit_ids[:10],  # 前 10 個單元
)
plt.savefig('acg_matrix.png')
```

## 尖峰序列圖

### 光柵圖

```python
sw.plot_rasters(
    sorting,
    time_range=(0, 30),  # 前 30 秒
    unit_ids=unit_ids[:5],
)
plt.savefig('raster.png')
```

### 放電率隨時間變化

```python
unit_id = 0
spike_train = sorting.get_unit_spike_train(unit_id)
fs = recording.get_sampling_frequency()
times = spike_train / fs

# 計算放電率直方圖
bin_width = 1.0  # 秒
bins = np.arange(0, recording.get_total_duration(), bin_width)
hist, _ = np.histogram(times, bins=bins)
firing_rate = hist / bin_width

plt.figure(figsize=(12, 3))
plt.bar(bins[:-1], firing_rate, width=bin_width, edgecolor='none')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
plt.title(f'Unit {unit_id} firing rate')
plt.savefig(f'unit_{unit_id}_firing_rate.png', dpi=300)
```

## 探針與位置圖

### 探針配置

```python
sw.plot_probe_map(recording, with_channel_ids=True)
plt.savefig('probe_layout.png')
```

### 探針上的單元位置

```python
sw.plot_unit_locations(analyzer, with_channel_ids=True)
plt.savefig('unit_locations.png')
```

### 尖峰位置

```python
sw.plot_spike_locations(analyzer, unit_ids=[unit_id])
plt.savefig(f'unit_{unit_id}_spike_locations.png')
```

## 振幅圖

### 振幅隨時間變化

```python
sw.plot_amplitudes(
    analyzer,
    unit_ids=[unit_id],
    plot_histograms=True,
)
plt.savefig(f'unit_{unit_id}_amplitudes.png')
```

### 振幅分布

```python
amplitudes = analyzer.get_extension('spike_amplitudes').get_data()
spike_vector = sorting.to_spike_vector()
unit_idx = list(sorting.unit_ids).index(unit_id)
unit_mask = spike_vector['unit_index'] == unit_idx
unit_amps = amplitudes[unit_mask]

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(unit_amps, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.median(unit_amps), color='red', linestyle='--', label='median')
ax.set_xlabel('Amplitude (uV)')
ax.set_ylabel('Count')
ax.set_title(f'Unit {unit_id} Amplitude Distribution')
ax.legend()
plt.savefig(f'unit_{unit_id}_amp_dist.png', dpi=300)
```

## ISI 圖

### ISI 直方圖

```python
sw.plot_isi_distribution(
    analyzer,
    unit_ids=[unit_id],
    window_ms=100,
    bin_ms=1,
)
plt.savefig(f'unit_{unit_id}_isi.png')
```

### 含反應期標記的 ISI

```python
spike_train = sorting.get_unit_spike_train(unit_id)
fs = recording.get_sampling_frequency()
isis = np.diff(spike_train) / fs * 1000  # ms

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(isis[isis < 100], bins=100, edgecolor='black', alpha=0.7)
ax.axvline(1.5, color='red', linestyle='--', label='1.5ms refractory')
ax.axvline(3.0, color='orange', linestyle='--', label='3ms threshold')
ax.set_xlabel('ISI (ms)')
ax.set_ylabel('Count')
ax.set_title(f'Unit {unit_id} ISI Distribution')
ax.legend()
plt.savefig(f'unit_{unit_id}_isi_detailed.png', dpi=300)
```

## 摘要圖

### 單元摘要面板

```python
npa.plot_unit_summary(analyzer, unit_id, output=f'unit_{unit_id}_summary.png')
```

### 手動多面板摘要

```python
fig = plt.figure(figsize=(16, 12))

# 波形
ax1 = fig.add_subplot(2, 3, 1)
wfs = analyzer.get_extension('waveforms').get_waveforms(unit_id)
for i in range(min(50, wfs.shape[0])):
    ax1.plot(wfs[i, :, 0], 'k', alpha=0.1, linewidth=0.5)
template = wfs.mean(axis=0)[:, 0]
ax1.plot(template, 'b', linewidth=2)
ax1.set_title('Waveforms')

# 範本
ax2 = fig.add_subplot(2, 3, 2)
templates_ext = analyzer.get_extension('templates')
template = templates_ext.get_unit_template(unit_id, operator='average')
template_std = templates_ext.get_unit_template(unit_id, operator='std')
x = range(template.shape[0])
ax2.plot(x, template[:, 0], 'b', linewidth=2)
ax2.fill_between(x, template[:, 0] - template_std[:, 0],
                 template[:, 0] + template_std[:, 0], alpha=0.3)
ax2.set_title('Template')

# 自相關圖
ax3 = fig.add_subplot(2, 3, 3)
correlograms = analyzer.get_extension('correlograms')
ccg, bins = correlograms.get_data()
unit_idx = list(sorting.unit_ids).index(unit_id)
ax3.bar(bins[:-1], ccg[unit_idx, unit_idx, :], width=bins[1]-bins[0], color='gray')
ax3.axvline(0, color='r', linestyle='--', alpha=0.5)
ax3.set_title('Autocorrelogram')

# 振幅
ax4 = fig.add_subplot(2, 3, 4)
amps_ext = analyzer.get_extension('spike_amplitudes')
amps = amps_ext.get_data()
spike_vector = sorting.to_spike_vector()
unit_mask = spike_vector['unit_index'] == unit_idx
unit_times = spike_vector['sample_index'][unit_mask] / fs
unit_amps = amps[unit_mask]
ax4.scatter(unit_times, unit_amps, s=1, alpha=0.3)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude')
ax4.set_title('Amplitudes')

# ISI
ax5 = fig.add_subplot(2, 3, 5)
isis = np.diff(sorting.get_unit_spike_train(unit_id)) / fs * 1000
ax5.hist(isis[isis < 100], bins=50, color='gray', edgecolor='black')
ax5.axvline(1.5, color='r', linestyle='--')
ax5.set_xlabel('ISI (ms)')
ax5.set_title('ISI Distribution')

# 指標
ax6 = fig.add_subplot(2, 3, 6)
unit_metrics = metrics.loc[unit_id]
text_lines = [f"{k}: {v:.4f}" for k, v in unit_metrics.items() if not np.isnan(v)]
ax6.text(0.1, 0.9, '\n'.join(text_lines[:8]), transform=ax6.transAxes,
         verticalalignment='top', fontsize=10, family='monospace')
ax6.axis('off')
ax6.set_title('Metrics')

plt.tight_layout()
plt.savefig(f'unit_{unit_id}_full_summary.png', dpi=300)
```

## 發表品質設定

### 圖形大小

```python
# 單欄（3.5 英寸）
fig, ax = plt.subplots(figsize=(3.5, 3))

# 雙欄（7 英寸）
fig, ax = plt.subplots(figsize=(7, 4))

# 全頁
fig, ax = plt.subplots(figsize=(7, 9))
```

### 字體設定

```python
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'font.family': 'Arial',
})
```

### 匯出設定

```python
# 用於發表
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure.svg', format='svg', bbox_inches='tight')

# 高解析度 PNG
plt.savefig('figure.png', dpi=600, bbox_inches='tight', facecolor='white')
```

### 色彩調色盤

```python
# 色盲友好
colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442']

# 用於 good/mua/noise
label_colors = {'good': '#2ecc71', 'mua': '#f39c12', 'noise': '#e74c3c'}
```
