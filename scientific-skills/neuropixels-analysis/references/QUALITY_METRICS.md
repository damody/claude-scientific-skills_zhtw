# 品質指標參考

使用 SpikeInterface 指標和 Allen/IBL 標準進行單元品質評估的綜合指南。

## 概述

品質指標評估已分選單元的三個面向：

| 類別 | 問題 | 關鍵指標 |
|----------|----------|-------------|
| **污染**（第一型） | 尖峰是否來自多個神經元？ | ISI 違規、SNR |
| **完整性**（第二型） | 是否遺漏尖峰？ | 振幅截止值、存在比率 |
| **穩定性** | 單元是否隨時間穩定？ | 漂移指標、振幅 CV |

## 計算品質指標

```python
import spikeinterface.full as si

# 建立已計算波形的分析器
analyzer = si.create_sorting_analyzer(sorting, recording, sparse=True)
analyzer.compute('random_spikes', max_spikes_per_unit=500)
analyzer.compute('waveforms', ms_before=1.5, ms_after=2.0)
analyzer.compute('templates')
analyzer.compute('noise_levels')
analyzer.compute('spike_amplitudes')
analyzer.compute('principal_components', n_components=5)

# 計算所有品質指標
analyzer.compute('quality_metrics')

# 或計算特定指標
analyzer.compute('quality_metrics', metric_names=[
    'firing_rate', 'snr', 'isi_violations_ratio',
    'presence_ratio', 'amplitude_cutoff'
])

# 取得結果
qm = analyzer.get_extension('quality_metrics').get_data()
print(qm.columns.tolist())  # 可用指標
```

## 指標定義與閾值

### 污染指標

#### ISI 違規比率
違反反應期的尖峰比例。所有神經元都有約 1.5ms 的反應期。

```python
# 使用自訂反應期計算
analyzer.compute('quality_metrics',
                 metric_names=['isi_violations_ratio'],
                 isi_threshold_ms=1.5,
                 min_isi_ms=0.0)
```

| 數值 | 解釋 |
|-------|---------------|
| < 0.01 | 優秀（分離良好的單一單元） |
| 0.01 - 0.1 | 良好（輕微污染） |
| 0.1 - 0.5 | 中等（可能為多單元活動） |
| > 0.5 | 差（可能為多單元） |

**參考文獻：** Hill et al. (2011) J Neurosci 31:8699-8705

#### 訊雜比（SNR）
波形峰值振幅與背景雜訊的比率。

```python
analyzer.compute('quality_metrics', metric_names=['snr'])
```

| 數值 | 解釋 |
|-------|---------------|
| > 10 | 優秀 |
| 5 - 10 | 良好 |
| 2 - 5 | 可接受 |
| < 2 | 差（可能為雜訊） |

#### 分離距離
PCA 空間中到最近群集的馬氏距離。

```python
analyzer.compute('quality_metrics',
                 metric_names=['isolation_distance'],
                 n_neighbors=4)
```

| 數值 | 解釋 |
|-------|---------------|
| > 50 | 分離良好 |
| 20 - 50 | 中等分離 |
| < 20 | 分離不佳 |

#### L 比率
基於馬氏距離的污染測量。

| 數值 | 解釋 |
|-------|---------------|
| < 0.05 | 分離良好 |
| 0.05 - 0.1 | 可接受 |
| > 0.1 | 有污染 |

#### D-prime
單元與最近鄰之間的可辨別性。

| 數值 | 解釋 |
|-------|---------------|
| > 8 | 優秀分離 |
| 5 - 8 | 良好分離 |
| < 5 | 分離不佳 |

### 完整性指標

#### 振幅截止值
估計低於偵測閾值的尖峰比例。

```python
analyzer.compute('quality_metrics',
                 metric_names=['amplitude_cutoff'],
                 peak_sign='neg')  # 'neg', 'pos' 或 'both'
```

| 數值 | 解釋 |
|-------|---------------|
| < 0.01 | 優秀（幾乎完整） |
| 0.01 - 0.1 | 良好 |
| 0.1 - 0.2 | 中等（部分遺漏尖峰） |
| > 0.2 | 差（大量遺漏尖峰） |

**精確時序分析：** 使用 < 0.01

#### 存在比率
有偵測到尖峰的記錄時間比例。

```python
analyzer.compute('quality_metrics',
                 metric_names=['presence_ratio'],
                 bin_duration_s=60)  # 1 分鐘 bin
```

| 數值 | 解釋 |
|-------|---------------|
| > 0.99 | 優秀 |
| 0.9 - 0.99 | 良好 |
| 0.8 - 0.9 | 可接受 |
| < 0.8 | 單元可能已漂移出去 |

### 穩定性指標

#### 漂移指標
測量單元隨時間的移動。

```python
analyzer.compute('quality_metrics',
                 metric_names=['drift_ptp', 'drift_std', 'drift_mad'])
```

| 指標 | 描述 | 良好值 |
|--------|-------------|------------|
| `drift_ptp` | 峰對峰漂移（μm） | < 40 |
| `drift_std` | 漂移標準差 | < 10 |
| `drift_mad` | 中位數絕對離差 | < 10 |

#### 振幅 CV
尖峰振幅的變異係數。

| 數值 | 解釋 |
|-------|---------------|
| < 0.25 | 非常穩定 |
| 0.25 - 0.5 | 可接受 |
| > 0.5 | 不穩定（漂移或污染） |

### 群集品質指標

#### 輪廓係數
群集內聚性與分離度（-1 到 1）。

| 數值 | 解釋 |
|-------|---------------|
| > 0.5 | 定義良好的群集 |
| 0.25 - 0.5 | 中等 |
| < 0.25 | 群集重疊 |

#### 最近鄰指標

```python
analyzer.compute('quality_metrics',
                 metric_names=['nn_hit_rate', 'nn_miss_rate'],
                 n_neighbors=4)
```

| 指標 | 描述 | 良好值 |
|--------|-------------|------------|
| `nn_hit_rate` | 具有相同單元鄰居的尖峰比例 | > 0.9 |
| `nn_miss_rate` | 具有其他單元鄰居的尖峰比例 | < 0.1 |

## 標準篩選標準

### Allen Institute 預設值

```python
# Allen Visual Coding / Behavior 預設值
allen_query = """
    presence_ratio > 0.95 and
    isi_violations_ratio < 0.5 and
    amplitude_cutoff < 0.1
"""
good_units = qm.query(allen_query).index.tolist()
```

### IBL 標準

```python
# IBL 可重現電生理標準
ibl_query = """
    presence_ratio > 0.9 and
    isi_violations_ratio < 0.1 and
    amplitude_cutoff < 0.1 and
    firing_rate > 0.1
"""
good_units = qm.query(ibl_query).index.tolist()
```

### 嚴格單一單元標準

```python
# 用於精確時序/尖峰時序分析
strict_query = """
    snr > 5 and
    presence_ratio > 0.99 and
    isi_violations_ratio < 0.01 and
    amplitude_cutoff < 0.01 and
    isolation_distance > 20 and
    drift_ptp < 40
"""
single_units = qm.query(strict_query).index.tolist()
```

### 多單元活動（MUA）

```python
# 包含多單元活動
mua_query = """
    snr > 2 and
    presence_ratio > 0.5 and
    isi_violations_ratio < 1.0
"""
all_units = qm.query(mua_query).index.tolist()
```

## 視覺化

### 品質指標摘要

```python
# 繪製所有指標
si.plot_quality_metrics(analyzer)
```

### 個別指標分布

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metrics = ['snr', 'isi_violations_ratio', 'presence_ratio',
           'amplitude_cutoff', 'firing_rate', 'drift_ptp']

for ax, metric in zip(axes.flat, metrics):
    ax.hist(qm[metric].dropna(), bins=50, edgecolor='black')
    ax.set_xlabel(metric)
    ax.set_ylabel('Count')
    # 加入閾值線
    if metric == 'snr':
        ax.axvline(5, color='r', linestyle='--', label='threshold')
    elif metric == 'isi_violations_ratio':
        ax.axvline(0.01, color='r', linestyle='--')
    elif metric == 'presence_ratio':
        ax.axvline(0.9, color='r', linestyle='--')

plt.tight_layout()
```

### 單元品質摘要

```python
# 綜合單元摘要圖
si.plot_unit_summary(analyzer, unit_id=0)
```

### 品質與放電率關係

```python
fig, ax = plt.subplots()
scatter = ax.scatter(qm['firing_rate'], qm['snr'],
                     c=qm['isi_violations_ratio'],
                     cmap='RdYlGn_r', alpha=0.6)
ax.set_xlabel('Firing Rate (Hz)')
ax.set_ylabel('SNR')
plt.colorbar(scatter, label='ISI Violations')
ax.set_xscale('log')
```

## 一次計算所有指標

```python
# 完整品質指標計算
all_metric_names = [
    # 放電特性
    'firing_rate', 'presence_ratio',
    # 波形
    'snr', 'amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range',
    # ISI
    'isi_violations_ratio', 'isi_violations_count',
    # 漂移
    'drift_ptp', 'drift_std', 'drift_mad',
    # 分離（需要 PCA）
    'isolation_distance', 'l_ratio', 'd_prime',
    # 最近鄰（需要 PCA）
    'nn_hit_rate', 'nn_miss_rate',
    # 群集品質
    'silhouette_score',
    # 同步性
    'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
]

# 先計算 PCA（部分指標需要）
analyzer.compute('principal_components', n_components=5)

# 計算指標
analyzer.compute('quality_metrics', metric_names=all_metric_names)
qm = analyzer.get_extension('quality_metrics').get_data()

# 儲存至 CSV
qm.to_csv('quality_metrics.csv')
```

## 自訂指標

```python
from spikeinterface.qualitymetrics import compute_firing_rates, compute_snrs

# 計算個別指標
firing_rates = compute_firing_rates(sorting)
snrs = compute_snrs(analyzer)

# 將自訂指標加入 DataFrame
qm['custom_score'] = qm['snr'] * qm['presence_ratio'] / (qm['isi_violations_ratio'] + 0.001)
```

## 參考資料

- [SpikeInterface Quality Metrics](https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html)
- [Allen Institute ecephys_quality_metrics](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html)
- Hill et al. (2011) "Quality metrics to accompany spike sorting of extracellular signals"
- Siegle et al. (2021) "Survey of spiking in the mouse visual system reveals functional hierarchy"
