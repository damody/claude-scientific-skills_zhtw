# 運動/漂移修正參考

急性探針插入期間的機械漂移是 Neuropixels 記錄的主要挑戰。本指南涵蓋運動偽影的偵測、估計和修正。

## 為何運動修正很重要

- Neuropixels 探針在記錄期間可能漂移 10-100+ μm
- 未修正的漂移導致：
  - 單元在記錄中途出現/消失
  - 波形振幅變化
  - 錯誤的尖峰-單元分配
  - 單元產量降低

## 偵測：分選前檢查

**務必在執行尖峰分選前視覺化漂移！**

```python
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

# 先預處理（不要白化 - 會影響峰值定位）
rec = si.highpass_filter(recording, freq_min=400.)
rec = si.common_reference(rec, operator='median', reference='global')

# 偵測峰值
noise_levels = si.get_noise_levels(rec, return_in_uV=False)
peaks = detect_peaks(
    rec,
    method='locally_exclusive',
    noise_levels=noise_levels,
    detect_threshold=5,
    radius_um=50.,
    n_jobs=8,
    chunk_duration='1s',
    progress_bar=True
)

# 定位峰值
peak_locations = localize_peaks(
    rec, peaks,
    method='center_of_mass',
    n_jobs=8,
    chunk_duration='1s'
)

# 視覺化漂移
si.plot_drift_raster_map(
    peaks=peaks,
    peak_locations=peak_locations,
    recording=rec,
    clim=(-200, 0)  # 調整顏色範圍
)
```

### 解讀漂移圖

| 模式 | 解釋 | 動作 |
|---------|---------------|--------|
| 水平條帶，穩定 | 無顯著漂移 | 跳過修正 |
| 對角條帶（緩慢） | 逐漸沉降漂移 | 使用運動修正 |
| 快速跳躍 | 腦脈動或移動 | 使用非剛性修正 |
| 混亂模式 | 嚴重不穩定 | 考慮捨棄該段 |

## 運動修正方法

### 快速修正（建議起點）

```python
# 使用預設的單行指令
rec_corrected = si.correct_motion(
    recording=rec,
    preset='nonrigid_fast_and_accurate'
)
```

### 可用預設

| 預設 | 速度 | 準確度 | 適用情況 |
|--------|-------|----------|----------|
| `rigid_fast` | 快 | 低 | 快速檢查、小漂移 |
| `kilosort_like` | 中 | 良好 | 與 Kilosort 相容的結果 |
| `nonrigid_accurate` | 慢 | 高 | 發表品質 |
| `nonrigid_fast_and_accurate` | 中 | 高 | **建議預設** |
| `dredge` | 慢 | 最高 | 最佳結果、複雜漂移 |
| `dredge_fast` | 中 | 高 | DREDge 低計算量版 |

### 完整控制流程

```python
from spikeinterface.sortingcomponents.motion import (
    estimate_motion,
    interpolate_motion
)

# 步驟 1：估計運動
motion, temporal_bins, spatial_bins = estimate_motion(
    rec,
    peaks,
    peak_locations,
    method='decentralized',
    direction='y',
    rigid=False,          # 對 Neuropixels 使用非剛性
    win_step_um=50,       # 空間視窗步進
    win_sigma_um=150,     # 空間平滑
    bin_s=2.0,            # 時間 bin 大小
    progress_bar=True
)

# 步驟 2：視覺化運動估計
si.plot_motion(
    motion,
    temporal_bins,
    spatial_bins,
    recording=rec
)

# 步驟 3：透過插值套用修正
rec_corrected = interpolate_motion(
    recording=rec,
    motion=motion,
    temporal_bins=temporal_bins,
    spatial_bins=spatial_bins,
    border_mode='force_extrapolate'
)
```

### 儲存運動估計

```python
# 儲存以供稍後使用
import numpy as np
np.savez('motion_estimate.npz',
         motion=motion,
         temporal_bins=temporal_bins,
         spatial_bins=spatial_bins)

# 稍後載入
data = np.load('motion_estimate.npz')
motion = data['motion']
temporal_bins = data['temporal_bins']
spatial_bins = data['spatial_bins']
```

## DREDge：最先進的方法

DREDge（Decentralized Registration of Electrophysiology Data，電生理資料分散式配準）是目前表現最佳的運動修正方法。

### 使用 DREDge 預設

```python
# AP 頻帶運動估計
rec_corrected = si.correct_motion(rec, preset='dredge')

# 或明確計算
motion, motion_info = si.compute_motion(
    rec,
    preset='dredge',
    output_motion_info=True,
    folder='motion_output/',
    **job_kwargs
)
```

### 基於 LFP 的運動估計

對於非常快速的漂移或 AP 頻帶估計失敗時：

```python
# 載入 LFP 串流
lfp = si.read_spikeglx('/path/to/data', stream_name='imec0.lf')

# 從 LFP 估計運動（更快，處理快速漂移）
motion_lfp, motion_info = si.compute_motion(
    lfp,
    preset='dredge_lfp',
    output_motion_info=True
)

# 套用至 AP 記錄
rec_corrected = interpolate_motion(
    recording=rec,  # AP 記錄
    motion=motion_lfp,
    temporal_bins=motion_info['temporal_bins'],
    spatial_bins=motion_info['spatial_bins']
)
```

## 與尖峰分選整合

### 選項 1：預先修正（建議）

```python
# 分選前修正
rec_corrected = si.correct_motion(rec, preset='nonrigid_fast_and_accurate')

# 儲存修正後的記錄
rec_corrected = rec_corrected.save(folder='preprocessed_motion_corrected/',
                                    format='binary', n_jobs=8)

# 在修正後的資料上執行尖峰分選
sorting = si.run_sorter('kilosort4', rec_corrected, output_folder='ks4/')
```

### 選項 2：讓 Kilosort 處理

Kilosort 2.5+ 有內建漂移修正：

```python
sorting = si.run_sorter(
    'kilosort4',
    rec,  # 未經運動修正
    output_folder='ks4/',
    nblocks=5,  # 用於漂移修正的非剛性區塊
    do_correction=True  # 啟用 Kilosort 的漂移修正
)
```

### 選項 3：事後修正

```python
# 先分選
sorting = si.run_sorter('kilosort4', rec, output_folder='ks4/')

# 然後從已分選的尖峰估計運動
#（更準確，因為使用實際尖峰時間）
from spikeinterface.sortingcomponents.motion import estimate_motion_from_sorting

motion = estimate_motion_from_sorting(sorting, rec)
```

## 參數深入探討

### 峰值偵測

```python
peaks = detect_peaks(
    rec,
    method='locally_exclusive',  # 最適合高密度探針
    noise_levels=noise_levels,
    detect_threshold=5,          # 較低 = 更多峰值（較嘈雜的估計）
    radius_um=50.,               # 排除半徑
    exclude_sweep_ms=0.1,        # 時間排除
)
```

### 運動估計

```python
motion = estimate_motion(
    rec, peaks, peak_locations,
    method='decentralized',      # 'decentralized' 或 'iterative_template'
    direction='y',               # 沿探針軸
    rigid=False,                 # False 為非剛性
    bin_s=2.0,                   # 時間解析度（秒）
    win_step_um=50,              # 空間視窗步進
    win_sigma_um=150,            # 空間平滑 sigma
    margin_um=0,                 # 探針邊緣邊距
    win_scale_um=150,            # 權重的視窗尺度
)
```

## 故障排除

### 過度修正（波浪狀模式）

```python
# 增加時間平滑
motion = estimate_motion(..., bin_s=5.0)  # 較大的 bin

# 或對小漂移使用剛性修正
motion = estimate_motion(..., rigid=True)
```

### 修正不足（漂移仍存在）

```python
# 減少空間視窗以獲得更精細的非剛性估計
motion = estimate_motion(..., win_step_um=25, win_sigma_um=75)

# 使用更多峰值
peaks = detect_peaks(..., detect_threshold=4)  # 較低閾值
```

### 邊緣偽影

```python
rec_corrected = interpolate_motion(
    rec, motion, temporal_bins, spatial_bins,
    border_mode='force_extrapolate',  # 或 'remove_channels'
    spatial_interpolation_method='kriging'
)
```

## 驗證

修正後，重新視覺化以確認：

```python
# 在修正後的記錄上重新偵測峰值
peaks_corrected = detect_peaks(rec_corrected, ...)
peak_locations_corrected = localize_peaks(rec_corrected, peaks_corrected, ...)

# 繪製前/後比較
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 修正前
si.plot_drift_raster_map(peaks, peak_locations, rec, ax=axes[0])
axes[0].set_title('Before Correction')

# 修正後
si.plot_drift_raster_map(peaks_corrected, peak_locations_corrected,
                         rec_corrected, ax=axes[1])
axes[1].set_title('After Correction')
```

## 參考資料

- [SpikeInterface Motion Correction Docs](https://spikeinterface.readthedocs.io/en/stable/modules/motion_correction.html)
- [Handle Drift Tutorial](https://spikeinterface.readthedocs.io/en/stable/how_to/handle_drift.html)
- [DREDge GitHub](https://github.com/evarol/DREDge)
- Windolf et al. (2023) "DREDge: robust motion correction for high-density extracellular recordings"
