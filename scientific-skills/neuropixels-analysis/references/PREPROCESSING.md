# Neuropixels 預處理參考

Neuropixels 神經記錄的綜合預處理技術。

## 標準預處理流程

```python
import spikeinterface.full as si

# 載入原始資料
recording = si.read_spikeglx('/path/to/data', stream_id='imec0.ap')

# 1. 相位偏移修正（用於 Neuropixels 1.0）
rec = si.phase_shift(recording)

# 2. 帶通濾波用於尖峰偵測
rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)

# 3. 共同中位數參考（移除相關雜訊）
rec = si.common_reference(rec, reference='global', operator='median')

# 4. 移除壞通道（可選）
rec = si.remove_bad_channels(rec, bad_channel_ids=bad_channels)
```

## 濾波選項

### 帶通濾波
```python
# 標準 AP 頻帶
rec = si.bandpass_filter(recording, freq_min=300, freq_max=6000)

# 較寬頻帶（保留更多波形形狀）
rec = si.bandpass_filter(recording, freq_min=150, freq_max=7500)

# 濾波參數
rec = si.bandpass_filter(
    recording,
    freq_min=300,
    freq_max=6000,
    filter_order=5,
    ftype='butter',  # 'butter', 'bessel' 或 'cheby1'
    margin_ms=5.0    # 防止邊緣偽影
)
```

### 僅高通濾波
```python
rec = si.highpass_filter(recording, freq_min=300)
```

### 陷波濾波（移除市電雜訊）
```python
# 移除 60Hz 及其諧波
rec = si.notch_filter(recording, freq=60, q=30)
rec = si.notch_filter(rec, freq=120, q=30)
rec = si.notch_filter(rec, freq=180, q=30)
```

## 參考方案

### 共同中位數參考（建議）
```python
# 全域中位數參考
rec = si.common_reference(recording, reference='global', operator='median')

# 每軸參考（多軸探針）
rec = si.common_reference(recording, reference='global', operator='median',
                          groups=recording.get_channel_groups())
```

### 共同平均參考
```python
rec = si.common_reference(recording, reference='global', operator='average')
```

### 局部參考
```python
# 按通道局部群組參考
rec = si.common_reference(recording, reference='local', local_radius=(30, 100))
```

## 壞通道偵測與移除

### 自動偵測
```python
# 偵測壞通道
bad_channel_ids, channel_labels = si.detect_bad_channels(
    recording,
    method='coherence+psd',
    dead_channel_threshold=-0.5,
    noisy_channel_threshold=1.0,
    outside_channel_threshold=-0.3,
    n_neighbors=11
)

print(f"Bad channels: {bad_channel_ids}")
print(f"Labels: {dict(zip(bad_channel_ids, channel_labels))}")
```

### 移除壞通道
```python
rec_clean = si.remove_bad_channels(recording, bad_channel_ids=bad_channel_ids)
```

### 插值壞通道
```python
rec_interp = si.interpolate_bad_channels(recording, bad_channel_ids=bad_channel_ids)
```

## 運動修正

### 估計運動
```python
# 估計運動（漂移）
motion, temporal_bins, spatial_bins = si.estimate_motion(
    recording,
    method='decentralized',
    rigid=False,              # 非剛性運動估計
    win_step_um=50,           # 空間視窗步進
    win_sigma_um=150,         # 空間視窗 sigma
    progress_bar=True
)
```

### 套用運動修正
```python
rec_corrected = si.correct_motion(
    recording,
    motion,
    temporal_bins,
    spatial_bins,
    interpolate_motion_border=True
)
```

### 運動視覺化
```python
si.plot_motion(motion, temporal_bins, spatial_bins)
```

## 探針特定處理

### Neuropixels 1.0
```python
# 相位偏移修正（每通道不同 ADC）
rec = si.phase_shift(recording)

# 然後標準流程
rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
rec = si.common_reference(rec, reference='global', operator='median')
```

### Neuropixels 2.0
```python
# 不需要相位偏移（單一 ADC）
rec = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
rec = si.common_reference(rec, reference='global', operator='median')
```

### 多軸（Neuropixels 2.0 4 軸）
```python
# 每軸參考
groups = recording.get_channel_groups()  # 回傳軸分配
rec = si.common_reference(recording, reference='global', operator='median', groups=groups)
```

## 白化

```python
# 白化資料（去相關通道）
rec_whitened = si.whiten(recording, mode='local', local_radius_um=100)

# 全域白化
rec_whitened = si.whiten(recording, mode='global')
```

## 偽影移除

### 移除刺激偽影
```python
# 定義偽影時間（以樣本為單位）
triggers = [10000, 20000, 30000]  # 樣本索引

rec = si.remove_artifacts(
    recording,
    triggers,
    ms_before=0.5,
    ms_after=3.0,
    mode='cubic'  # 'zeros', 'linear', 'cubic'
)
```

### 遮蔽飽和期間
```python
rec = si.blank_staturation(recording, threshold=0.95, fill_value=0)
```

## 儲存預處理資料

### 二進位格式（建議）
```python
rec_preprocessed.save(folder='preprocessed/', format='binary', n_jobs=4)
```

### Zarr 格式（壓縮）
```python
rec_preprocessed.save(folder='preprocessed.zarr', format='zarr')
```

### 儲存為 Recording Extractor
```python
# 儲存以供稍後使用
rec_preprocessed.save(folder='preprocessed/', format='binary')

# 稍後載入
rec_loaded = si.load_extractor('preprocessed/')
```

## 完整流程範例

```python
import spikeinterface.full as si

def preprocess_neuropixels(data_path, output_path):
    """標準 Neuropixels 預處理流程。"""

    # 載入資料
    recording = si.read_spikeglx(data_path, stream_id='imec0.ap')
    print(f"Loaded: {recording.get_num_channels()} channels, "
          f"{recording.get_total_duration():.1f}s")

    # 相位偏移（僅 NP 1.0）
    rec = si.phase_shift(recording)

    # 濾波
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)

    # 偵測並移除壞通道
    bad_ids, _ = si.detect_bad_channels(rec)
    if len(bad_ids) > 0:
        print(f"Removing {len(bad_ids)} bad channels: {bad_ids}")
        rec = si.interpolate_bad_channels(rec, bad_ids)

    # 共同參考
    rec = si.common_reference(rec, reference='global', operator='median')

    # 儲存
    rec.save(folder=output_path, format='binary', n_jobs=4)
    print(f"Saved to: {output_path}")

    return rec

# 使用方式
rec_preprocessed = preprocess_neuropixels(
    '/path/to/spikeglx/data',
    '/path/to/preprocessed'
)
```

## 效能提示

```python
# 使用平行處理
rec.save(folder='output/', n_jobs=-1)  # 使用所有核心

# 使用 job kwargs 進行記憶體管理
job_kwargs = dict(n_jobs=8, chunk_duration='1s', progress_bar=True)
rec.save(folder='output/', **job_kwargs)

# 設定全域 job kwargs
si.set_global_job_kwargs(n_jobs=8, chunk_duration='1s')
```
