# API 參考

按模組組織的 neuropixels_analysis 函數快速參考。

## 核心模組

### load_recording

```python
npa.load_recording(
    path: str,
    format: str = 'auto',  # 'spikeglx', 'openephys', 'nwb'
    stream_id: str = None,  # 例如 'imec0.ap'
) -> Recording
```

從各種格式載入 Neuropixels 記錄。

### run_pipeline

```python
npa.run_pipeline(
    recording: Recording,
    output_dir: str,
    sorter: str = 'kilosort4',
    preprocess: bool = True,
    correct_motion: bool = True,
    postprocess: bool = True,
    curate: bool = True,
    curation_method: str = 'allen',
) -> dict
```

執行完整分析流程。回傳包含所有結果的字典。

## 預處理模組

### preprocess

```python
npa.preprocess(
    recording: Recording,
    freq_min: float = 300,
    freq_max: float = 6000,
    phase_shift: bool = True,
    common_ref: bool = True,
    bad_channel_detection: bool = True,
) -> Recording
```

套用標準預處理鏈。

### detect_bad_channels

```python
npa.detect_bad_channels(
    recording: Recording,
    method: str = 'coherence+psd',
    **kwargs,
) -> list
```

偵測並回傳壞通道 ID 列表。

### apply_filters

```python
npa.apply_filters(
    recording: Recording,
    freq_min: float = 300,
    freq_max: float = 6000,
    filter_type: str = 'bandpass',
) -> Recording
```

套用頻率濾波器。

### common_reference

```python
npa.common_reference(
    recording: Recording,
    operator: str = 'median',
    reference: str = 'global',
) -> Recording
```

套用共同參考（CMR/CAR）。

## 運動模組

### check_drift

```python
npa.check_drift(
    recording: Recording,
    plot: bool = True,
    output: str = None,
) -> dict
```

檢查記錄的漂移。回傳漂移統計。

### estimate_motion

```python
npa.estimate_motion(
    recording: Recording,
    preset: str = 'kilosort_like',
    **kwargs,
) -> dict
```

估計運動但不套用修正。

### correct_motion

```python
npa.correct_motion(
    recording: Recording,
    preset: str = 'nonrigid_accurate',
    folder: str = None,
    **kwargs,
) -> Recording
```

套用運動修正。

**預設：**
- `'kilosort_like'`：快速、剛性修正
- `'nonrigid_accurate'`：較慢，適合嚴重漂移
- `'nonrigid_fast_and_accurate'`：平衡選項

## 分選模組

### run_sorting

```python
npa.run_sorting(
    recording: Recording,
    sorter: str = 'kilosort4',
    output_folder: str = None,
    sorter_params: dict = None,
    **kwargs,
) -> Sorting
```

執行尖峰分選器。

**支援的分選器：**
- `'kilosort4'`：基於 GPU，建議使用
- `'kilosort3'`：舊版，需要 MATLAB
- `'spykingcircus2'`：基於 CPU 的替代方案
- `'mountainsort5'`：快速，適合短記錄

### compare_sorters

```python
npa.compare_sorters(
    sortings: list,
    delta_time: float = 0.4,  # ms
    match_score: float = 0.5,
) -> Comparison
```

比較多個分選器的結果。

## 後處理模組

### create_analyzer

```python
npa.create_analyzer(
    sorting: Sorting,
    recording: Recording,
    output_folder: str = None,
    sparse: bool = True,
) -> SortingAnalyzer
```

建立用於後處理的 SortingAnalyzer。

### postprocess

```python
npa.postprocess(
    sorting: Sorting,
    recording: Recording,
    output_folder: str = None,
    compute_all: bool = True,
    n_jobs: int = -1,
) -> tuple[SortingAnalyzer, DataFrame]
```

完整後處理。回傳（analyzer, metrics）。

### compute_quality_metrics

```python
npa.compute_quality_metrics(
    analyzer: SortingAnalyzer,
    metric_names: list = None,  # None = 全部
    **kwargs,
) -> DataFrame
```

計算所有單元的品質指標。

**可用指標：**
- `snr`：訊雜比
- `isi_violations_ratio`：ISI 違規
- `presence_ratio`：記錄存在比率
- `amplitude_cutoff`：振幅分布截止值
- `firing_rate`：平均放電率
- `amplitude_cv`：振幅變異係數
- `sliding_rp_violation`：滑動視窗反應期違規
- `d_prime`：分離品質
- `nearest_neighbor`：最近鄰重疊

## 篩選模組

### curate

```python
npa.curate(
    metrics: DataFrame,
    method: str = 'allen',  # 'allen', 'ibl', 'strict', 'custom'
    **thresholds,
) -> dict
```

套用自動篩選。回傳 {unit_id: label}。

### auto_classify

```python
npa.auto_classify(
    metrics: DataFrame,
    snr_threshold: float = 5.0,
    isi_threshold: float = 0.01,
    presence_threshold: float = 0.9,
) -> dict
```

基於自訂閾值分類單元。

### filter_units

```python
npa.filter_units(
    sorting: Sorting,
    labels: dict,
    keep: list = ['good'],
) -> Sorting
```

篩選分選結果，僅保留指定標籤。

## AI 篩選模組

### generate_unit_report

```python
npa.generate_unit_report(
    analyzer: SortingAnalyzer,
    unit_id: int,
    output_dir: str = None,
    figsize: tuple = (16, 12),
) -> dict
```

產生用於 AI 分析的視覺報告。

回傳：
- `'image_path'`：已儲存圖形的路徑
- `'image_base64'`：Base64 編碼圖片
- `'metrics'`：品質指標字典
- `'unit_id'`：單元 ID

### analyze_unit_visually

```python
npa.analyze_unit_visually(
    analyzer: SortingAnalyzer,
    unit_id: int,
    api_client: Any = None,
    model: str = 'claude-3-5-sonnet-20241022',
    task: str = 'quality_assessment',
    custom_prompt: str = None,
) -> dict
```

使用視覺語言模型分析單元。

**任務：**
- `'quality_assessment'`：分類為 good/mua/noise
- `'merge_candidate'`：檢查單元是否應合併
- `'drift_assessment'`：評估運動/漂移

### batch_visual_curation

```python
npa.batch_visual_curation(
    analyzer: SortingAnalyzer,
    unit_ids: list = None,
    api_client: Any = None,
    model: str = 'claude-3-5-sonnet-20241022',
    output_dir: str = None,
    progress_callback: callable = None,
) -> dict
```

對多個單元執行視覺篩選。

### CurationSession

```python
session = npa.CurationSession.create(
    analyzer: SortingAnalyzer,
    output_dir: str,
    session_id: str = None,
    unit_ids: list = None,
    sort_by_confidence: bool = True,
)

# 導覽
session.current_unit() -> UnitCuration
session.next_unit() -> UnitCuration
session.prev_unit() -> UnitCuration
session.go_to_unit(unit_id: int) -> UnitCuration

# 決策
session.set_decision(unit_id, decision, notes='')
session.set_ai_classification(unit_id, classification)

# 匯出
session.get_final_labels() -> dict
session.export_decisions(output_path) -> DataFrame
session.get_summary() -> dict

# 持久化
session.save()
session = npa.CurationSession.load(session_dir)
```

## 視覺化模組

### plot_drift

```python
npa.plot_drift(
    recording: Recording,
    motion: dict = None,
    output: str = None,
    figsize: tuple = (12, 8),
)
```

繪製漂移/運動圖。

### plot_quality_metrics

```python
npa.plot_quality_metrics(
    analyzer: SortingAnalyzer,
    metrics: DataFrame = None,
    output: str = None,
)
```

繪製品質指標概覽。

### plot_unit_summary

```python
npa.plot_unit_summary(
    analyzer: SortingAnalyzer,
    unit_id: int,
    output: str = None,
)
```

繪製綜合單元摘要。

## SpikeInterface 整合

所有 neuropixels_analysis 函數都與 SpikeInterface 物件相容：

```python
import spikeinterface.full as si
import neuropixels_analysis as npa

# SpikeInterface recording 可與 npa 函數配合使用
recording = si.read_spikeglx('/path/')
rec = npa.preprocess(recording)

# 直接存取 SpikeInterface 進行進階使用
rec_filtered = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
```

## 常用參數

### 記錄參數
- `freq_min`：高通截止頻率（Hz）
- `freq_max`：低通截止頻率（Hz）
- `n_jobs`：平行工作數（-1 = 所有核心）

### 分選參數
- `output_folder`：結果儲存位置
- `sorter_params`：分選器特定參數字典

### 品質指標閾值
- `snr_threshold`：SNR 截止值（典型值 5）
- `isi_threshold`：ISI 違規截止值（典型值 0.01）
- `presence_threshold`：存在比率截止值（典型值 0.9）
