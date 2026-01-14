# 訊號處理

## 概述

PyOpenMS 提供處理原始質譜資料的演算法，包括平滑、濾波、峰值提取、質心化、正規化和去卷積。

## 演算法模式

大多數訊號處理演算法遵循標準模式：

```python
import pyopenms as ms

# 1. 建立演算法實例
algo = ms.AlgorithmName()

# 2. 取得並修改參數
params = algo.getParameters()
params.setValue("parameter_name", value)
algo.setParameters(params)

# 3. 應用到資料
algo.filterExperiment(exp)  # 或 filterSpectrum(spec)
```

## 平滑

### 高斯濾波器

應用高斯平滑以減少雜訊：

```python
# 建立高斯濾波器
gaussian = ms.GaussFilter()

# 設定參數
params = gaussian.getParameters()
params.setValue("gaussian_width", 0.2)  # m/z 或 RT 單位的寬度
params.setValue("ppm_tolerance", 10.0)  # 用於 m/z 維度
params.setValue("use_ppm_tolerance", "true")
gaussian.setParameters(params)

# 應用到實驗
gaussian.filterExperiment(exp)

# 或應用到單一光譜
spec = exp.getSpectrum(0)
gaussian.filterSpectrum(spec)
```

### Savitzky-Golay 濾波器

保留峰形的多項式平滑：

```python
# 建立 Savitzky-Golay 濾波器
sg_filter = ms.SavitzkyGolayFilter()

# 設定參數
params = sg_filter.getParameters()
params.setValue("frame_length", 11)  # 視窗大小（必須為奇數）
params.setValue("polynomial_order", 4)  # 多項式次數
sg_filter.setParameters(params)

# 應用平滑
sg_filter.filterExperiment(exp)
```

## 峰值提取和質心化

### 高解析度峰值提取器

在高解析度資料中偵測峰值：

```python
# 建立峰值提取器
peak_picker = ms.PeakPickerHiRes()

# 設定參數
params = peak_picker.getParameters()
params.setValue("signal_to_noise", 3.0)  # S/N 閾值
params.setValue("spacing_difference", 1.5)  # 最小峰值間距
peak_picker.setParameters(params)

# 提取峰值
exp_picked = ms.MSExperiment()
peak_picker.pickExperiment(exp, exp_picked)
```

### CWT 峰值提取器

連續小波轉換峰值提取：

```python
# 建立 CWT 峰值提取器
cwt_picker = ms.PeakPickerCWT()

# 設定參數
params = cwt_picker.getParameters()
params.setValue("signal_to_noise", 1.0)
params.setValue("peak_width", 0.15)  # 預期峰值寬度
cwt_picker.setParameters(params)

# 提取峰值
cwt_picker.pickExperiment(exp, exp_picked)
```

## 正規化

### 正規化器

正規化光譜內的峰值強度：

```python
# 建立正規化器
normalizer = ms.Normalizer()

# 設定正規化方法
params = normalizer.getParameters()
params.setValue("method", "to_one")  # 選項："to_one"、"to_TIC"
normalizer.setParameters(params)

# 應用正規化
normalizer.filterExperiment(exp)
```

## 峰值過濾

### 閾值過濾器

移除強度低於閾值的峰值：

```python
# 建立閾值過濾器
mower = ms.ThresholdMower()

# 設定閾值
params = mower.getParameters()
params.setValue("threshold", 1000.0)  # 絕對強度閾值
mower.setParameters(params)

# 應用過濾器
mower.filterExperiment(exp)
```

### 視窗過濾器

在滑動視窗中只保留最高峰值：

```python
# 建立視窗過濾器
window_mower = ms.WindowMower()

# 設定參數
params = window_mower.getParameters()
params.setValue("windowsize", 50.0)  # m/z 中的視窗大小
params.setValue("peakcount", 2)  # 每個視窗保留前 N 個峰值
window_mower.setParameters(params)

# 應用過濾器
window_mower.filterExperiment(exp)
```

### N 個最大峰值

只保留 N 個最強峰值：

```python
# 建立 N 個最大過濾器
n_largest = ms.NLargest()

# 設定參數
params = n_largest.getParameters()
params.setValue("n", 200)  # 保留 200 個最強峰值
n_largest.setParameters(params)

# 應用過濾器
n_largest.filterExperiment(exp)
```

## 基線消除

### 形態學濾波器

使用形態學操作移除基線：

```python
# 建立形態學濾波器
morph_filter = ms.MorphologicalFilter()

# 設定參數
params = morph_filter.getParameters()
params.setValue("struc_elem_length", 3.0)  # 結構元素大小
params.setValue("method", "tophat")  # 方法："tophat"、"bothat"、"erosion"、"dilation"
morph_filter.setParameters(params)

# 應用濾波器
morph_filter.filterExperiment(exp)
```

## 光譜合併

### 光譜合併器

將多個光譜合併為一個：

```python
# 建立合併器
merger = ms.SpectraMerger()

# 設定參數
params = merger.getParameters()
params.setValue("average_gaussian:spectrum_type", "profile")
params.setValue("average_gaussian:rt_FWHM", 5.0)  # RT 視窗
merger.setParameters(params)

# 合併光譜
merger.mergeSpectraBlockWise(exp)
```

## 去卷積

### 電荷去卷積

確定電荷狀態並轉換為中性質量：

```python
# 建立特徵去卷積器
deconvoluter = ms.FeatureDeconvolution()

# 設定參數
params = deconvoluter.getParameters()
params.setValue("charge_min", 1)
params.setValue("charge_max", 4)
params.setValue("potential_charge_states", "1,2,3,4")
deconvoluter.setParameters(params)

# 應用去卷積
feature_map_out = ms.FeatureMap()
deconvoluter.compute(exp, feature_map, feature_map_out, ms.ConsensusMap())
```

### 同位素去卷積

移除同位素模式：

```python
# 建立同位素小波轉換
isotope_wavelet = ms.IsotopeWaveletTransform()

# 設定參數
params = isotope_wavelet.getParameters()
params.setValue("max_charge", 3)
params.setValue("intensity_threshold", 10.0)
isotope_wavelet.setParameters(params)

# 應用轉換
isotope_wavelet.transform(exp)
```

## 滯留時間對齊

### 圖對齊

跨多次執行對齊滯留時間：

```python
# 建立圖對齊器
aligner = ms.MapAlignmentAlgorithmPoseClustering()

# 載入多個實驗
exp1 = ms.MSExperiment()
exp2 = ms.MSExperiment()
ms.MzMLFile().load("run1.mzML", exp1)
ms.MzMLFile().load("run2.mzML", exp2)

# 建立參考
reference = ms.MSExperiment()

# 對齊實驗
transformations = []
aligner.align(exp1, exp2, transformations)

# 應用轉換
transformer = ms.MapAlignmentTransformer()
transformer.transformRetentionTimes(exp2, transformations[0])
```

## 質量校正

### 內部校正

使用已知參考質量校正質量軸：

```python
# 建立內部校正
calibration = ms.InternalCalibration()

# 設定參考質量
reference_masses = [500.0, 1000.0, 1500.0]  # 已知 m/z 值

# 校正
calibration.calibrate(exp, reference_masses)
```

## 品質控制

### 光譜統計

計算品質指標：

```python
# 取得光譜
spec = exp.getSpectrum(0)

# 計算統計
mz, intensity = spec.get_peaks()

# 總離子流
tic = sum(intensity)

# 基峰
base_peak_intensity = max(intensity)
base_peak_mz = mz[intensity.argmax()]

print(f"TIC: {tic}")
print(f"Base peak: {base_peak_mz} m/z at {base_peak_intensity}")
```

## 光譜前處理管線

### 完整前處理範例

```python
import pyopenms as ms

def preprocess_experiment(input_file, output_file):
    """完整前處理管線。"""

    # 載入資料
    exp = ms.MSExperiment()
    ms.MzMLFile().load(input_file, exp)

    # 1. 使用高斯濾波器平滑
    gaussian = ms.GaussFilter()
    gaussian.filterExperiment(exp)

    # 2. 提取峰值
    picker = ms.PeakPickerHiRes()
    exp_picked = ms.MSExperiment()
    picker.pickExperiment(exp, exp_picked)

    # 3. 正規化強度
    normalizer = ms.Normalizer()
    params = normalizer.getParameters()
    params.setValue("method", "to_TIC")
    normalizer.setParameters(params)
    normalizer.filterExperiment(exp_picked)

    # 4. 過濾低強度峰值
    mower = ms.ThresholdMower()
    params = mower.getParameters()
    params.setValue("threshold", 10.0)
    mower.setParameters(params)
    mower.filterExperiment(exp_picked)

    # 儲存處理後的資料
    ms.MzMLFile().store(output_file, exp_picked)

    return exp_picked

# 執行管線
exp_processed = preprocess_experiment("raw_data.mzML", "processed_data.mzML")
```

## 最佳實務

### 參數最佳化

在代表性資料上測試參數：

```python
# 嘗試不同的高斯寬度
widths = [0.1, 0.2, 0.5]

for width in widths:
    exp_test = ms.MSExperiment()
    ms.MzMLFile().load("test_data.mzML", exp_test)

    gaussian = ms.GaussFilter()
    params = gaussian.getParameters()
    params.setValue("gaussian_width", width)
    gaussian.setParameters(params)
    gaussian.filterExperiment(exp_test)

    # 評估品質
    # ... 添加評估程式碼 ...
```

### 保留原始資料

保留原始資料以進行比較：

```python
# 載入原始
exp_original = ms.MSExperiment()
ms.MzMLFile().load("data.mzML", exp_original)

# 建立處理用複本
exp_processed = ms.MSExperiment(exp_original)

# 處理複本
gaussian = ms.GaussFilter()
gaussian.filterExperiment(exp_processed)

# 原始保持不變
```

### 輪廓 vs 質心資料

處理前檢查資料類型：

```python
# 檢查光譜是否已質心化
spec = exp.getSpectrum(0)

if spec.isSorted():
    # 可能是質心化的
    print("Centroid data")
else:
    # 可能是輪廓
    print("Profile data - apply peak picking")
```
