# 代謝體學工作流程

## 概述

PyOpenMS 為非標靶代謝體學分析提供專門工具，包括針對小分子最佳化的特徵偵測、加合物分組、化合物鑑定，以及與代謝體學資料庫的整合。

## 非標靶代謝體學管線

### 完整工作流程

```python
import pyopenms as ms

def metabolomics_pipeline(input_files, output_dir):
    """
    完整的非標靶代謝體學工作流程。

    Args:
        input_files: mzML 檔案路徑列表（每個樣本一個）
        output_dir: 輸出檔案目錄
    """

    # 步驟 1：峰值提取和特徵偵測
    feature_maps = []

    for mzml_file in input_files:
        print(f"Processing {mzml_file}...")

        # 載入資料
        exp = ms.MSExperiment()
        ms.MzMLFile().load(mzml_file, exp)

        # 如果需要進行峰值提取
        if not exp.getSpectrum(0).isSorted():
            picker = ms.PeakPickerHiRes()
            exp_picked = ms.MSExperiment()
            picker.pickExperiment(exp, exp_picked)
            exp = exp_picked

        # 特徵偵測
        ff = ms.FeatureFinder()
        params = ff.getParameters("centroided")

        # 代謝體學特定參數
        params.setValue("mass_trace:mz_tolerance", 5.0)  # ppm，代謝物較嚴格
        params.setValue("mass_trace:min_spectra", 5)
        params.setValue("isotopic_pattern:charge_low", 1)
        params.setValue("isotopic_pattern:charge_high", 2)  # 主要是單電荷

        features = ms.FeatureMap()
        ff.run("centroided", exp, features, params, ms.FeatureMap())

        features.setPrimaryMSRunPath([mzml_file.encode()])
        feature_maps.append(features)

        print(f"  Detected {features.size()} features")

    # 步驟 2：加合物偵測和分組
    print("Detecting adducts...")
    adduct_grouped_maps = []

    adduct_detector = ms.MetaboliteAdductDecharger()
    params = adduct_detector.getParameters()
    params.setValue("potential_adducts", "[M+H]+,[M+Na]+,[M+K]+,[M+NH4]+,[M-H]-,[M+Cl]-")
    params.setValue("charge_min", 1)
    params.setValue("charge_max", 1)
    adduct_detector.setParameters(params)

    for fm in feature_maps:
        fm_out = ms.FeatureMap()
        adduct_detector.compute(fm, fm_out, ms.ConsensusMap())
        adduct_grouped_maps.append(fm_out)

    # 步驟 3：RT 對齊
    print("Aligning retention times...")
    aligner = ms.MapAlignmentAlgorithmPoseClustering()

    params = aligner.getParameters()
    params.setValue("max_num_peaks_considered", 1000)
    params.setValue("pairfinder:distance_MZ:max_difference", 10.0)
    params.setValue("pairfinder:distance_MZ:unit", "ppm")
    aligner.setParameters(params)

    aligned_maps = []
    transformations = []
    aligner.align(adduct_grouped_maps, aligned_maps, transformations)

    # 步驟 4：特徵連結
    print("Linking features...")
    grouper = ms.FeatureGroupingAlgorithmQT()

    params = grouper.getParameters()
    params.setValue("distance_RT:max_difference", 60.0)  # 秒
    params.setValue("distance_MZ:max_difference", 5.0)  # ppm
    params.setValue("distance_MZ:unit", "ppm")
    grouper.setParameters(params)

    consensus_map = ms.ConsensusMap()
    grouper.group(aligned_maps, consensus_map)

    print(f"Created {consensus_map.size()} consensus features")

    # 步驟 5：填補間隙（填補遺漏值）
    print("Filling gaps...")
    # Python API 中無法直接使用間隙填補
    # 需要使用 TOPP 工具 FeatureFinderMetaboIdent

    # 步驟 6：匯出結果
    consensus_file = f"{output_dir}/consensus.consensusXML"
    ms.ConsensusXMLFile().store(consensus_file, consensus_map)

    # 匯出到 CSV 以進行下游分析
    df = consensus_map.get_df()
    csv_file = f"{output_dir}/metabolite_table.csv"
    df.to_csv(csv_file, index=False)

    print(f"Results saved to {output_dir}")

    return consensus_map

# 執行管線
input_files = ["sample1.mzML", "sample2.mzML", "sample3.mzML"]
consensus = metabolomics_pipeline(input_files, "output")
```

## 加合物偵測

### 設定加合物類型

```python
# 建立加合物偵測器
adduct_detector = ms.MetaboliteAdductDecharger()

# 設定常見加合物
params = adduct_detector.getParameters()

# 正離子模式加合物
positive_adducts = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M+NH4]+",
    "[2M+H]+",
    "[M+H-H2O]+"
]

# 負離子模式加合物
negative_adducts = [
    "[M-H]-",
    "[M+Cl]-",
    "[M+FA-H]-",  # 甲酸根
    "[2M-H]-"
]

# 設定正離子模式
params.setValue("potential_adducts", ",".join(positive_adducts))
params.setValue("charge_min", 1)
params.setValue("charge_max", 1)
params.setValue("max_neutrals", 1)
adduct_detector.setParameters(params)

# 應用加合物偵測
feature_map_out = ms.FeatureMap()
adduct_detector.compute(feature_map, feature_map_out, ms.ConsensusMap())
```

### 存取加合物資訊

```python
# 檢查加合物註釋
for feature in feature_map_out:
    # 如果有註釋則取得加合物類型
    if feature.metaValueExists("adduct"):
        adduct = feature.getMetaValue("adduct")
        neutral_mass = feature.getMetaValue("neutral_mass")
        print(f"m/z: {feature.getMZ():.4f}")
        print(f"  Adduct: {adduct}")
        print(f"  Neutral mass: {neutral_mass:.4f}")
```

## 化合物鑑定

### 基於質量的註釋

```python
# 使用化合物資料庫註釋特徵
from pyopenms import MassDecomposition

# 載入化合物資料庫（範例結構）
# 實務上使用外部資料庫如 HMDB、METLIN

compound_db = [
    {"name": "Glucose", "formula": "C6H12O6", "mass": 180.0634},
    {"name": "Citric acid", "formula": "C6H8O7", "mass": 192.0270},
    # ... 更多化合物
]

# 註釋特徵
mass_tolerance = 5.0  # ppm

for feature in feature_map:
    observed_mz = feature.getMZ()

    # 計算中性質量（假設 [M+H]+）
    neutral_mass = observed_mz - 1.007276  # 質子質量

    # 搜尋資料庫
    for compound in compound_db:
        mass_error_ppm = abs(neutral_mass - compound["mass"]) / compound["mass"] * 1e6

        if mass_error_ppm <= mass_tolerance:
            print(f"Potential match: {compound['name']}")
            print(f"  Observed m/z: {observed_mz:.4f}")
            print(f"  Expected mass: {compound['mass']:.4f}")
            print(f"  Error: {mass_error_ppm:.2f} ppm")
```

### 基於 MS/MS 的鑑定

```python
# 載入 MS2 資料
exp = ms.MSExperiment()
ms.MzMLFile().load("data_with_ms2.mzML", exp)

# 提取 MS2 光譜
ms2_spectra = []
for spec in exp:
    if spec.getMSLevel() == 2:
        ms2_spectra.append(spec)

print(f"Found {len(ms2_spectra)} MS2 spectra")

# 與光譜庫匹配
# （需要外部工具或自訂實作）
```

## 資料正規化

### 總離子流（TIC）正規化

```python
import numpy as np

# 載入共識圖
consensus_map = ms.ConsensusMap()
ms.ConsensusXMLFile().load("consensus.consensusXML", consensus_map)

# 計算每個樣本的 TIC
n_samples = len(consensus_map.getColumnHeaders())
tic_per_sample = np.zeros(n_samples)

for cons_feature in consensus_map:
    for handle in cons_feature.getFeatureList():
        map_idx = handle.getMapIndex()
        tic_per_sample[map_idx] += handle.getIntensity()

print("TIC per sample:", tic_per_sample)

# 正規化到中位數 TIC
median_tic = np.median(tic_per_sample)
normalization_factors = median_tic / tic_per_sample

print("Normalization factors:", normalization_factors)

# 應用正規化
consensus_map_normalized = ms.ConsensusMap(consensus_map)
for cons_feature in consensus_map_normalized:
    feature_list = cons_feature.getFeatureList()
    for handle in feature_list:
        map_idx = handle.getMapIndex()
        normalized_intensity = handle.getIntensity() * normalization_factors[map_idx]
        handle.setIntensity(normalized_intensity)
```

## 品質控制

### 變異係數（CV）過濾

```python
import pandas as pd
import numpy as np

# 匯出到 pandas
df = consensus_map.get_df()

# 假設 QC 樣本是名稱中含有 'QC' 的欄位
qc_cols = [col for col in df.columns if 'QC' in col]

if qc_cols:
    # 計算 QC 樣本中每個特徵的 CV
    qc_data = df[qc_cols]
    cv = (qc_data.std(axis=1) / qc_data.mean(axis=1)) * 100

    # 過濾 QC 樣本中 CV < 30% 的特徵
    good_features = df[cv < 30]

    print(f"Features before CV filter: {len(df)}")
    print(f"Features after CV filter: {len(good_features)}")
```

### 空白過濾

```python
# 移除空白樣本中存在的特徵
blank_cols = [col for col in df.columns if 'Blank' in col]
sample_cols = [col for col in df.columns if 'Sample' in col]

if blank_cols and sample_cols:
    # 計算空白和樣本中的平均強度
    blank_mean = df[blank_cols].mean(axis=1)
    sample_mean = df[sample_cols].mean(axis=1)

    # 保留樣本中強度比空白高 3 倍的特徵
    ratio = sample_mean / (blank_mean + 1)  # 加 1 避免除以零
    filtered_df = df[ratio > 3]

    print(f"Features before blank filtering: {len(df)}")
    print(f"Features after blank filtering: {len(filtered_df)}")
```

## 遺漏值填補

```python
import pandas as pd
import numpy as np

# 載入資料
df = consensus_map.get_df()

# 將零值替換為 NaN
df = df.replace(0, np.nan)

# 計算遺漏值
missing_per_feature = df.isnull().sum(axis=1)
print(f"Features with >50% missing: {sum(missing_per_feature > len(df.columns)/2)}")

# 簡單填補：用最小值替換
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        min_val = df[col].min() / 2  # 最小值的一半
        df[col].fillna(min_val, inplace=True)
```

## 代謝物表格匯出

### 建立分析就緒表格

```python
import pandas as pd

def create_metabolite_table(consensus_map, output_file):
    """
    建立用於統計分析的代謝物定量表格。
    """

    # 取得欄位標頭（檔案描述）
    headers = consensus_map.getColumnHeaders()

    # 初始化資料結構
    data = {
        'mz': [],
        'rt': [],
        'feature_id': []
    }

    # 添加樣本欄位
    for map_idx, header in headers.items():
        sample_name = header.label or f"Sample_{map_idx}"
        data[sample_name] = []

    # 提取特徵資料
    for idx, cons_feature in enumerate(consensus_map):
        data['mz'].append(cons_feature.getMZ())
        data['rt'].append(cons_feature.getRT())
        data['feature_id'].append(f"F{idx:06d}")

        # 初始化強度
        intensities = {map_idx: 0.0 for map_idx in headers.keys()}

        # 填入測量的強度
        for handle in cons_feature.getFeatureList():
            map_idx = handle.getMapIndex()
            intensities[map_idx] = handle.getIntensity()

        # 添加到資料結構
        for map_idx, header in headers.items():
            sample_name = header.label or f"Sample_{map_idx}"
            data[sample_name].append(intensities[map_idx])

    # 建立 DataFrame
    df = pd.DataFrame(data)

    # 按 RT 排序
    df = df.sort_values('rt')

    # 儲存到 CSV
    df.to_csv(output_file, index=False)

    print(f"Metabolite table with {len(df)} features saved to {output_file}")

    return df

# 建立表格
df = create_metabolite_table(consensus_map, "metabolite_table.csv")
```

## 與外部工具整合

### 匯出到 MetaboAnalyst

```python
def export_for_metaboanalyst(df, output_file):
    """
    格式化資料以供 MetaboAnalyst 輸入。

    需要樣本名稱作為欄位，特徵作為列。
    """

    # 轉置 DataFrame
    # 移除中繼資料欄位
    sample_cols = [col for col in df.columns if col not in ['mz', 'rt', 'feature_id']]

    # 提取樣本資料
    sample_data = df[sample_cols]

    # 轉置（樣本作為列，特徵作為欄位）
    df_transposed = sample_data.T

    # 添加特徵識別符作為欄位名稱
    df_transposed.columns = df['feature_id']

    # 儲存
    df_transposed.to_csv(output_file)

    print(f"MetaboAnalyst format saved to {output_file}")

# 匯出
export_for_metaboanalyst(df, "for_metaboanalyst.csv")
```

## 最佳實務

### 樣本量和重複

- 每 5-10 次注射包含 QC 樣本（混合樣本）
- 執行空白樣本以識別污染
- 每組至少使用 3 個生物重複
- 隨機化樣本注射順序

### 參數最佳化

在混合 QC 樣本上測試參數：

```python
# 測試不同的質量追蹤參數
mz_tolerances = [3.0, 5.0, 10.0]
min_spectra_values = [3, 5, 7]

for tol in mz_tolerances:
    for min_spec in min_spectra_values:
        ff = ms.FeatureFinder()
        params = ff.getParameters("centroided")
        params.setValue("mass_trace:mz_tolerance", tol)
        params.setValue("mass_trace:min_spectra", min_spec)

        features = ms.FeatureMap()
        ff.run("centroided", exp, features, params, ms.FeatureMap())

        print(f"tol={tol}, min_spec={min_spec}: {features.size()} features")
```

### 滯留時間視窗

根據層析方法調整：

```python
# 10 分鐘 LC 梯度
params.setValue("distance_RT:max_difference", 30.0)  # 30 秒

# 60 分鐘 LC 梯度
params.setValue("distance_RT:max_difference", 90.0)  # 90 秒
```
