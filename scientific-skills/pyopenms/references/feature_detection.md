# 特徵偵測和連結

## 概述

特徵偵測識別 LC-MS 資料中的持續訊號（層析峰）。特徵連結結合多個樣本的特徵以進行定量比較。

## 特徵偵測基礎

特徵代表層析峰，特徵包括：
- m/z 值（質荷比）
- 滯留時間（RT）
- 強度
- 品質分數
- 凸包（RT-m/z 空間中的空間範圍）

## 特徵發現

### Feature Finder Multiples（FFM）

質心化資料特徵偵測的標準演算法：

```python
import pyopenms as ms

# 載入質心化資料
exp = ms.MSExperiment()
ms.MzMLFile().load("centroided.mzML", exp)

# 建立特徵發現器
ff = ms.FeatureFinder()

# 取得預設參數
params = ff.getParameters("centroided")

# 修改關鍵參數
params.setValue("mass_trace:mz_tolerance", 10.0)  # ppm
params.setValue("mass_trace:min_spectra", 7)  # 每個特徵最少掃描次數
params.setValue("isotopic_pattern:charge_low", 1)
params.setValue("isotopic_pattern:charge_high", 4)

# 執行特徵偵測
features = ms.FeatureMap()
ff.run("centroided", exp, features, params, ms.FeatureMap())

print(f"Detected {features.size()} features")

# 儲存特徵
ms.FeatureXMLFile().store("features.featureXML", features)
```

### 代謝體學特徵發現器

針對小分子最佳化：

```python
# 建立代謝體學特徵發現器
ff = ms.FeatureFinder()

# 取得代謝體學特定參數
params = ff.getParameters("centroided")

# 設定代謝體學
params.setValue("mass_trace:mz_tolerance", 5.0)  # 較低的容差
params.setValue("mass_trace:min_spectra", 5)
params.setValue("isotopic_pattern:charge_low", 1)  # 主要是單電荷
params.setValue("isotopic_pattern:charge_high", 2)

# 執行偵測
features = ms.FeatureMap()
ff.run("centroided", exp, features, params, ms.FeatureMap())
```

## 存取特徵資料

### 迭代特徵

```python
# 載入特徵
feature_map = ms.FeatureMap()
ms.FeatureXMLFile().load("features.featureXML", feature_map)

# 存取個別特徵
for feature in feature_map:
    print(f"m/z: {feature.getMZ():.4f}")
    print(f"RT: {feature.getRT():.2f}")
    print(f"Intensity: {feature.getIntensity():.0f}")
    print(f"Charge: {feature.getCharge()}")
    print(f"Quality: {feature.getOverallQuality():.3f}")
    print(f"Width (RT): {feature.getWidth():.2f}")

    # 取得凸包
    hull = feature.getConvexHull()
    print(f"Hull points: {hull.getHullPoints().size()}")
```

### 特徵附屬（同位素模式）

```python
# 存取同位素模式
for feature in feature_map:
    # 取得附屬特徵（同位素）
    subordinates = feature.getSubordinates()

    if subordinates:
        print(f"Main feature m/z: {feature.getMZ():.4f}")
        for sub in subordinates:
            print(f"  Isotope m/z: {sub.getMZ():.4f}")
            print(f"  Isotope intensity: {sub.getIntensity():.0f}")
```

### 匯出到 Pandas

```python
import pandas as pd

# 轉換為 DataFrame
df = feature_map.get_df()

print(df.columns)
# 典型欄位：RT、mz、intensity、charge、quality

# 分析特徵
print(f"Mean intensity: {df['intensity'].mean()}")
print(f"RT range: {df['RT'].min():.1f} - {df['RT'].max():.1f}")
```

## 特徵連結

### 圖對齊

連結前對齊滯留時間：

```python
# 載入多個特徵圖
fm1 = ms.FeatureMap()
fm2 = ms.FeatureMap()
ms.FeatureXMLFile().load("sample1.featureXML", fm1)
ms.FeatureXMLFile().load("sample2.featureXML", fm2)

# 建立對齊器
aligner = ms.MapAlignmentAlgorithmPoseClustering()

# 對齊圖
fm_aligned = []
transformations = []
aligner.align([fm1, fm2], fm_aligned, transformations)
```

### 特徵連結演算法

跨樣本連結特徵：

```python
# 建立特徵分組演算法
grouper = ms.FeatureGroupingAlgorithmQT()

# 設定參數
params = grouper.getParameters()
params.setValue("distance_RT:max_difference", 30.0)  # 最大 RT 差異（秒）
params.setValue("distance_MZ:max_difference", 10.0)  # 最大 m/z 差異（ppm）
params.setValue("distance_MZ:unit", "ppm")
grouper.setParameters(params)

# 準備特徵圖
feature_maps = [fm1, fm2, fm3]

# 建立共識圖
consensus_map = ms.ConsensusMap()

# 連結特徵
grouper.group(feature_maps, consensus_map)

print(f"Created {consensus_map.size()} consensus features")

# 儲存共識圖
ms.ConsensusXMLFile().store("consensus.consensusXML", consensus_map)
```

## 共識特徵

### 存取共識資料

```python
# 載入共識圖
consensus_map = ms.ConsensusMap()
ms.ConsensusXMLFile().load("consensus.consensusXML", consensus_map)

# 迭代共識特徵
for cons_feature in consensus_map:
    print(f"Consensus m/z: {cons_feature.getMZ():.4f}")
    print(f"Consensus RT: {cons_feature.getRT():.2f}")

    # 取得來自個別圖的特徵
    for handle in cons_feature.getFeatureList():
        map_idx = handle.getMapIndex()
        intensity = handle.getIntensity()
        print(f"  Sample {map_idx}: intensity {intensity:.0f}")
```

### 共識圖中繼資料

```python
# 存取檔案描述（圖中繼資料）
file_descriptions = consensus_map.getColumnHeaders()

for map_idx, description in file_descriptions.items():
    print(f"Map {map_idx}:")
    print(f"  Filename: {description.filename}")
    print(f"  Label: {description.label}")
    print(f"  Size: {description.size}")
```

## 加合物偵測

識別相同分子的不同離子化形式：

```python
# 建立加合物偵測器
adduct_detector = ms.MetaboliteAdductDecharger()

# 設定參數
params = adduct_detector.getParameters()
params.setValue("potential_adducts", "[M+H]+,[M+Na]+,[M+K]+,[M-H]-")
params.setValue("charge_min", 1)
params.setValue("charge_max", 1)
params.setValue("max_neutrals", 1)
adduct_detector.setParameters(params)

# 偵測加合物
feature_map_out = ms.FeatureMap()
adduct_detector.compute(feature_map, feature_map_out, ms.ConsensusMap())
```

## 完整特徵偵測工作流程

### 端到端範例

```python
import pyopenms as ms

def feature_detection_workflow(input_files, output_consensus):
    """
    完整工作流程：跨樣本的特徵偵測和連結。

    Args:
        input_files: mzML 檔案路徑列表
        output_consensus: 輸出 consensusXML 檔案路徑
    """

    feature_maps = []

    # 步驟 1：在每個檔案中偵測特徵
    for mzml_file in input_files:
        print(f"Processing {mzml_file}...")

        # 載入實驗
        exp = ms.MSExperiment()
        ms.MzMLFile().load(mzml_file, exp)

        # 發現特徵
        ff = ms.FeatureFinder()
        params = ff.getParameters("centroided")
        params.setValue("mass_trace:mz_tolerance", 10.0)
        params.setValue("mass_trace:min_spectra", 7)

        features = ms.FeatureMap()
        ff.run("centroided", exp, features, params, ms.FeatureMap())

        # 在特徵圖中儲存檔案名稱
        features.setPrimaryMSRunPath([mzml_file.encode()])

        feature_maps.append(features)
        print(f"  Found {features.size()} features")

    # 步驟 2：對齊滯留時間
    print("Aligning retention times...")
    aligner = ms.MapAlignmentAlgorithmPoseClustering()
    aligned_maps = []
    transformations = []
    aligner.align(feature_maps, aligned_maps, transformations)

    # 步驟 3：連結特徵
    print("Linking features across samples...")
    grouper = ms.FeatureGroupingAlgorithmQT()
    params = grouper.getParameters()
    params.setValue("distance_RT:max_difference", 30.0)
    params.setValue("distance_MZ:max_difference", 10.0)
    params.setValue("distance_MZ:unit", "ppm")
    grouper.setParameters(params)

    consensus_map = ms.ConsensusMap()
    grouper.group(aligned_maps, consensus_map)

    # 儲存結果
    ms.ConsensusXMLFile().store(output_consensus, consensus_map)

    print(f"Created {consensus_map.size()} consensus features")
    print(f"Results saved to {output_consensus}")

    return consensus_map

# 執行工作流程
input_files = ["sample1.mzML", "sample2.mzML", "sample3.mzML"]
consensus = feature_detection_workflow(input_files, "consensus.consensusXML")
```

## 特徵過濾

### 按品質過濾

```python
# 按品質分數過濾特徵
filtered_features = ms.FeatureMap()

for feature in feature_map:
    if feature.getOverallQuality() > 0.5:  # 品質閾值
        filtered_features.push_back(feature)

print(f"Kept {filtered_features.size()} high-quality features")
```

### 按強度過濾

```python
# 只保留高強度特徵
min_intensity = 10000

filtered_features = ms.FeatureMap()
for feature in feature_map:
    if feature.getIntensity() >= min_intensity:
        filtered_features.push_back(feature)
```

### 按 m/z 範圍過濾

```python
# 提取特定 m/z 範圍的特徵
mz_min = 200.0
mz_max = 800.0

filtered_features = ms.FeatureMap()
for feature in feature_map:
    mz = feature.getMZ()
    if mz_min <= mz <= mz_max:
        filtered_features.push_back(feature)
```

## 特徵註釋

### 添加鑑定資訊

```python
# 使用肽段鑑定註釋特徵
# 載入鑑定
protein_ids = []
peptide_ids = []
ms.IdXMLFile().load("identifications.idXML", protein_ids, peptide_ids)

# 建立 ID 映射器
mapper = ms.IDMapper()

# 將 ID 映射到特徵
mapper.annotate(feature_map, peptide_ids, protein_ids)

# 檢查註釋
for feature in feature_map:
    peptide_ids_for_feature = feature.getPeptideIdentifications()
    if peptide_ids_for_feature:
        print(f"Feature at {feature.getMZ():.4f} m/z identified")
```

## 最佳實務

### 參數最佳化

針對您的資料類型最佳化參數：

```python
# 測試不同的容差值
mz_tolerances = [5.0, 10.0, 20.0]  # ppm

for tol in mz_tolerances:
    ff = ms.FeatureFinder()
    params = ff.getParameters("centroided")
    params.setValue("mass_trace:mz_tolerance", tol)

    features = ms.FeatureMap()
    ff.run("centroided", exp, features, params, ms.FeatureMap())

    print(f"Tolerance {tol} ppm: {features.size()} features")
```

### 視覺檢查

匯出特徵以進行視覺化：

```python
# 轉換為 DataFrame 以進行繪圖
df = feature_map.get_df()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['RT'], df['mz'], s=df['intensity']/1000, alpha=0.5)
plt.xlabel('Retention Time (s)')
plt.ylabel('m/z')
plt.title('Feature Map')
plt.colorbar(label='Intensity (scaled)')
plt.show()
```
