# 核心資料結構

## 概述

PyOpenMS 使用具有 Python 綁定的 C++ 物件。了解這些核心資料結構對於有效的資料操作至關重要。

## 光譜和實驗物件

### MSExperiment

完整 LC-MS 實驗資料（光譜和層析圖）的容器。

```python
import pyopenms as ms

# 建立實驗
exp = ms.MSExperiment()

# 從檔案載入
ms.MzMLFile().load("data.mzML", exp)

# 存取屬性
print(f"Number of spectra: {exp.getNrSpectra()}")
print(f"Number of chromatograms: {exp.getNrChromatograms()}")

# 取得 RT 範圍
rts = [spec.getRT() for spec in exp]
print(f"RT range: {min(rts):.1f} - {max(rts):.1f} seconds")

# 存取個別光譜
spec = exp.getSpectrum(0)

# 迭代光譜
for spec in exp:
    if spec.getMSLevel() == 2:
        print(f"MS2 spectrum at RT {spec.getRT():.2f}")

# 取得中繼資料
exp_settings = exp.getExperimentalSettings()
instrument = exp_settings.getInstrument()
print(f"Instrument: {instrument.getName()}")
```

### MSSpectrum

具有 m/z 和強度陣列的個別質譜。

```python
# 建立空光譜
spec = ms.MSSpectrum()

# 從實驗取得
exp = ms.MSExperiment()
ms.MzMLFile().load("data.mzML", exp)
spec = exp.getSpectrum(0)

# 基本屬性
print(f"MS level: {spec.getMSLevel()}")
print(f"Retention time: {spec.getRT():.2f} seconds")
print(f"Number of peaks: {spec.size()}")

# 將峰值資料取得為 numpy 陣列
mz, intensity = spec.get_peaks()
print(f"m/z range: {mz.min():.2f} - {mz.max():.2f}")
print(f"Max intensity: {intensity.max():.0f}")

# 存取個別峰值
for i in range(min(5, spec.size())):  # 前 5 個峰值
    print(f"Peak {i}: m/z={mz[i]:.4f}, intensity={intensity[i]:.0f}")

# 前驅離子資訊（用於 MS2）
if spec.getMSLevel() == 2:
    precursors = spec.getPrecursors()
    if precursors:
        precursor = precursors[0]
        print(f"Precursor m/z: {precursor.getMZ():.4f}")
        print(f"Precursor charge: {precursor.getCharge()}")
        print(f"Precursor intensity: {precursor.getIntensity():.0f}")

# 設定峰值資料
new_mz = [100.0, 200.0, 300.0]
new_intensity = [1000.0, 2000.0, 1500.0]
spec.set_peaks((new_mz, new_intensity))
```

### MSChromatogram

層析軌跡（TIC、XIC 或 SRM 轉換）。

```python
# 從實驗存取層析圖
for chrom in exp.getChromatograms():
    print(f"Chromatogram ID: {chrom.getNativeID()}")

    # 取得資料
    rt, intensity = chrom.get_peaks()

    print(f"  RT points: {len(rt)}")
    print(f"  Max intensity: {intensity.max():.0f}")

    # 前驅離子資訊（用於 XIC）
    precursor = chrom.getPrecursor()
    print(f"  Precursor m/z: {precursor.getMZ():.4f}")
```

## 特徵物件

### Feature

偵測到的層析峰，具有 2D 空間範圍（RT-m/z）。

```python
# 載入特徵
feature_map = ms.FeatureMap()
ms.FeatureXMLFile().load("features.featureXML", feature_map)

# 存取個別特徵
feature = feature_map[0]

# 核心屬性
print(f"m/z: {feature.getMZ():.4f}")
print(f"RT: {feature.getRT():.2f} seconds")
print(f"Intensity: {feature.getIntensity():.0f}")
print(f"Charge: {feature.getCharge()}")

# 品質指標
print(f"Overall quality: {feature.getOverallQuality():.3f}")
print(f"Width (RT): {feature.getWidth():.2f}")

# 凸包（空間範圍）
hull = feature.getConvexHull()
print(f"Hull points: {hull.getHullPoints().size()}")

# 邊界框
bbox = hull.getBoundingBox()
print(f"RT range: {bbox.minPosition()[0]:.2f} - {bbox.maxPosition()[0]:.2f}")
print(f"m/z range: {bbox.minPosition()[1]:.4f} - {bbox.maxPosition()[1]:.4f}")

# 附屬特徵（同位素）
subordinates = feature.getSubordinates()
if subordinates:
    print(f"Isotopic features: {len(subordinates)}")
    for sub in subordinates:
        print(f"  m/z: {sub.getMZ():.4f}, intensity: {sub.getIntensity():.0f}")

# 中繼資料值
if feature.metaValueExists("label"):
    label = feature.getMetaValue("label")
    print(f"Label: {label}")
```

### FeatureMap

來自單一 LC-MS 執行的特徵集合。

```python
# 建立特徵圖
feature_map = ms.FeatureMap()

# 從檔案載入
ms.FeatureXMLFile().load("features.featureXML", feature_map)

# 存取屬性
print(f"Number of features: {feature_map.size()}")

# 取得唯一特徵
print(f"Unique features: {feature_map.getUniqueId()}")

# 中繼資料
primary_path = feature_map.getPrimaryMSRunPath()
if primary_path:
    print(f"Source file: {primary_path[0].decode()}")

# 迭代特徵
for feature in feature_map:
    print(f"Feature: m/z={feature.getMZ():.4f}, RT={feature.getRT():.2f}")

# 添加新特徵
new_feature = ms.Feature()
new_feature.setMZ(500.0)
new_feature.setRT(300.0)
new_feature.setIntensity(10000.0)
feature_map.push_back(new_feature)

# 排序特徵
feature_map.sortByRT()  # 或 sortByMZ()、sortByIntensity()

# 匯出到 pandas
df = feature_map.get_df()
print(df.head())
```

### ConsensusFeature

跨多個樣本連結的特徵。

```python
# 載入共識圖
consensus_map = ms.ConsensusMap()
ms.ConsensusXMLFile().load("consensus.consensusXML", consensus_map)

# 存取共識特徵
cons_feature = consensus_map[0]

# 共識屬性
print(f"Consensus m/z: {cons_feature.getMZ():.4f}")
print(f"Consensus RT: {cons_feature.getRT():.2f}")
print(f"Consensus intensity: {cons_feature.getIntensity():.0f}")

# 取得特徵控制代碼（個別圖的特徵）
feature_list = cons_feature.getFeatureList()
print(f"Present in {len(feature_list)} maps")

for handle in feature_list:
    map_idx = handle.getMapIndex()
    intensity = handle.getIntensity()
    mz = handle.getMZ()
    rt = handle.getRT()

    print(f"  Map {map_idx}: m/z={mz:.4f}, RT={rt:.2f}, intensity={intensity:.0f}")

# 取得原始圖中的唯一 ID
for handle in feature_list:
    unique_id = handle.getUniqueId()
    print(f"Unique ID: {unique_id}")
```

### ConsensusMap

跨樣本的共識特徵集合。

```python
# 建立共識圖
consensus_map = ms.ConsensusMap()

# 從檔案載入
ms.ConsensusXMLFile().load("consensus.consensusXML", consensus_map)

# 存取屬性
print(f"Consensus features: {consensus_map.size()}")

# 欄位標頭（檔案描述）
headers = consensus_map.getColumnHeaders()
print(f"Number of files: {len(headers)}")

for map_idx, description in headers.items():
    print(f"Map {map_idx}:")
    print(f"  Filename: {description.filename}")
    print(f"  Label: {description.label}")
    print(f"  Size: {description.size}")

# 迭代共識特徵
for cons_feature in consensus_map:
    print(f"Consensus feature: m/z={cons_feature.getMZ():.4f}")

# 匯出到 DataFrame
df = consensus_map.get_df()
```

## 鑑定物件

### PeptideIdentification

單一光譜的鑑定結果。

```python
# 載入鑑定
protein_ids = []
peptide_ids = []
ms.IdXMLFile().load("identifications.idXML", protein_ids, peptide_ids)

# 存取肽段鑑定
peptide_id = peptide_ids[0]

# 光譜中繼資料
print(f"RT: {peptide_id.getRT():.2f}")
print(f"m/z: {peptide_id.getMZ():.4f}")

# 鑑定中繼資料
print(f"Identifier: {peptide_id.getIdentifier()}")
print(f"Score type: {peptide_id.getScoreType()}")
print(f"Higher score better: {peptide_id.isHigherScoreBetter()}")

# 取得肽段命中
hits = peptide_id.getHits()
print(f"Number of hits: {len(hits)}")

for hit in hits:
    print(f"  Sequence: {hit.getSequence().toString()}")
    print(f"  Score: {hit.getScore()}")
    print(f"  Charge: {hit.getCharge()}")
```

### PeptideHit

與光譜匹配的個別肽段。

```python
# 存取命中
hit = peptide_id.getHits()[0]

# 序列資訊
sequence = hit.getSequence()
print(f"Sequence: {sequence.toString()}")
print(f"Mass: {sequence.getMonoWeight():.4f}")

# 分數和排名
print(f"Score: {hit.getScore()}")
print(f"Rank: {hit.getRank()}")

# 電荷狀態
print(f"Charge: {hit.getCharge()}")

# 蛋白質登錄號
accessions = hit.extractProteinAccessionsSet()
for acc in accessions:
    print(f"Protein: {acc.decode()}")

# 中繼值（額外分數、誤差）
if hit.metaValueExists("MS:1002252"):  # 質量誤差
    mass_error = hit.getMetaValue("MS:1002252")
    print(f"Mass error: {mass_error:.4f} ppm")
```

### ProteinIdentification

蛋白質層級的鑑定資訊。

```python
# 存取蛋白質鑑定
protein_id = protein_ids[0]

# 搜尋引擎資訊
print(f"Search engine: {protein_id.getSearchEngine()}")
print(f"Search engine version: {protein_id.getSearchEngineVersion()}")

# 搜尋參數
search_params = protein_id.getSearchParameters()
print(f"Database: {search_params.db}")
print(f"Enzyme: {search_params.digestion_enzyme.getName()}")
print(f"Missed cleavages: {search_params.missed_cleavages}")
print(f"Precursor tolerance: {search_params.precursor_mass_tolerance}")

# 蛋白質命中
hits = protein_id.getHits()
for hit in hits:
    print(f"Accession: {hit.getAccession()}")
    print(f"Score: {hit.getScore()}")
    print(f"Coverage: {hit.getCoverage():.1f}%")
```

### ProteinHit

個別蛋白質鑑定。

```python
# 存取蛋白質命中
protein_hit = protein_id.getHits()[0]

# 蛋白質資訊
print(f"Accession: {protein_hit.getAccession()}")
print(f"Description: {protein_hit.getDescription()}")
print(f"Sequence: {protein_hit.getSequence()}")

# 評分
print(f"Score: {protein_hit.getScore()}")
print(f"Coverage: {protein_hit.getCoverage():.1f}%")

# 排名
print(f"Rank: {protein_hit.getRank()}")
```

## 序列物件

### AASequence

帶有修飾的胺基酸序列。

```python
# 從字串建立序列
seq = ms.AASequence.fromString("PEPTIDE")

# 基本屬性
print(f"Sequence: {seq.toString()}")
print(f"Length: {seq.size()}")
print(f"Monoisotopic mass: {seq.getMonoWeight():.4f}")
print(f"Average mass: {seq.getAverageWeight():.4f}")

# 個別殘基
for i in range(seq.size()):
    residue = seq.getResidue(i)
    print(f"Position {i}: {residue.getOneLetterCode()}")
    print(f"  Mass: {residue.getMonoWeight():.4f}")
    print(f"  Formula: {residue.getFormula().toString()}")

# 修飾序列
mod_seq = ms.AASequence.fromString("PEPTIDEM(Oxidation)K")
print(f"Modified: {mod_seq.isModified()}")

# 檢查修飾
for i in range(mod_seq.size()):
    residue = mod_seq.getResidue(i)
    if residue.isModified():
        print(f"Modification at {i}: {residue.getModificationName()}")

# N 端和 C 端修飾
term_mod_seq = ms.AASequence.fromString("(Acetyl)PEPTIDE(Amidated)")
```

### EmpiricalFormula

分子式表示。

```python
# 建立分子式
formula = ms.EmpiricalFormula("C6H12O6")  # 葡萄糖

# 屬性
print(f"Formula: {formula.toString()}")
print(f"Monoisotopic mass: {formula.getMonoWeight():.4f}")
print(f"Average mass: {formula.getAverageWeight():.4f}")

# 元素組成
print(f"Carbon atoms: {formula.getNumberOf(b'C')}")
print(f"Hydrogen atoms: {formula.getNumberOf(b'H')}")
print(f"Oxygen atoms: {formula.getNumberOf(b'O')}")

# 算術運算
formula2 = ms.EmpiricalFormula("H2O")
combined = formula + formula2  # 加水
print(f"Combined: {combined.toString()}")
```

## 參數物件

### Param

演算法使用的通用參數容器。

```python
# 取得演算法參數
algo = ms.GaussFilter()
params = algo.getParameters()

# 列出所有參數
for key in params.keys():
    value = params.getValue(key)
    print(f"{key}: {value}")

# 取得特定參數
gaussian_width = params.getValue("gaussian_width")
print(f"Gaussian width: {gaussian_width}")

# 設定參數
params.setValue("gaussian_width", 0.2)

# 應用修改的參數
algo.setParameters(params)

# 複製參數
params_copy = ms.Param(params)
```

## 最佳實務

### 記憶體管理

```python
# 對於大型檔案，使用索引存取而非完整載入
indexed_mzml = ms.IndexedMzMLFileLoader()
indexed_mzml.load("large_file.mzML")

# 存取特定光譜而不載入整個檔案
spec = indexed_mzml.getSpectrumById(100)
```

### 型別轉換

```python
# 將峰值陣列轉換為 numpy
import numpy as np

mz, intensity = spec.get_peaks()
# 這些已經是 numpy 陣列

# 可以執行 numpy 操作
filtered_mz = mz[intensity > 1000]
```

### 物件複製

```python
# 建立深度複製
exp_copy = ms.MSExperiment(exp)

# 對複製的修改不會影響原始
```
