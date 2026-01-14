# 檔案 I/O 和資料格式

## 概述

PyOpenMS 支援多種質譜檔案格式的讀取和寫入。本指南涵蓋檔案處理策略和格式特定的操作。

## 支援的格式

### 光譜資料格式

- **mzML**：標準的 XML 質譜資料格式
- **mzXML**：較早期的 XML 格式
- **mzData**：XML 格式（已棄用但仍支援）

### 鑑定格式

- **idXML**：OpenMS 原生鑑定格式
- **mzIdentML**：標準的鑑定資料 XML 格式
- **pepXML**：X! Tandem 格式
- **protXML**：蛋白質鑑定格式

### 特徵和定量格式

- **featureXML**：偵測到的特徵的 OpenMS 格式
- **consensusXML**：跨樣本共識特徵的格式
- **mzTab**：報告用的 Tab 分隔格式

### 序列和資料庫格式

- **FASTA**：蛋白質/肽段序列
- **TraML**：標靶實驗的轉換列表

## 讀取 mzML 檔案

### 記憶體內載入

將整個檔案載入記憶體（適合較小的檔案）：

```python
import pyopenms as ms

# 建立實驗容器
exp = ms.MSExperiment()

# 載入檔案
ms.MzMLFile().load("sample.mzML", exp)

# 存取資料
print(f"Spectra: {exp.getNrSpectra()}")
print(f"Chromatograms: {exp.getNrChromatograms()}")
```

### 索引存取

大型檔案的高效隨機存取：

```python
# 建立索引存取
indexed_mzml = ms.IndexedMzMLFileLoader()
indexed_mzml.load("large_file.mzML")

# 按索引取得特定光譜
spec = indexed_mzml.getSpectrumById(100)

# 按原生 ID 存取
spec = indexed_mzml.getSpectrumByNativeId("scan=5000")
```

### 串流存取

非常大型檔案的記憶體高效處理：

```python
# 定義消費者函數
class SpectrumProcessor(ms.MSExperimentConsumer):
    def __init__(self):
        super().__init__()
        self.count = 0

    def consumeSpectrum(self, spec):
        # 處理光譜
        if spec.getMSLevel() == 2:
            self.count += 1

# 串流檔案
consumer = SpectrumProcessor()
ms.MzMLFile().transform("large.mzML", consumer)
print(f"Processed {consumer.count} MS2 spectra")
```

### 快取存取

記憶體使用和速度之間的平衡：

```python
# 使用磁碟快取
options = ms.CachedmzML()
options.setMetaDataOnly(False)

exp = ms.MSExperiment()
ms.CachedmzMLHandler().load("sample.mzML", exp, options)
```

## 寫入 mzML 檔案

### 基本寫入

```python
# 建立或修改實驗
exp = ms.MSExperiment()
# ... 添加光譜 ...

# 寫入檔案
ms.MzMLFile().store("output.mzML", exp)
```

### 壓縮選項

```python
# 設定壓縮
file_handler = ms.MzMLFile()

options = ms.PeakFileOptions()
options.setCompression(True)  # 啟用壓縮
file_handler.setOptions(options)

file_handler.store("compressed.mzML", exp)
```

## 讀取鑑定資料

### idXML 格式

```python
# 載入鑑定結果
protein_ids = []
peptide_ids = []

ms.IdXMLFile().load("identifications.idXML", protein_ids, peptide_ids)

# 存取肽段鑑定
for peptide_id in peptide_ids:
    print(f"RT: {peptide_id.getRT()}")
    print(f"MZ: {peptide_id.getMZ()}")

    # 取得肽段命中
    for hit in peptide_id.getHits():
        print(f"  Sequence: {hit.getSequence().toString()}")
        print(f"  Score: {hit.getScore()}")
        print(f"  Charge: {hit.getCharge()}")
```

### mzIdentML 格式

```python
# 讀取 mzIdentML
protein_ids = []
peptide_ids = []

ms.MzIdentMLFile().load("results.mzid", protein_ids, peptide_ids)
```

### pepXML 格式

```python
# 載入 pepXML
protein_ids = []
peptide_ids = []

ms.PepXMLFile().load("results.pep.xml", protein_ids, peptide_ids)
```

## 讀取特徵資料

### featureXML

```python
# 載入特徵
feature_map = ms.FeatureMap()
ms.FeatureXMLFile().load("features.featureXML", feature_map)

# 存取特徵
for feature in feature_map:
    print(f"RT: {feature.getRT()}")
    print(f"MZ: {feature.getMZ()}")
    print(f"Intensity: {feature.getIntensity()}")
    print(f"Quality: {feature.getOverallQuality()}")
```

### consensusXML

```python
# 載入共識特徵
consensus_map = ms.ConsensusMap()
ms.ConsensusXMLFile().load("consensus.consensusXML", consensus_map)

# 存取共識特徵
for consensus_feature in consensus_map:
    print(f"RT: {consensus_feature.getRT()}")
    print(f"MZ: {consensus_feature.getMZ()}")

    # 取得特徵控制代碼（來自不同圖的子特徵）
    for handle in consensus_feature.getFeatureList():
        map_index = handle.getMapIndex()
        intensity = handle.getIntensity()
        print(f"  Map {map_index}: {intensity}")
```

## 讀取 FASTA 檔案

```python
# 載入蛋白質序列
fasta_entries = []
ms.FASTAFile().load("database.fasta", fasta_entries)

for entry in fasta_entries:
    print(f"Identifier: {entry.identifier}")
    print(f"Description: {entry.description}")
    print(f"Sequence: {entry.sequence}")
```

## 讀取 TraML 檔案

```python
# 載入標靶實驗的轉換列表
targeted_exp = ms.TargetedExperiment()
ms.TraMLFile().load("transitions.TraML", targeted_exp)

# 存取轉換
for transition in targeted_exp.getTransitions():
    print(f"Precursor MZ: {transition.getPrecursorMZ()}")
    print(f"Product MZ: {transition.getProductMZ()}")
```

## 寫入 mzTab 檔案

```python
# 建立 mzTab 用於報告
mztab = ms.MzTab()

# 添加中繼資料
metadata = mztab.getMetaData()
metadata.mz_tab_version.set("1.0.0")
metadata.title.set("Proteomics Analysis Results")

# 添加蛋白質資料
protein_section = mztab.getProteinSectionRows()
# ... 填入蛋白質資料 ...

# 寫入檔案
ms.MzTabFile().store("report.mzTab", mztab)
```

## 格式轉換

### mzXML 到 mzML

```python
# 讀取 mzXML
exp = ms.MSExperiment()
ms.MzXMLFile().load("data.mzXML", exp)

# 寫入為 mzML
ms.MzMLFile().store("data.mzML", exp)
```

### 從 mzML 提取層析圖

```python
# 載入實驗
exp = ms.MSExperiment()
ms.MzMLFile().load("data.mzML", exp)

# 提取特定層析圖
for chrom in exp.getChromatograms():
    if chrom.getNativeID() == "TIC":
        rt, intensity = chrom.get_peaks()
        print(f"TIC has {len(rt)} data points")
```

## 檔案中繼資料

### 存取 mzML 中繼資料

```python
# 載入檔案
exp = ms.MSExperiment()
ms.MzMLFile().load("sample.mzML", exp)

# 取得實驗設定
exp_settings = exp.getExperimentalSettings()

# 儀器資訊
instrument = exp_settings.getInstrument()
print(f"Instrument: {instrument.getName()}")
print(f"Model: {instrument.getModel()}")

# 樣本資訊
sample = exp_settings.getSample()
print(f"Sample name: {sample.getName()}")

# 來源檔案
for source_file in exp_settings.getSourceFiles():
    print(f"Source: {source_file.getNameOfFile()}")
```

## 最佳實務

### 記憶體管理

對於大型檔案：
1. 使用索引或串流存取而非完整記憶體內載入
2. 分塊處理資料
3. 不再需要時清除資料結構

```python
# 適合大型檔案
indexed_mzml = ms.IndexedMzMLFileLoader()
indexed_mzml.load("huge_file.mzML")

# 一次處理一個光譜
for i in range(indexed_mzml.getNrSpectra()):
    spec = indexed_mzml.getSpectrumById(i)
    # 處理光譜
    # 處理後光譜自動清理
```

### 錯誤處理

```python
try:
    exp = ms.MSExperiment()
    ms.MzMLFile().load("data.mzML", exp)
except Exception as e:
    print(f"Failed to load file: {e}")
```

### 檔案驗證

```python
# 檢查檔案是否存在且可讀取
import os

if os.path.exists("data.mzML") and os.path.isfile("data.mzML"):
    exp = ms.MSExperiment()
    ms.MzMLFile().load("data.mzML", exp)
else:
    print("File not found")
```
