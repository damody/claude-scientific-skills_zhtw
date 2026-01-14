# 肽段和蛋白質鑑定

## 概述

PyOpenMS 透過與搜尋引擎的整合支援肽段/蛋白質鑑定，並提供後處理鑑定結果的工具，包括 FDR 控制、蛋白質推斷和註釋。

## 支援的搜尋引擎

PyOpenMS 整合這些搜尋引擎：

- **Comet**：快速串聯質譜搜尋
- **Mascot**：商業搜尋引擎
- **MSGFPlus**：基於光譜機率的搜尋
- **XTandem**：開源搜尋工具
- **OMSSA**：NCBI 搜尋引擎
- **Myrimatch**：高通量搜尋
- **MSFragger**：超快速搜尋

## 讀取鑑定資料

### idXML 格式

```python
import pyopenms as ms

# 載入鑑定結果
protein_ids = []
peptide_ids = []

ms.IdXMLFile().load("identifications.idXML", protein_ids, peptide_ids)

print(f"Protein identifications: {len(protein_ids)}")
print(f"Peptide identifications: {len(peptide_ids)}")
```

### 存取肽段鑑定

```python
# 迭代肽段 ID
for peptide_id in peptide_ids:
    # 光譜中繼資料
    print(f"RT: {peptide_id.getRT():.2f}")
    print(f"m/z: {peptide_id.getMZ():.4f}")

    # 取得肽段命中（按分數排序）
    hits = peptide_id.getHits()
    print(f"Number of hits: {len(hits)}")

    for hit in hits:
        sequence = hit.getSequence()
        print(f"  Sequence: {sequence.toString()}")
        print(f"  Score: {hit.getScore()}")
        print(f"  Charge: {hit.getCharge()}")
        print(f"  Mass error (ppm): {hit.getMetaValue('mass_error_ppm')}")

        # 取得修飾
        if sequence.isModified():
            for i in range(sequence.size()):
                residue = sequence.getResidue(i)
                if residue.isModified():
                    print(f"    Modification at position {i}: {residue.getModificationName()}")
```

### 存取蛋白質鑑定

```python
# 存取蛋白質層級資訊
for protein_id in protein_ids:
    # 搜尋參數
    search_params = protein_id.getSearchParameters()
    print(f"Search engine: {protein_id.getSearchEngine()}")
    print(f"Database: {search_params.db}")

    # 蛋白質命中
    hits = protein_id.getHits()
    for hit in hits:
        print(f"  Accession: {hit.getAccession()}")
        print(f"  Score: {hit.getScore()}")
        print(f"  Coverage: {hit.getCoverage()}")
        print(f"  Sequence: {hit.getSequence()}")
```

## 偽陽性率（FDR）

### FDR 過濾

應用 FDR 過濾以控制偽陽性：

```python
# 建立 FDR 物件
fdr = ms.FalseDiscoveryRate()

# 在 PSM 層級應用 FDR
fdr.apply(peptide_ids)

# 按 FDR 閾值過濾
fdr_threshold = 0.01  # 1% FDR
filtered_peptide_ids = []

for peptide_id in peptide_ids:
    # 保留低於 FDR 閾值的命中
    filtered_hits = []
    for hit in peptide_id.getHits():
        if hit.getScore() <= fdr_threshold:  # 分數越低越好
            filtered_hits.append(hit)

    if filtered_hits:
        peptide_id.setHits(filtered_hits)
        filtered_peptide_ids.append(peptide_id)

print(f"Peptides passing FDR: {len(filtered_peptide_ids)}")
```

### 分數轉換

將分數轉換為 q 值：

```python
# 應用分數轉換
fdr.apply(peptide_ids)

# 存取 q 值
for peptide_id in peptide_ids:
    for hit in peptide_id.getHits():
        q_value = hit.getMetaValue("q-value")
        print(f"Sequence: {hit.getSequence().toString()}, q-value: {q_value}")
```

## 蛋白質推斷

### ID 映射器

將肽段鑑定映射到蛋白質：

```python
# 建立映射器
mapper = ms.IDMapper()

# 映射到特徵
feature_map = ms.FeatureMap()
ms.FeatureXMLFile().load("features.featureXML", feature_map)

# 使用 ID 註釋特徵
mapper.annotate(feature_map, peptide_ids, protein_ids)

# 檢查註釋的特徵
for feature in feature_map:
    pep_ids = feature.getPeptideIdentifications()
    if pep_ids:
        for pep_id in pep_ids:
            for hit in pep_id.getHits():
                print(f"Feature {feature.getMZ():.4f}: {hit.getSequence().toString()}")
```

### 蛋白質分組

按共享肽段分組蛋白質：

```python
# 建立蛋白質推斷演算法
inference = ms.BasicProteinInferenceAlgorithm()

# 執行推斷
inference.run(peptide_ids, protein_ids)

# 存取蛋白質群組
for protein_id in protein_ids:
    hits = protein_id.getHits()
    if len(hits) > 1:
        print("Protein group:")
        for hit in hits:
            print(f"  {hit.getAccession()}")
```

## 肽段序列處理

### AASequence 物件

處理肽段序列：

```python
# 建立肽段序列
seq = ms.AASequence.fromString("PEPTIDE")

print(f"Sequence: {seq.toString()}")
print(f"Monoisotopic mass: {seq.getMonoWeight():.4f}")
print(f"Average mass: {seq.getAverageWeight():.4f}")
print(f"Length: {seq.size()}")

# 存取個別胺基酸
for i in range(seq.size()):
    residue = seq.getResidue(i)
    print(f"Position {i}: {residue.getOneLetterCode()}, mass: {residue.getMonoWeight():.4f}")
```

### 修飾序列

處理轉譯後修飾：

```python
# 帶修飾的序列
mod_seq = ms.AASequence.fromString("PEPTIDEM(Oxidation)K")

print(f"Modified sequence: {mod_seq.toString()}")
print(f"Mass with mods: {mod_seq.getMonoWeight():.4f}")

# 檢查是否修飾
print(f"Is modified: {mod_seq.isModified()}")

# 取得修飾資訊
for i in range(mod_seq.size()):
    residue = mod_seq.getResidue(i)
    if residue.isModified():
        print(f"Residue {residue.getOneLetterCode()} at position {i}")
        print(f"  Modification: {residue.getModificationName()}")
```

### 肽段消化

模擬酵素消化：

```python
# 建立消化酵素
enzyme = ms.ProteaseDigestion()
enzyme.setEnzyme("Trypsin")

# 設定漏切位點
enzyme.setMissedCleavages(2)

# 消化蛋白質序列
protein_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

# 取得肽段
peptides = []
enzyme.digest(ms.AASequence.fromString(protein_seq), peptides)

print(f"Generated {len(peptides)} peptides")
for peptide in peptides[:5]:  # 顯示前 5 個
    print(f"  {peptide.toString()}, mass: {peptide.getMonoWeight():.2f}")
```

## 理論光譜生成

### 碎片離子計算

生成理論碎片離子：

```python
# 建立肽段
peptide = ms.AASequence.fromString("PEPTIDE")

# 生成 b 和 y 離子
fragments = []
ms.TheoreticalSpectrumGenerator().getSpectrum(fragments, peptide, 1, 1)

print(f"Generated {fragments.size()} fragment ions")

# 存取碎片
mz, intensity = fragments.get_peaks()
for m, i in zip(mz[:10], intensity[:10]):  # 顯示前 10 個
    print(f"m/z: {m:.4f}, intensity: {i}")
```

## 完整鑑定工作流程

### 端到端範例

```python
import pyopenms as ms

def identification_workflow(spectrum_file, fasta_file, output_file):
    """
    具有 FDR 控制的完整鑑定工作流程。

    Args:
        spectrum_file: 輸入 mzML 檔案
        fasta_file: 蛋白質資料庫（FASTA）
        output_file: 輸出 idXML 檔案
    """

    # 步驟 1：載入光譜
    exp = ms.MSExperiment()
    ms.MzMLFile().load(spectrum_file, exp)
    print(f"Loaded {exp.getNrSpectra()} spectra")

    # 步驟 2：設定搜尋參數
    search_params = ms.SearchParameters()
    search_params.db = fasta_file
    search_params.precursor_mass_tolerance = 10.0  # ppm
    search_params.fragment_mass_tolerance = 0.5  # Da
    search_params.enzyme = "Trypsin"
    search_params.missed_cleavages = 2
    search_params.modifications = ["Oxidation (M)", "Carbamidomethyl (C)"]

    # 步驟 3：執行搜尋（Comet 適配器範例）
    # 注意：需要安裝搜尋引擎
    # comet = ms.CometAdapter()
    # protein_ids, peptide_ids = comet.search(exp, search_params)

    # 此範例載入預先計算的結果
    protein_ids = []
    peptide_ids = []
    ms.IdXMLFile().load("raw_identifications.idXML", protein_ids, peptide_ids)

    print(f"Initial peptide IDs: {len(peptide_ids)}")

    # 步驟 4：應用 FDR 過濾
    fdr = ms.FalseDiscoveryRate()
    fdr.apply(peptide_ids)

    # 按 1% FDR 過濾
    filtered_peptide_ids = []
    for peptide_id in peptide_ids:
        filtered_hits = []
        for hit in peptide_id.getHits():
            q_value = hit.getMetaValue("q-value")
            if q_value <= 0.01:
                filtered_hits.append(hit)

        if filtered_hits:
            peptide_id.setHits(filtered_hits)
            filtered_peptide_ids.append(peptide_id)

    print(f"Peptides after FDR (1%): {len(filtered_peptide_ids)}")

    # 步驟 5：蛋白質推斷
    inference = ms.BasicProteinInferenceAlgorithm()
    inference.run(filtered_peptide_ids, protein_ids)

    print(f"Identified proteins: {len(protein_ids)}")

    # 步驟 6：儲存結果
    ms.IdXMLFile().store(output_file, protein_ids, filtered_peptide_ids)
    print(f"Results saved to {output_file}")

    return protein_ids, filtered_peptide_ids

# 執行工作流程
protein_ids, peptide_ids = identification_workflow(
    "spectra.mzML",
    "database.fasta",
    "identifications_fdr.idXML"
)
```

## 光譜庫搜尋

### 庫匹配

```python
# 載入光譜庫
library = ms.MSPFile()
library_spectra = []
library.load("spectral_library.msp", library_spectra)

# 載入實驗光譜
exp = ms.MSExperiment()
ms.MzMLFile().load("data.mzML", exp)

# 比較光譜
spectra_compare = ms.SpectraSTSimilarityScore()

for exp_spec in exp:
    if exp_spec.getMSLevel() == 2:
        best_match_score = 0
        best_match_lib = None

        for lib_spec in library_spectra:
            score = spectra_compare.operator()(exp_spec, lib_spec)
            if score > best_match_score:
                best_match_score = score
                best_match_lib = lib_spec

        if best_match_score > 0.7:  # 閾值
            print(f"Match found: score {best_match_score:.3f}")
```

## 最佳實務

### 誘餌資料庫

使用靶標-誘餌方法計算 FDR：

```python
# 生成誘餌資料庫
decoy_generator = ms.DecoyGenerator()

# 載入靶標資料庫
fasta_entries = []
ms.FASTAFile().load("target.fasta", fasta_entries)

# 生成誘餌
decoy_entries = []
for entry in fasta_entries:
    decoy_entry = decoy_generator.reverseProtein(entry)
    decoy_entries.append(decoy_entry)

# 儲存組合資料庫
all_entries = fasta_entries + decoy_entries
ms.FASTAFile().store("target_decoy.fasta", all_entries)
```

### 分數解讀

了解不同引擎的分數類型：

```python
# 根據搜尋引擎解讀分數
for peptide_id in peptide_ids:
    search_engine = peptide_id.getIdentifier()

    for hit in peptide_id.getHits():
        score = hit.getScore()

        # 分數解讀因引擎而異
        if "Comet" in search_engine:
            # Comet：E 值越高越差
            print(f"E-value: {score}")
        elif "Mascot" in search_engine:
            # Mascot：分數越高越好
            print(f"Ion score: {score}")
```
