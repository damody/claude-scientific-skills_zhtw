# Matchms 常見工作流程

本文件提供使用 matchms 進行常見質譜分析工作流程的詳細範例。

## 工作流程 1：基本質譜庫匹配

將未知質譜與參考庫匹配以識別化合物。

```python
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities
from matchms.filtering import select_by_relative_intensity, require_minimum_number_of_peaks
from matchms import calculate_scores
from matchms.similarity import CosineGreedy

# 載入參考庫
print("Loading reference library...")
library = list(load_from_mgf("reference_library.mgf"))

# 載入查詢質譜（未知物）
print("Loading query spectra...")
queries = list(load_from_mgf("unknown_spectra.mgf"))

# 處理庫質譜
print("Processing library...")
processed_library = []
for spectrum in library:
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    if spectrum is not None:
        processed_library.append(spectrum)

# 處理查詢質譜
print("Processing queries...")
processed_queries = []
for spectrum in queries:
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    if spectrum is not None:
        processed_queries.append(spectrum)

# 計算相似性
print("Calculating similarities...")
scores = calculate_scores(references=processed_library,
                         queries=processed_queries,
                         similarity_function=CosineGreedy(tolerance=0.1))

# 獲取每個查詢的最佳匹配
print("\nTop matches:")
for i, query in enumerate(processed_queries):
    top_matches = scores.scores_by_query(query, sort=True)[:5]

    query_name = query.get("compound_name", f"Query {i}")
    print(f"\n{query_name}:")

    for ref_idx, score in top_matches:
        ref_spectrum = processed_library[ref_idx]
        ref_name = ref_spectrum.get("compound_name", f"Ref {ref_idx}")
        print(f"  {ref_name}: {score:.4f}")
```

---

## 工作流程 2：品質控制和資料清理

在分析前過濾和清理質譜資料。

```python
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import (default_filters, normalize_intensities,
                               require_precursor_mz, require_minimum_number_of_peaks,
                               require_minimum_number_of_high_peaks,
                               select_by_relative_intensity, remove_peaks_around_precursor_mz)

# 載入質譜
spectra = list(load_from_mgf("raw_data.mgf"))
print(f"Loaded {len(spectra)} raw spectra")

# 應用品質過濾器
cleaned_spectra = []
for spectrum in spectra:
    # 協調元資料
    spectrum = default_filters(spectrum)

    # 品質要求
    spectrum = require_precursor_mz(spectrum, minimum_accepted_mz=50.0)
    if spectrum is None:
        continue

    spectrum = require_minimum_number_of_peaks(spectrum, n_required=10)
    if spectrum is None:
        continue

    # 清理峰值
    spectrum = normalize_intensities(spectrum)
    spectrum = remove_peaks_around_precursor_mz(spectrum, mz_tolerance=17)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)

    # 要求高品質峰值
    spectrum = require_minimum_number_of_high_peaks(spectrum,
                                                     n_required=5,
                                                     intensity_threshold=0.05)
    if spectrum is None:
        continue

    cleaned_spectra.append(spectrum)

print(f"Retained {len(cleaned_spectra)} high-quality spectra")
print(f"Removed {len(spectra) - len(cleaned_spectra)} low-quality spectra")

# 儲存清理後的資料
save_as_mgf(cleaned_spectra, "cleaned_data.mgf")
```

---

## 工作流程 3：多指標相似性評分

結合多個相似性指標以實現穩健的化合物識別。

```python
from matchms.importing import load_from_mgf
from matchms.filtering import (default_filters, normalize_intensities,
                               derive_inchi_from_smiles, add_fingerprint, add_losses)
from matchms import calculate_scores
from matchms.similarity import (CosineGreedy, ModifiedCosine,
                                NeutralLossesCosine, FingerprintSimilarity)
import numpy as np

# 載入質譜
library = list(load_from_mgf("library.mgf"))
queries = list(load_from_mgf("queries.mgf"))

# 處理多重特徵
def process_for_multimetric(spectrum):
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)

    # 添加化學指紋
    spectrum = derive_inchi_from_smiles(spectrum)
    spectrum = add_fingerprint(spectrum, fingerprint_type="morgan2", nbits=2048)

    # 添加中性丟失
    spectrum = add_losses(spectrum, loss_mz_from=5.0, loss_mz_to=200.0)

    return spectrum

processed_library = [process_for_multimetric(s) for s in library if s is not None]
processed_queries = [process_for_multimetric(s) for s in queries if s is not None]

# 計算多個相似性分數
print("Calculating Cosine similarity...")
cosine_scores = calculate_scores(processed_library, processed_queries,
                                 CosineGreedy(tolerance=0.1))

print("Calculating Modified Cosine similarity...")
modified_cosine_scores = calculate_scores(processed_library, processed_queries,
                                         ModifiedCosine(tolerance=0.1))

print("Calculating Neutral Losses similarity...")
neutral_losses_scores = calculate_scores(processed_library, processed_queries,
                                        NeutralLossesCosine(tolerance=0.1))

print("Calculating Fingerprint similarity...")
fingerprint_scores = calculate_scores(processed_library, processed_queries,
                                      FingerprintSimilarity(similarity_measure="jaccard"))

# 用權重組合分數
weights = {
    'cosine': 0.4,
    'modified_cosine': 0.3,
    'neutral_losses': 0.2,
    'fingerprint': 0.1
}

# 獲取每個查詢的組合分數
for i, query in enumerate(processed_queries):
    query_name = query.get("compound_name", f"Query {i}")

    combined_scores = []
    for j, ref in enumerate(processed_library):
        combined = (weights['cosine'] * cosine_scores.scores[j, i] +
                   weights['modified_cosine'] * modified_cosine_scores.scores[j, i] +
                   weights['neutral_losses'] * neutral_losses_scores.scores[j, i] +
                   weights['fingerprint'] * fingerprint_scores.scores[j, i])
        combined_scores.append((j, combined))

    # 按組合分數排序
    combined_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{query_name} - Top 3 matches:")
    for ref_idx, score in combined_scores[:3]:
        ref_name = processed_library[ref_idx].get("compound_name", f"Ref {ref_idx}")
        print(f"  {ref_name}: {score:.4f}")
```

---

## 工作流程 4：前驅離子過濾庫搜尋

在質譜匹配前按前驅離子質量預過濾以加快搜尋速度。

```python
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities
from matchms import calculate_scores
from matchms.similarity import PrecursorMzMatch, CosineGreedy
import numpy as np

# 載入資料
library = list(load_from_mgf("large_library.mgf"))
queries = list(load_from_mgf("queries.mgf"))

# 處理質譜
processed_library = [normalize_intensities(default_filters(s)) for s in library]
processed_queries = [normalize_intensities(default_filters(s)) for s in queries]

# 步驟 1：快速前驅離子質量過濾
print("Filtering by precursor mass...")
mass_filter = calculate_scores(processed_library, processed_queries,
                               PrecursorMzMatch(tolerance=0.1, tolerance_type="Dalton"))

# 步驟 2：只對匹配的前驅離子計算餘弦
print("Calculating cosine similarity for filtered candidates...")
cosine_scores = calculate_scores(processed_library, processed_queries,
                                CosineGreedy(tolerance=0.1))

# 步驟 3：將質量過濾器應用於餘弦分數
for i, query in enumerate(processed_queries):
    candidates = []

    for j, ref in enumerate(processed_library):
        # 只考慮前驅離子匹配的
        if mass_filter.scores[j, i] > 0:
            cosine_score = cosine_scores.scores[j, i]
            candidates.append((j, cosine_score))

    # 按餘弦分數排序
    candidates.sort(key=lambda x: x[1], reverse=True)

    query_name = query.get("compound_name", f"Query {i}")
    print(f"\n{query_name} - Top 5 matches (from {len(candidates)} candidates):")

    for ref_idx, score in candidates[:5]:
        ref_name = processed_library[ref_idx].get("compound_name", f"Ref {ref_idx}")
        ref_mz = processed_library[ref_idx].get("precursor_mz", "N/A")
        print(f"  {ref_name} (m/z {ref_mz}): {score:.4f}")
```

---

## 工作流程 5：建立可重用的處理流程

建立標準化流程以實現一致的處理。

```python
from matchms import SpectrumProcessor
from matchms.filtering import (default_filters, normalize_intensities,
                               select_by_relative_intensity,
                               remove_peaks_around_precursor_mz,
                               require_minimum_number_of_peaks,
                               derive_inchi_from_smiles, add_fingerprint)
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_pickle

# 定義自訂處理流程
def create_standard_pipeline():
    """建立可重用的處理流程"""
    return SpectrumProcessor([
        default_filters,
        normalize_intensities,
        lambda s: remove_peaks_around_precursor_mz(s, mz_tolerance=17),
        lambda s: select_by_relative_intensity(s, intensity_from=0.01),
        lambda s: require_minimum_number_of_peaks(s, n_required=5),
        derive_inchi_from_smiles,
        lambda s: add_fingerprint(s, fingerprint_type="morgan2")
    ])

# 建立流程實例
pipeline = create_standard_pipeline()

# 使用相同流程處理多個資料集
datasets = ["dataset1.mgf", "dataset2.mgf", "dataset3.mgf"]

for dataset_file in datasets:
    print(f"\nProcessing {dataset_file}...")

    # 載入質譜
    spectra = list(load_from_mgf(dataset_file))

    # 應用流程
    processed = []
    for spectrum in spectra:
        result = pipeline(spectrum)
        if result is not None:
            processed.append(result)

    print(f"  Loaded: {len(spectra)}")
    print(f"  Processed: {len(processed)}")

    # 儲存處理後的資料
    output_file = dataset_file.replace(".mgf", "_processed.pkl")
    save_as_pickle(processed, output_file)
    print(f"  Saved to: {output_file}")
```

---

## 工作流程 6：格式轉換和標準化

在不同質譜檔案格式之間轉換。

```python
from matchms.importing import load_from_mzml, load_from_mgf
from matchms.exporting import save_as_mgf, save_as_msp, save_as_json
from matchms.filtering import default_filters, normalize_intensities

def convert_and_standardize(input_file, output_format="mgf"):
    """
    載入、標準化和轉換質譜資料

    參數：
    -----------
    input_file : str
        輸入檔案路徑（支援 .mzML、.mzXML、.mgf）
    output_format : str
        輸出格式（'mgf'、'msp' 或 'json'）
    """
    # 確定輸入格式並載入
    if input_file.endswith('.mzML') or input_file.endswith('.mzXML'):
        from matchms.importing import load_from_mzml
        spectra = list(load_from_mzml(input_file, ms_level=2))
    elif input_file.endswith('.mgf'):
        spectra = list(load_from_mgf(input_file))
    else:
        raise ValueError(f"Unsupported format: {input_file}")

    print(f"Loaded {len(spectra)} spectra from {input_file}")

    # 標準化
    processed = []
    for spectrum in spectra:
        spectrum = default_filters(spectrum)
        spectrum = normalize_intensities(spectrum)
        if spectrum is not None:
            processed.append(spectrum)

    print(f"Standardized {len(processed)} spectra")

    # 匯出
    output_file = input_file.rsplit('.', 1)[0] + f'_standardized.{output_format}'

    if output_format == 'mgf':
        save_as_mgf(processed, output_file)
    elif output_format == 'msp':
        save_as_msp(processed, output_file)
    elif output_format == 'json':
        save_as_json(processed, output_file)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Saved to {output_file}")
    return processed

# 將 mzML 轉換為 MGF
convert_and_standardize("raw_data.mzML", output_format="mgf")

# 將 MGF 轉換為 MSP 庫格式
convert_and_standardize("library.mgf", output_format="msp")
```

---

## 工作流程 7：元資料豐富和驗證

使用化學結構資訊豐富質譜並驗證註釋。

```python
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import (default_filters, derive_inchi_from_smiles,
                               derive_inchikey_from_inchi, derive_smiles_from_inchi,
                               add_fingerprint, repair_not_matching_annotation,
                               require_valid_annotation)

# 載入質譜
spectra = list(load_from_mgf("spectra.mgf"))

# 豐富和驗證
enriched_spectra = []
validation_failures = []

for i, spectrum in enumerate(spectra):
    # 基本協調
    spectrum = default_filters(spectrum)

    # 衍生化學結構
    spectrum = derive_inchi_from_smiles(spectrum)
    spectrum = derive_inchikey_from_inchi(spectrum)
    spectrum = derive_smiles_from_inchi(spectrum)

    # 修復不匹配
    spectrum = repair_not_matching_annotation(spectrum)

    # 添加分子指紋
    spectrum = add_fingerprint(spectrum, fingerprint_type="morgan2", nbits=2048)

    # 驗證
    validated = require_valid_annotation(spectrum)

    if validated is not None:
        enriched_spectra.append(validated)
    else:
        validation_failures.append(i)

print(f"Successfully enriched: {len(enriched_spectra)}")
print(f"Validation failures: {len(validation_failures)}")

# 儲存豐富後的資料
save_as_mgf(enriched_spectra, "enriched_spectra.mgf")

# 報告失敗
if validation_failures:
    print("\nSpectra that failed validation:")
    for idx in validation_failures[:10]:  # 顯示前 10 個
        original = spectra[idx]
        name = original.get("compound_name", f"Spectrum {idx}")
        print(f"  - {name}")
```

---

## 工作流程 8：大規模庫比較

高效比較兩個大型質譜庫。

```python
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
import numpy as np

# 載入兩個庫
print("Loading libraries...")
library1 = list(load_from_mgf("library1.mgf"))
library2 = list(load_from_mgf("library2.mgf"))

# 處理
processed_lib1 = [normalize_intensities(default_filters(s)) for s in library1]
processed_lib2 = [normalize_intensities(default_filters(s)) for s in library2]

# 計算全對全相似性
print("Calculating similarities...")
scores = calculate_scores(processed_lib1, processed_lib2,
                         CosineGreedy(tolerance=0.1))

# 尋找高相似性配對（潛在重複或相似化合物）
threshold = 0.8
similar_pairs = []

for i, spec1 in enumerate(processed_lib1):
    for j, spec2 in enumerate(processed_lib2):
        score = scores.scores[i, j]
        if score >= threshold:
            similar_pairs.append({
                'lib1_idx': i,
                'lib2_idx': j,
                'lib1_name': spec1.get("compound_name", f"L1_{i}"),
                'lib2_name': spec2.get("compound_name", f"L2_{j}"),
                'similarity': score
            })

# 按相似性排序
similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

print(f"\nFound {len(similar_pairs)} pairs with similarity >= {threshold}")
print("\nTop 10 most similar pairs:")
for pair in similar_pairs[:10]:
    print(f"{pair['lib1_name']} <-> {pair['lib2_name']}: {pair['similarity']:.4f}")

# 匯出為 CSV
import pandas as pd
df = pd.DataFrame(similar_pairs)
df.to_csv("library_comparison.csv", index=False)
print("\nFull results saved to library_comparison.csv")
```

---

## 工作流程 9：離子模式特定處理

分別處理正離子模式和負離子模式質譜。

```python
from matchms.importing import load_from_mgf
from matchms.filtering import (default_filters, normalize_intensities,
                               require_correct_ionmode, derive_ionmode)
from matchms.exporting import save_as_mgf

# 載入混合模式質譜
spectra = list(load_from_mgf("mixed_modes.mgf"))

# 按離子模式分離
positive_spectra = []
negative_spectra = []
unknown_mode = []

for spectrum in spectra:
    # 協調並衍生離子模式
    spectrum = default_filters(spectrum)
    spectrum = derive_ionmode(spectrum)

    # 按模式分離
    ionmode = spectrum.get("ionmode")

    if ionmode == "positive":
        spectrum = normalize_intensities(spectrum)
        positive_spectra.append(spectrum)
    elif ionmode == "negative":
        spectrum = normalize_intensities(spectrum)
        negative_spectra.append(spectrum)
    else:
        unknown_mode.append(spectrum)

print(f"Positive mode: {len(positive_spectra)}")
print(f"Negative mode: {len(negative_spectra)}")
print(f"Unknown mode: {len(unknown_mode)}")

# 儲存分離的資料
save_as_mgf(positive_spectra, "positive_mode.mgf")
save_as_mgf(negative_spectra, "negative_mode.mgf")

# 處理模式特定分析
from matchms import calculate_scores
from matchms.similarity import CosineGreedy

if len(positive_spectra) > 1:
    print("\nCalculating positive mode similarities...")
    pos_scores = calculate_scores(positive_spectra, positive_spectra,
                                  CosineGreedy(tolerance=0.1))

if len(negative_spectra) > 1:
    print("Calculating negative mode similarities...")
    neg_scores = calculate_scores(negative_spectra, negative_spectra,
                                  CosineGreedy(tolerance=0.1))
```

---

## 工作流程 10：自動化合物識別報告

生成詳細的化合物識別報告。

```python
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine
import pandas as pd

def identify_compounds(query_file, library_file, output_csv="identification_report.csv"):
    """
    帶有詳細報告的自動化合物識別
    """
    # 載入資料
    print("Loading data...")
    queries = list(load_from_mgf(query_file))
    library = list(load_from_mgf(library_file))

    # 處理
    proc_queries = [normalize_intensities(default_filters(s)) for s in queries]
    proc_library = [normalize_intensities(default_filters(s)) for s in library]

    # 計算相似性
    print("Calculating similarities...")
    cosine_scores = calculate_scores(proc_library, proc_queries, CosineGreedy())
    modified_scores = calculate_scores(proc_library, proc_queries, ModifiedCosine())

    # 生成報告
    results = []
    for i, query in enumerate(proc_queries):
        query_name = query.get("compound_name", f"Unknown_{i}")
        query_mz = query.get("precursor_mz", "N/A")

        # 獲取前 5 個匹配
        cosine_matches = cosine_scores.scores_by_query(query, sort=True)[:5]

        for rank, (lib_idx, cos_score) in enumerate(cosine_matches, 1):
            ref = proc_library[lib_idx]
            mod_score = modified_scores.scores[lib_idx, i]

            results.append({
                'Query': query_name,
                'Query_mz': query_mz,
                'Rank': rank,
                'Match': ref.get("compound_name", f"Ref_{lib_idx}"),
                'Match_mz': ref.get("precursor_mz", "N/A"),
                'Cosine_Score': cos_score,
                'Modified_Cosine': mod_score,
                'InChIKey': ref.get("inchikey", "N/A")
            })

    # 建立 DataFrame 並儲存
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nReport saved to {output_csv}")

    # 摘要統計
    print("\nSummary:")
    high_confidence = len(df[df['Cosine_Score'] >= 0.8])
    medium_confidence = len(df[(df['Cosine_Score'] >= 0.6) & (df['Cosine_Score'] < 0.8)])
    low_confidence = len(df[df['Cosine_Score'] < 0.6])

    print(f"  High confidence (≥0.8): {high_confidence}")
    print(f"  Medium confidence (0.6-0.8): {medium_confidence}")
    print(f"  Low confidence (<0.6): {low_confidence}")

    return df

# 執行識別
report = identify_compounds("unknowns.mgf", "reference_library.mgf")
```

---

## 最佳實踐

1. **始終處理查詢和參考**：對兩者應用相同的過濾器以確保一致的比較
2. **儲存中間結果**：使用 pickle 格式快速重新載入處理後的質譜
3. **監控記憶體使用**：對大型檔案使用生成器而非一次全部載入
4. **驗證資料品質**：在相似性計算前應用品質過濾器
5. **選擇適當的相似性指標**：CosineGreedy 求速度，ModifiedCosine 用於相關化合物
6. **結合多個指標**：使用多個相似性分數以實現穩健識別
7. **首先按前驅離子質量過濾**：大幅加速大型庫搜尋
8. **記錄您的流程**：儲存處理參數以實現可重現性

## 進一步資源

- matchms 文件：https://matchms.readthedocs.io
- GNPS 平台：https://gnps.ucsd.edu
- matchms GitHub：https://github.com/matchms/matchms

