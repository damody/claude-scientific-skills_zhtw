# Matchms 相似性函數參考

本文件提供 matchms 中所有可用相似性評分方法的詳細資訊。

## 概述

Matchms 提供多種相似性函數用於比較質譜。使用 `calculate_scores()` 計算參考質譜和查詢質譜集合之間的成對相似性。

```python
from matchms import calculate_scores
from matchms.similarity import CosineGreedy

scores = calculate_scores(references=library_spectra,
                         queries=query_spectra,
                         similarity_function=CosineGreedy())
```

## 基於峰值的相似性函數

這些函數根據峰值模式（m/z 和強度值）比較質譜。

### CosineGreedy

**描述**：使用快速貪婪匹配演算法計算兩個質譜之間的餘弦相似性。峰值在指定容差範圍內匹配，相似性基於匹配峰值強度計算。

**使用時機**：
- 大型資料集的快速相似性計算
- 通用質譜匹配
- 當速度優先於數學上最優匹配時

**參數**：
- `tolerance` (float, 預設=0.1)：峰值匹配的最大 m/z 差異（道爾頓）
- `mz_power` (float, 預設=0.0)：m/z 加權指數（0 = 無加權）
- `intensity_power` (float, 預設=1.0)：強度加權指數

**範例**：
```python
from matchms.similarity import CosineGreedy

similarity_func = CosineGreedy(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
scores = calculate_scores(references, queries, similarity_func)
```

**輸出**：0.0 到 1.0 之間的相似性分數，以及匹配峰值數量。

---

### CosineHungarian

**描述**：使用匈牙利演算法計算餘弦相似性以實現最優峰值匹配。提供數學上最優的峰值分配，但比 CosineGreedy 慢。

**使用時機**：
- 當需要最優峰值匹配時
- 高品質參考庫比較
- 需要可重現、數學嚴謹結果的研究

**參數**：
- `tolerance` (float, 預設=0.1)：峰值匹配的最大 m/z 差異
- `mz_power` (float, 預設=0.0)：m/z 加權指數
- `intensity_power` (float, 預設=1.0)：強度加權指數

**範例**：
```python
from matchms.similarity import CosineHungarian

similarity_func = CosineHungarian(tolerance=0.1)
scores = calculate_scores(references, queries, similarity_func)
```

**輸出**：0.0 到 1.0 之間的最優相似性分數，以及匹配峰值。

**注意**：比 CosineGreedy 慢；用於較小的資料集或當準確性至關重要時。

---

### ModifiedCosine

**描述**：通過考慮前驅離子 m/z 差異來擴展餘弦相似性。允許峰值在應用基於前驅離子質量差異的質量偏移後匹配。適用於比較相關化合物（同位素、加合物、類似物）的質譜。

**使用時機**：
- 比較來自不同前驅離子質量的質譜
- 識別結構類似物或衍生物
- 跨離子化模式比較
- 當前驅離子質量差異有意義時

**參數**：
- `tolerance` (float, 預設=0.1)：偏移後峰值匹配的最大 m/z 差異
- `mz_power` (float, 預設=0.0)：m/z 加權指數
- `intensity_power` (float, 預設=1.0)：強度加權指數

**範例**：
```python
from matchms.similarity import ModifiedCosine

similarity_func = ModifiedCosine(tolerance=0.1)
scores = calculate_scores(references, queries, similarity_func)
```

**要求**：兩個質譜都必須有有效的 precursor_mz 元資料。

---

### NeutralLossesCosine

**描述**：基於中性丟失模式而非碎片 m/z 值計算相似性。中性丟失通過從前驅離子 m/z 減去碎片 m/z 得出。特別適用於識別具有相似碎裂模式的化合物。

**使用時機**：
- 跨不同前驅離子質量比較碎裂模式
- 識別具有相似中性丟失譜的化合物
- 作為常規餘弦評分的補充
- 代謝物識別和分類

**參數**：
- `tolerance` (float, 預設=0.1)：匹配的最大中性丟失差異
- `mz_power` (float, 預設=0.0)：損失值加權指數
- `intensity_power` (float, 預設=1.0)：強度加權指數

**範例**：
```python
from matchms.similarity import NeutralLossesCosine
from matchms.filtering import add_losses

# 首先向質譜添加損失
spectra_with_losses = [add_losses(s) for s in spectra]

similarity_func = NeutralLossesCosine(tolerance=0.1)
scores = calculate_scores(references_with_losses, queries_with_losses, similarity_func)
```

**要求**：
- 兩個質譜都必須有有效的 precursor_mz 元資料
- 在評分前使用 `add_losses()` 過濾器計算中性丟失

---

## 結構相似性函數

這些函數比較分子結構而非質譜峰值。

### FingerprintSimilarity

**描述**：計算從化學結構（SMILES 或 InChI）衍生的分子指紋之間的相似性。支援多種指紋類型和相似性指標。

**使用時機**：
- 沒有質譜資料的結構相似性
- 結合結構和質譜相似性
- 在質譜匹配前預過濾候選物
- 結構-活性關係研究

**參數**：
- `fingerprint_type` (str, 預設="daylight")：指紋類型
  - `"daylight"`：Daylight 指紋
  - `"morgan1"`、`"morgan2"`、`"morgan3"`：半徑 1、2 或 3 的 Morgan 指紋
- `similarity_measure` (str, 預設="jaccard")：相似性指標
  - `"jaccard"`：Jaccard 指數（交集 / 聯集）
  - `"dice"`：Dice 係數（2 * 交集 / (大小1 + 大小2)）
  - `"cosine"`：餘弦相似性

**範例**：
```python
from matchms.similarity import FingerprintSimilarity
from matchms.filtering import add_fingerprint

# 向質譜添加指紋
spectra_with_fps = [add_fingerprint(s, fingerprint_type="morgan2", nbits=2048)
                    for s in spectra]

similarity_func = FingerprintSimilarity(similarity_measure="jaccard")
scores = calculate_scores(references_with_fps, queries_with_fps, similarity_func)
```

**要求**：
- 質譜必須有有效的 SMILES 或 InChI 元資料
- 使用 `add_fingerprint()` 過濾器計算指紋
- 需要 rdkit 庫

---

## 基於元資料的相似性函數

這些函數比較元資料欄位而非質譜或結構資料。

### MetadataMatch

**描述**：比較質譜之間使用者定義的元資料欄位。支援分類資料的精確匹配和數值資料的容差匹配。

**使用時機**：
- 按實驗條件過濾（碰撞能量、滯留時間）
- 儀器特定匹配
- 結合元資料限制和質譜相似性
- 自訂元資料過濾

**參數**：
- `field` (str)：要比較的元資料欄位名稱
- `matching_type` (str, 預設="exact")：匹配方法
  - `"exact"`：精確字串/值匹配
  - `"difference"`：數值的絕對差異
  - `"relative_difference"`：數值的相對差異
- `tolerance` (float, 可選)：數值匹配的最大差異

**範例（精確匹配）**：
```python
from matchms.similarity import MetadataMatch

# 按儀器類型匹配
similarity_func = MetadataMatch(field="instrument_type", matching_type="exact")
scores = calculate_scores(references, queries, similarity_func)
```

**範例（數值匹配）**：
```python
# 在 0.5 分鐘內匹配滯留時間
similarity_func = MetadataMatch(field="retention_time",
                                matching_type="difference",
                                tolerance=0.5)
scores = calculate_scores(references, queries, similarity_func)
```

**輸出**：精確匹配返回 1.0（匹配）或 0.0（不匹配）。數值匹配返回基於差異的相似性分數。

---

### PrecursorMzMatch

**描述**：基於前驅離子 m/z 值的二元匹配。根據前驅離子質量是否在指定容差範圍內匹配返回 True/False。

**使用時機**：
- 按前驅離子質量預過濾質譜庫
- 快速基於質量的候選選擇
- 與其他相似性指標結合
- 同量異構化合物識別

**參數**：
- `tolerance` (float, 預設=0.1)：匹配的最大 m/z 差異
- `tolerance_type` (str, 預設="Dalton")：容差單位
  - `"Dalton"`：絕對質量差異
  - `"ppm"`：百萬分之一（相對）

**範例**：
```python
from matchms.similarity import PrecursorMzMatch

# 在 0.1 Da 內匹配前驅離子
similarity_func = PrecursorMzMatch(tolerance=0.1, tolerance_type="Dalton")
scores = calculate_scores(references, queries, similarity_func)

# 在 10 ppm 內匹配前驅離子
similarity_func = PrecursorMzMatch(tolerance=10, tolerance_type="ppm")
scores = calculate_scores(references, queries, similarity_func)
```

**輸出**：1.0（匹配）或 0.0（不匹配）

**要求**：兩個質譜都必須有有效的 precursor_mz 元資料。

---

### ParentMassMatch

**描述**：基於母離子質量（中性質量）值的二元匹配。類似於 PrecursorMzMatch，但使用計算的母離子質量而非前驅離子 m/z。

**使用時機**：
- 比較來自不同離子化模式的質譜
- 與加合物無關的匹配
- 基於中性質量的庫搜尋

**參數**：
- `tolerance` (float, 預設=0.1)：匹配的最大質量差異
- `tolerance_type` (str, 預設="Dalton")：容差單位（"Dalton" 或 "ppm"）

**範例**：
```python
from matchms.similarity import ParentMassMatch

similarity_func = ParentMassMatch(tolerance=0.1, tolerance_type="Dalton")
scores = calculate_scores(references, queries, similarity_func)
```

**輸出**：1.0（匹配）或 0.0（不匹配）

**要求**：兩個質譜都必須有有效的 parent_mass 元資料。

---

## 組合多個相似性函數

結合多個相似性指標以實現穩健的化合物識別：

```python
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine, FingerprintSimilarity

# 計算多個相似性分數
cosine_scores = calculate_scores(refs, queries, CosineGreedy())
modified_cosine_scores = calculate_scores(refs, queries, ModifiedCosine())
fingerprint_scores = calculate_scores(refs, queries, FingerprintSimilarity())

# 用權重組合分數
for i, query in enumerate(queries):
    for j, ref in enumerate(refs):
        combined_score = (0.5 * cosine_scores.scores[j, i] +
                         0.3 * modified_cosine_scores.scores[j, i] +
                         0.2 * fingerprint_scores.scores[j, i])
```

## 存取分數結果

`Scores` 物件提供多種方法存取結果：

```python
# 獲取查詢的最佳匹配
best_matches = scores.scores_by_query(query_spectrum, sort=True)[:10]

# 獲取分數為 numpy 陣列
score_array = scores.scores

# 獲取分數為 pandas DataFrame
import pandas as pd
df = scores.to_dataframe()

# 按閾值過濾
high_scores = [(i, j, score) for i, j, score in scores.to_list() if score > 0.7]

# 儲存分數
scores.to_json("scores.json")
scores.to_pickle("scores.pkl")
```

## 效能考量

**快速方法**（大型資料集）：
- CosineGreedy
- PrecursorMzMatch
- ParentMassMatch

**較慢方法**（較小資料集或高準確度）：
- CosineHungarian
- ModifiedCosine（比 CosineGreedy 慢）
- NeutralLossesCosine
- FingerprintSimilarity（需要指紋計算）

**建議**：對於大規模庫搜尋，使用 PrecursorMzMatch 預過濾候選物，然後對過濾結果應用 CosineGreedy 或 ModifiedCosine。

## 常見相似性工作流程

### 標準庫匹配
```python
from matchms.similarity import CosineGreedy

scores = calculate_scores(library_spectra, query_spectra,
                         CosineGreedy(tolerance=0.1))
```

### 多指標匹配
```python
from matchms.similarity import CosineGreedy, ModifiedCosine, FingerprintSimilarity

# 質譜相似性
cosine = calculate_scores(refs, queries, CosineGreedy())
modified = calculate_scores(refs, queries, ModifiedCosine())

# 結構相似性
fingerprint = calculate_scores(refs, queries, FingerprintSimilarity())
```

### 前驅離子過濾匹配
```python
from matchms.similarity import PrecursorMzMatch, CosineGreedy

# 首先按前驅離子質量過濾
mass_filter = calculate_scores(refs, queries, PrecursorMzMatch(tolerance=0.1))

# 然後只對匹配的前驅離子計算餘弦
cosine_scores = calculate_scores(refs, queries, CosineGreedy())
```

## 進一步閱讀

有關詳細的 API 文件、參數描述和數學公式，請參閱：
https://matchms.readthedocs.io/en/latest/api/matchms.similarity.html

