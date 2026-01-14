# PyDESeq2 工作流程指南

本文件提供常見 PyDESeq2 分析模式的詳細逐步工作流程。

## 目錄
1. [完整差異表現分析](#完整差異表現分析)
2. [資料載入與準備](#資料載入與準備)
3. [單因子分析](#單因子分析)
4. [多因子分析](#多因子分析)
5. [結果匯出與視覺化](#結果匯出與視覺化)
6. [常見模式與最佳實務](#常見模式與最佳實務)
7. [疑難排解](#疑難排解)

---

## 完整差異表現分析

### 概述
標準 PyDESeq2 分析包含兩個階段共 12 個主要步驟：

**第一階段：讀取計數建模（步驟 1-7）**
- 標準化和離散度估計
- 對數倍數變化擬合
- 離群值偵測

**第二階段：統計分析（步驟 8-12）**
- Wald 檢定
- 多重檢定校正
- 可選的 LFC 收縮

### 完整工作流程程式碼

```python
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 載入資料
counts_df = pd.read_csv("counts.csv", index_col=0).T  # 如需要則轉置
metadata = pd.read_csv("metadata.csv", index_col=0)

# 過濾低計數基因
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# 移除缺少詮釋資料的樣本
samples_to_keep = ~metadata.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
metadata = metadata.loc[samples_to_keep]

# 初始化 DeseqDataSet
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True
)

# 執行標準化和擬合
dds.deseq2()

# 執行統計檢定
ds = DeseqStats(
    dds,
    contrast=["condition", "treated", "control"],
    alpha=0.05,
    cooks_filter=True,
    independent_filter=True
)
ds.summary()

# 可選：應用 LFC 收縮以進行視覺化
ds.lfc_shrink()

# 存取結果
results = ds.results_df
print(results.head())
```

---

## 資料載入與準備

### 載入 CSV 檔案

計數資料通常以基因 × 樣本格式呈現，但需要轉置：

```python
import pandas as pd

# 載入計數矩陣（基因 × 樣本）
counts_df = pd.read_csv("counts.csv", index_col=0)

# 轉置為樣本 × 基因
counts_df = counts_df.T

# 載入詮釋資料（已為樣本 × 變數格式）
metadata = pd.read_csv("metadata.csv", index_col=0)
```

### 從其他格式載入

**從 TSV：**
```python
counts_df = pd.read_csv("counts.tsv", sep="\t", index_col=0).T
metadata = pd.read_csv("metadata.tsv", sep="\t", index_col=0)
```

**從儲存的 pickle：**
```python
import pickle

with open("counts.pkl", "rb") as f:
    counts_df = pickle.load(f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
```

**從 AnnData：**
```python
import anndata as ad

adata = ad.read_h5ad("data.h5ad")
counts_df = pd.DataFrame(
    adata.X,
    index=adata.obs_names,
    columns=adata.var_names
)
metadata = adata.obs
```

### 資料過濾

**過濾低計數基因：**
```python
# 移除總讀取數少於 10 的基因
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]
```

**過濾缺少詮釋資料的樣本：**
```python
# 移除 'condition' 欄位為 NA 的樣本
samples_to_keep = ~metadata.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
metadata = metadata.loc[samples_to_keep]
```

**根據多個條件過濾：**
```python
# 僅保留符合所有條件的樣本
mask = (
    ~metadata.condition.isna() &
    (metadata.batch.isin(["batch1", "batch2"])) &
    (metadata.age >= 18)
)
counts_df = counts_df.loc[mask]
metadata = metadata.loc[mask]
```

### 資料驗證

**檢查資料結構：**
```python
print(f"計數形狀：{counts_df.shape}")  # 應為（樣本, 基因）
print(f"詮釋資料形狀：{metadata.shape}")  # 應為（樣本, 變數）
print(f"索引匹配：{all(counts_df.index == metadata.index)}")

# 檢查負值
assert (counts_df >= 0).all().all(), "計數必須為非負"

# 檢查非整數值
assert counts_df.applymap(lambda x: x == int(x)).all().all(), "計數必須為整數"
```

---

## 單因子分析

### 簡單雙組比較

比較處理組與對照組樣本：

```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 設計：將表現建模為條件的函數
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition"
)

dds.deseq2()

# 檢定處理組 vs 對照組
ds = DeseqStats(
    dds,
    contrast=["condition", "treated", "control"]
)
ds.summary()

# 結果
results = ds.results_df
significant = results[results.padj < 0.05]
print(f"發現 {len(significant)} 個顯著基因")
```

### 多重配對比較

比較多個組別時：

```python
# 針對對照組檢定每個處理
treatments = ["treated_A", "treated_B", "treated_C"]
all_results = {}

for treatment in treatments:
    ds = DeseqStats(
        dds,
        contrast=["condition", treatment, "control"]
    )
    ds.summary()
    all_results[treatment] = ds.results_df

# 比較各處理的結果
for name, results in all_results.items():
    sig = results[results.padj < 0.05]
    print(f"{name}：{len(sig)} 個顯著基因")
```

---

## 多因子分析

### 雙因子設計

在檢定條件時考慮批次效應：

```python
# 設計包含批次和條件
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~batch + condition"
)

dds.deseq2()

# 在控制批次的同時檢定條件效應
ds = DeseqStats(
    dds,
    contrast=["condition", "treated", "control"]
)
ds.summary()
```

### 交互作用效應

檢定處理效應是否在不同組別間有差異：

```python
# 設計包含交互作用項
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~group + condition + group:condition"
)

dds.deseq2()

# 檢定交互作用項
ds = DeseqStats(dds, contrast=["group:condition", ...])
ds.summary()
```

### 連續型共變數

包含連續變數如年齡：

```python
# 確保年齡在詮釋資料中為數值型
metadata["age"] = pd.to_numeric(metadata["age"])

dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~age + condition"
)

dds.deseq2()
```

---

## 結果匯出與視覺化

### 儲存結果

**匯出為 CSV：**
```python
# 儲存統計結果
ds.results_df.to_csv("deseq2_results.csv")

# 僅儲存顯著基因
significant = ds.results_df[ds.results_df.padj < 0.05]
significant.to_csv("significant_genes.csv")

# 儲存排序後的結果
sorted_results = ds.results_df.sort_values("padj")
sorted_results.to_csv("sorted_results.csv")
```

**儲存 DeseqDataSet：**
```python
import pickle

# 儲存為 AnnData 以供後續使用
with open("dds_result.pkl", "wb") as f:
    pickle.dump(dds.to_picklable_anndata(), f)
```

**載入已儲存的結果：**
```python
# 載入結果
results = pd.read_csv("deseq2_results.csv", index_col=0)

# 載入 AnnData
with open("dds_result.pkl", "rb") as f:
    adata = pickle.load(f)
```

### 基本視覺化

**火山圖：**
```python
import matplotlib.pyplot as plt
import numpy as np

results = ds.results_df.copy()
results["-log10(padj)"] = -np.log10(results.padj)

# 繪圖
plt.figure(figsize=(10, 6))
plt.scatter(
    results.log2FoldChange,
    results["-log10(padj)"],
    alpha=0.5,
    s=10
)
plt.axhline(-np.log10(0.05), color='red', linestyle='--', label='padj=0.05')
plt.axvline(1, color='gray', linestyle='--')
plt.axvline(-1, color='gray', linestyle='--')
plt.xlabel("Log2 倍數變化")
plt.ylabel("-Log10(校正後 P 值)")
plt.title("火山圖")
plt.legend()
plt.savefig("volcano_plot.png", dpi=300)
```

**MA 圖：**
```python
plt.figure(figsize=(10, 6))
plt.scatter(
    np.log10(results.baseMean + 1),
    results.log2FoldChange,
    alpha=0.5,
    s=10,
    c=(results.padj < 0.05),
    cmap='bwr'
)
plt.xlabel("Log10(基礎平均值 + 1)")
plt.ylabel("Log2 倍數變化")
plt.title("MA 圖")
plt.savefig("ma_plot.png", dpi=300)
```

---

## 常見模式與最佳實務

### 1. 資料預處理檢查清單

執行 PyDESeq2 前：
- 確保計數為非負整數
- 驗證樣本 × 基因的方向
- 檢查計數和詮釋資料之間的樣本名稱是否匹配
- 移除或處理缺失的詮釋資料值
- 過濾低計數基因（通常 < 10 總讀取數）
- 驗證實驗因子已正確編碼

### 2. 設計公式最佳實務

**順序很重要：** 將調整變數放在感興趣變數之前
```python
# 正確：控制批次，檢定條件
design = "~batch + condition"

# 較不理想：條件列在前面
design = "~condition + batch"
```

**離散變數使用類別型：**
```python
# 確保正確的資料類型
metadata["condition"] = metadata["condition"].astype("category")
metadata["batch"] = metadata["batch"].astype("category")
```

### 3. 統計檢定指南

**設定適當的 alpha：**
```python
# 標準顯著性閾值
ds = DeseqStats(dds, alpha=0.05)

# 探索性分析使用更嚴格的閾值
ds = DeseqStats(dds, alpha=0.01)
```

**使用獨立過濾：**
```python
# 建議：過濾低統計檢定力的檢定
ds = DeseqStats(dds, independent_filter=True)

# 僅在有特定原因時停用
ds = DeseqStats(dds, independent_filter=False)
```

### 4. LFC 收縮

**何時使用：**
- 用於視覺化（火山圖、熱圖）
- 用於根據效應大小對基因進行排名
- 用於優先選擇後續研究的基因

**何時不使用：**
- 報告統計顯著性（使用未收縮的 p 值）
- 基因集富集分析（通常使用未收縮的值）

```python
# 儲存兩個版本
ds.results_df.to_csv("results_unshrunken.csv")
ds.lfc_shrink()
ds.results_df.to_csv("results_shrunken.csv")
```

### 5. 記憶體管理

對於大型資料集：
```python
# 使用平行處理
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    n_cpus=4  # 根據可用核心數調整
)

# 如需要可分批處理
# （將基因分成區塊，分別分析，合併結果）
```

---

## 疑難排解

### 錯誤：計數和詮釋資料之間的索引不匹配

**問題：** 樣本名稱不匹配
```
KeyError: Sample names in counts and metadata don't match
```

**解決方案：**
```python
# 檢查索引
print("計數樣本：", counts_df.index.tolist())
print("詮釋資料樣本：", metadata.index.tolist())

# 如需要則對齊
common_samples = counts_df.index.intersection(metadata.index)
counts_df = counts_df.loc[common_samples]
metadata = metadata.loc[common_samples]
```

### 錯誤：所有基因計數都為零

**問題：** 資料可能需要轉置
```
ValueError: All genes have zero total counts
```

**解決方案：**
```python
# 檢查資料方向
print(f"計數形狀：{counts_df.shape}")

# 如果基因數 > 樣本數，可能需要轉置
if counts_df.shape[1] < counts_df.shape[0]:
    counts_df = counts_df.T
```

### 警告：許多基因被過濾掉

**問題：** 太多低計數基因被移除

**檢查：**
```python
# 查看基因計數的分佈
print(counts_df.sum(axis=0).describe())

# 視覺化
import matplotlib.pyplot as plt
plt.hist(counts_df.sum(axis=0), bins=50, log=True)
plt.xlabel("每個基因的總計數")
plt.ylabel("頻率")
plt.show()
```

**如需要調整過濾：**
```python
# 嘗試較低的閾值
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 5]
```

### 錯誤：設計矩陣不是滿秩的

**問題：** 混淆的設計（例如，所有處理樣本都在同一批次中）

**解決方案：**
```python
# 檢查設計混淆
print(pd.crosstab(metadata.condition, metadata.batch))

# 移除混淆變數或新增交互作用項
design = "~condition"  # 移除批次
# 或
design = "~condition + batch + condition:batch"  # 新增交互作用
```

### 問題：沒有發現顯著基因

**可能原因：**
1. 效應大小較小
2. 生物學變異性高
3. 樣本量不足
4. 技術問題（批次效應、離群值）

**診斷：**
```python
# 檢查離散度估計
import matplotlib.pyplot as plt
dispersions = dds.varm["dispersions"]
plt.hist(dispersions, bins=50)
plt.xlabel("離散度")
plt.ylabel("頻率")
plt.show()

# 檢查大小因子（應接近 1）
print("大小因子：", dds.obsm["size_factors"])

# 即使不顯著也查看最佳基因
top_genes = ds.results_df.nsmallest(20, "pvalue")
print(top_genes)
```

### 大型資料集的記憶體錯誤

**解決方案：**
```python
# 1. 使用較少的 CPU（矛盾的是可能有幫助）
dds = DeseqDataSet(..., n_cpus=1)

# 2. 更積極地過濾
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 20]

# 3. 分批處理
# 按基因子集分割分析並合併結果
```
