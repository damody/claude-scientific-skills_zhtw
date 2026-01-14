# scvi-tools 中的差異表達分析

本文檔提供有關使用 scvi-tools 機率框架進行差異表達（DE）分析的詳細資訊。

## 概述

scvi-tools 實作了貝葉斯差異表達檢驗，利用學習到的生成模型來估計組間表達差異。這種方法相較於傳統方法具有多項優勢：

- **批次校正**：在批次校正的表示上進行 DE 檢驗
- **不確定性量化**：效應量的機率估計
- **零膨脹處理**：正確建模丟失和零值
- **靈活比較**：可比較任意組或細胞類型
- **多模態**：適用於 RNA、蛋白質（totalVI）和可及性（PeakVI）

## 核心統計框架

### 問題定義

目標是估計兩個條件之間表達的對數倍數變化（log fold-change）：

```
log fold-change = log(μ_B) - log(μ_A)
```

其中 μ_A 和 μ_B 分別是條件 A 和 B 的平均表達水平。

### 三階段流程

**階段一：估計表達水平**
- 從細胞狀態的後驗分佈中取樣
- 從學習到的生成模型生成表達值
- 跨細胞聚合以獲得群體層級估計

**階段二：檢測相關特徵（假設檢驗）**
- 使用貝葉斯框架檢驗差異表達
- 有兩種檢驗模式可用：
  - **「vanilla」模式**：點虛無假設（β = 0）
  - **「change」模式**：複合假設（|β| ≤ δ）

**階段三：控制錯誤發現**
- 後驗期望錯誤發現比例（FDP）控制
- 選擇最大發現數量以確保 E[FDP] ≤ α

## 基本用法

### 簡單的兩組比較

```python
import scvi

# 訓練模型後
model = scvi.model.SCVI(adata)
model.train()

# 比較兩種細胞類型
de_results = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    group2="B cells"
)

# 查看前幾個 DE 基因
top_genes = de_results.sort_values("lfc_mean", ascending=False).head(20)
print(top_genes[["lfc_mean", "lfc_std", "bayes_factor", "is_de_fdr_0.05"]])
```

### 一對其餘比較

```python
# 將一組與其他所有組比較
de_results = model.differential_expression(
    groupby="cell_type",
    group1="T cells"  # 沒有 group2 = 與其餘比較
)
```

### 所有成對比較

```python
# 成對比較所有細胞類型
all_comparisons = {}

cell_types = adata.obs["cell_type"].unique()

for ct1 in cell_types:
    for ct2 in cell_types:
        if ct1 != ct2:
            key = f"{ct1}_vs_{ct2}"
            all_comparisons[key] = model.differential_expression(
                groupby="cell_type",
                group1=ct1,
                group2=ct2
            )
```

## 關鍵參數

### `groupby`（必需）
`adata.obs` 中定義要比較組別的欄位。

```python
# 必須是分類變數
de_results = model.differential_expression(groupby="cell_type")
```

### `group1` 和 `group2`
要比較的組別。如果 `group2` 為 None，則將 `group1` 與所有其他組比較。

```python
# 特定比較
de = model.differential_expression(groupby="condition", group1="treated", group2="control")

# 一對其餘
de = model.differential_expression(groupby="cell_type", group1="T cells")
```

### `mode`（假設檢驗模式）

**「vanilla」模式**（預設）：點虛無假設
- 檢驗 β = 0 是否精確成立
- 更敏感，但可能找到微不足道的小效應

**「change」模式**：複合虛無假設
- 檢驗 |β| ≤ δ
- 要求生物學上有意義的變化
- 減少對微小效應的錯誤發現

```python
# 帶有最小效應量的 change 模式
de = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    group2="B cells",
    mode="change",
    delta=0.25  # 最小對數倍數變化
)
```

### `delta`
「change」模式的最小效應量閾值。
- 典型值：0.25、0.5、0.7（對數尺度）
- log2(1.5) ≈ 0.58（1.5 倍變化）
- log2(2) = 1.0（2 倍變化）

```python
# 要求至少 1.5 倍變化
de = model.differential_expression(
    groupby="condition",
    group1="disease",
    group2="healthy",
    mode="change",
    delta=0.58  # log2(1.5)
)
```

### `fdr_target`
錯誤發現率閾值（預設：0.05）

```python
# 更嚴格的 FDR 控制
de = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    fdr_target=0.01
)
```

### `batch_correction`
是否在 DE 檢驗期間執行批次校正（預設：True）

```python
# 在特定批次內檢驗
de = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    group2="B cells",
    batch_correction=False
)
```

### `n_samples`
用於估計的後驗樣本數（預設：5000）
- 更多樣本 = 更準確但更慢
- 減少以提高速度，增加以提高精確度

```python
# 高精確度分析
de = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    n_samples=10000
)
```

## 解讀結果

### 輸出欄位

結果 DataFrame 包含幾個重要欄位：

**效應量估計**：
- `lfc_mean`：平均對數倍數變化
- `lfc_median`：中位對數倍數變化
- `lfc_std`：對數倍數變化的標準差
- `lfc_min`：效應量下界
- `lfc_max`：效應量上界

**統計顯著性**：
- `bayes_factor`：差異表達的貝葉斯因子
  - 較高值 = 更強證據
  - >3 通常被認為有意義
- `is_de_fdr_0.05`：布林值，指示基因在 FDR 0.05 下是否為 DE
- `is_de_fdr_0.1`：布林值，指示基因在 FDR 0.1 下是否為 DE

**表達水平**：
- `mean1`：組 1 中的平均表達
- `mean2`：組 2 中的平均表達
- `non_zeros_proportion1`：組 1 中非零細胞的比例
- `non_zeros_proportion2`：組 2 中非零細胞的比例

### 解讀範例

```python
de_results = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    group2="B cells"
)

# 找到在 T 細胞中顯著上調的基因
upreg_tcells = de_results[
    (de_results["is_de_fdr_0.05"]) &
    (de_results["lfc_mean"] > 0)
].sort_values("lfc_mean", ascending=False)

print(f"T 細胞中上調的基因：{len(upreg_tcells)}")
print(upreg_tcells.head(10))

# 找到具有大效應量的基因
large_effect = de_results[
    (de_results["is_de_fdr_0.05"]) &
    (abs(de_results["lfc_mean"]) > 1)  # 2 倍變化
]
```

## 進階用法

### 特定細胞內的 DE

```python
# 僅在細胞子集內檢驗 DE
subset_indices = adata.obs["tissue"] == "lung"

de = model.differential_expression(
    idx1=adata.obs["cell_type"] == "T cells" & subset_indices,
    idx2=adata.obs["cell_type"] == "B cells" & subset_indices
)
```

### 批次特定的 DE

```python
# 在每個批次內分別檢驗 DE
batches = adata.obs["batch"].unique()

batch_de_results = {}
for batch in batches:
    batch_idx = adata.obs["batch"] == batch
    batch_de_results[batch] = model.differential_expression(
        idx1=(adata.obs["condition"] == "treated") & batch_idx,
        idx2=(adata.obs["condition"] == "control") & batch_idx
    )
```

### 偽批量 DE

```python
# 在 DE 檢驗前聚合細胞
# 對於每組細胞數量少的情況很有用

de = model.differential_expression(
    groupby="cell_type",
    group1="rare_cell_type",
    group2="common_cell_type",
    n_samples=10000,  # 更多樣本以提高穩定性
    batch_correction=True
)
```

## 視覺化

### 火山圖（Volcano Plot）

```python
import matplotlib.pyplot as plt
import numpy as np

de = model.differential_expression(
    groupby="condition",
    group1="treated",
    group2="control"
)

# 火山圖
plt.figure(figsize=(10, 6))
plt.scatter(
    de["lfc_mean"],
    -np.log10(1 / (de["bayes_factor"] + 1)),
    c=de["is_de_fdr_0.05"],
    cmap="coolwarm",
    alpha=0.5
)
plt.xlabel("Log Fold Change")
plt.ylabel("-log10(1/Bayes Factor)")
plt.title("火山圖：處理 vs 對照")
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.show()
```

### 前幾個 DE 基因的熱圖

```python
import seaborn as sns

# 獲取前幾個 DE 基因
top_genes = de.sort_values("lfc_mean", ascending=False).head(50).index

# 獲取標準化表達
norm_expr = model.get_normalized_expression(
    adata,
    indices=adata.obs["condition"].isin(["treated", "control"]),
    gene_list=top_genes
)

# 繪製熱圖
plt.figure(figsize=(12, 10))
sns.heatmap(
    norm_expr.T,
    cmap="viridis",
    xticklabels=False,
    yticklabels=top_genes
)
plt.title("前 50 個 DE 基因")
plt.show()
```

### 排序基因圖

```python
# 繪製按效應量排序的基因
de_sorted = de.sort_values("lfc_mean", ascending=False)

plt.figure(figsize=(12, 6))
plt.plot(range(len(de_sorted)), de_sorted["lfc_mean"].values)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("基因排名")
plt.ylabel("Log Fold Change")
plt.title("按效應量排序的基因")
plt.show()
```

## 與傳統方法的比較

### scvi-tools vs. Wilcoxon 檢驗

```python
import scanpy as sc

# 傳統 Wilcoxon 檢驗
sc.tl.rank_genes_groups(
    adata,
    groupby="cell_type",
    method="wilcoxon",
    key_added="wilcoxon"
)

# scvi-tools DE
de_scvi = model.differential_expression(
    groupby="cell_type",
    group1="T cells"
)

# 比較結果
wilcox_results = sc.get.rank_genes_groups_df(adata, group="T cells", key="wilcoxon")
```

**scvi-tools 的優勢**：
- 自動考慮批次效應
- 正確處理零膨脹
- 提供不確定性量化
- 不需要任意的偽計數
- 更好的統計特性

**何時使用 Wilcoxon**：
- 非常快速的探索性分析
- 與使用 Wilcoxon 的已發表結果進行比較

## 多模態 DE

### 蛋白質 DE（totalVI）

```python
# 在 CITE-seq 數據上訓練 totalVI
totalvi_model = scvi.model.TOTALVI(adata)
totalvi_model.train()

# RNA 差異表達
rna_de = totalvi_model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    group2="B cells",
    protein_expression=False  # 預設
)

# 蛋白質差異表達
protein_de = totalvi_model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    group2="B cells",
    protein_expression=True
)

print(f"DE 基因：{rna_de['is_de_fdr_0.05'].sum()}")
print(f"DE 蛋白質：{protein_de['is_de_fdr_0.05'].sum()}")
```

### 差異可及性（PeakVI）

```python
# 在 ATAC-seq 數據上訓練 PeakVI
peakvi_model = scvi.model.PEAKVI(atac_adata)
peakvi_model.train()

# 差異可及性
da = peakvi_model.differential_accessibility(
    groupby="cell_type",
    group1="T cells",
    group2="B cells"
)

# 與 DE 相同的解讀方式
```

## 處理特殊情況

### 低細胞數量組

```python
# 增加後驗樣本以提高穩定性
de = model.differential_expression(
    groupby="cell_type",
    group1="rare_type",  # 例如 50 個細胞
    group2="common_type",  # 例如 5000 個細胞
    n_samples=10000
)
```

### 不平衡比較

```python
# 當組別大小非常不同時
# 使用 change 模式以避免微小效應

de = model.differential_expression(
    groupby="condition",
    group1="rare_condition",
    group2="common_condition",
    mode="change",
    delta=0.5
)
```

### 多重檢驗校正

```python
# 已通過 FDP 控制包含
# 但可以應用額外校正

from statsmodels.stats.multitest import multipletests

# Bonferroni 校正（非常保守）
_, pvals_corrected, _, _ = multipletests(
    1 / (de["bayes_factor"] + 1),
    method="bonferroni"
)
```

## 效能考量

### 速度優化

```python
# 大數據集的更快 DE 檢驗
de = model.differential_expression(
    groupby="cell_type",
    group1="T cells",
    n_samples=1000,  # 減少樣本
    batch_size=512    # 增加批次大小
)
```

### 記憶體管理

```python
# 對於非常大的數據集
# 一次檢驗一個比較而不是所有成對

cell_types = adata.obs["cell_type"].unique()
for ct in cell_types:
    de = model.differential_expression(
        groupby="cell_type",
        group1=ct
    )
    # 儲存結果
    de.to_csv(f"de_results_{ct}.csv")
```

## 最佳實踐

1. **使用「change」模式**：獲得生物學上可解讀的結果
2. **設定適當的 delta**：基於生物學顯著性
3. **檢查表達水平**：過濾低表達基因
4. **驗證發現**：檢查標記基因以確保合理
5. **視覺化結果**：始終繪製前幾個 DE 基因
6. **報告參數**：記錄使用的模式、delta、FDR
7. **考慮批次效應**：使用 batch_correction=True
8. **多重比較**：注意檢驗多個組
9. **樣本量**：確保每組有足夠的細胞（建議 >50）
10. **生物學驗證**：後續進行功能實驗

## 範例：完整的 DE 分析工作流程

```python
import scvi
import scanpy as sc
import matplotlib.pyplot as plt

# 1. 訓練模型
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
model = scvi.model.SCVI(adata)
model.train()

# 2. 執行 DE 分析
de_results = model.differential_expression(
    groupby="cell_type",
    group1="Disease_T_cells",
    group2="Healthy_T_cells",
    mode="change",
    delta=0.5,
    fdr_target=0.05
)

# 3. 過濾和分析
sig_genes = de_results[de_results["is_de_fdr_0.05"]]
upreg = sig_genes[sig_genes["lfc_mean"] > 0].sort_values("lfc_mean", ascending=False)
downreg = sig_genes[sig_genes["lfc_mean"] < 0].sort_values("lfc_mean")

print(f"顯著基因：{len(sig_genes)}")
print(f"上調：{len(upreg)}")
print(f"下調：{len(downreg)}")

# 4. 視覺化前幾個基因
top_genes = upreg.head(10).index.tolist() + downreg.head(10).index.tolist()

sc.pl.violin(
    adata[adata.obs["cell_type"].isin(["Disease_T_cells", "Healthy_T_cells"])],
    keys=top_genes,
    groupby="cell_type",
    rotation=90
)

# 5. 功能富集（使用外部工具）
# 例如 g:Profiler、DAVID 或 gprofiler-official Python 套件
upreg_genes = upreg.head(100).index.tolist()
# 執行通路分析...

# 6. 儲存結果
de_results.to_csv("de_results_disease_vs_healthy.csv")
upreg.to_csv("upregulated_genes.csv")
downreg.to_csv("downregulated_genes.csv")
```
