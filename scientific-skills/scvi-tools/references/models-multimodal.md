# 多模態與多組學整合模型

本文檔涵蓋 scvi-tools 中用於多種數據模態聯合分析的模型。

## totalVI（Total Variational Inference，總變分推斷）

**目的**：CITE-seq 數據（來自同一細胞的同時 RNA 和蛋白質測量）的聯合分析。

**主要特點**：
- 聯合建模基因表達和蛋白質豐度
- 學習共享的低維表示
- 支援從 RNA 數據進行蛋白質插補
- 對兩種模態執行差異表達
- 處理 RNA 和蛋白質層的批次效應

**何時使用**：
- 分析 CITE-seq 或 REAP-seq 數據
- 聯合 RNA + 表面蛋白測量
- 插補缺失的蛋白質
- 整合蛋白質和 RNA 資訊
- 多批次 CITE-seq 整合

**數據要求**：
- 在 `.X` 或層中包含基因表達的 AnnData
- 在 `.obsm["protein_expression"]` 中的蛋白質測量
- 兩種模態測量的是同一批細胞

**基本用法**：
```python
import scvi

# 設定數據 - 指定 RNA 和蛋白質層
scvi.model.TOTALVI.setup_anndata(
    adata,
    layer="counts",  # RNA 計數
    protein_expression_obsm_key="protein_expression",  # 蛋白質計數
    batch_key="batch"
)

# 訓練模型
model = scvi.model.TOTALVI(adata)
model.train()

# 獲取聯合潛在表示
latent = model.get_latent_representation()

# 獲取兩種模態的標準化值
rna_normalized = model.get_normalized_expression()
protein_normalized = model.get_normalized_expression(
    transform_batch="batch1",
    protein_expression=True
)

# 差異表達（適用於 RNA 和蛋白質）
rna_de = model.differential_expression(groupby="cell_type")
protein_de = model.differential_expression(
    groupby="cell_type",
    protein_expression=True
)
```

**關鍵參數**：
- `n_latent`：潛在空間維度（預設：20）
- `n_layers_encoder`：編碼器層數（預設：1）
- `n_layers_decoder`：解碼器層數（預設：1）
- `protein_dispersion`：蛋白質離散度處理（「protein」或「protein-batch」）
- `empirical_protein_background_prior`：使用蛋白質的經驗背景

**進階功能**：

**蛋白質插補**：
```python
# 為僅有 RNA 的細胞插補缺失的蛋白質
# （用於將 RNA-seq 映射到 CITE-seq 參考）
protein_foreground = model.get_protein_foreground_probability()
imputed_proteins = model.get_normalized_expression(
    protein_expression=True,
    n_samples=25
)
```

**去噪**：
```python
# 獲取兩種模態的去噪計數
denoised_rna = model.get_normalized_expression(n_samples=25)
denoised_protein = model.get_normalized_expression(
    protein_expression=True,
    n_samples=25
)
```

**最佳實踐**：
1. 對於具有環境蛋白質的數據集，使用經驗蛋白質背景先驗
2. 對於異質蛋白質數據，考慮蛋白質特定的離散度
3. 使用聯合潛在空間進行聚類（優於僅使用 RNA）
4. 使用已知標記驗證蛋白質插補
5. 訓練前檢查蛋白質 QC 指標

## MultiVI（Multi-modal Variational Inference，多模態變分推斷）

**目的**：配對和非配對多組學數據的整合（例如 RNA + ATAC，配對和非配對細胞）。

**主要特點**：
- 處理配對數據（同一細胞）和非配對數據（不同細胞）
- 整合多種模態：RNA、ATAC、蛋白質等
- 缺失模態插補
- 學習跨模態的共享表示
- 靈活的整合策略

**何時使用**：
- 10x Multiome 數據（配對 RNA + ATAC）
- 整合分開的 RNA-seq 和 ATAC-seq 實驗
- 部分細胞有兩種模態，部分只有一種
- 跨模態插補任務

**數據要求**：
- 包含多種模態的 AnnData
- 模態指示器（每個細胞有哪些測量）
- 可處理：
  - 所有細胞都有兩種模態（完全配對）
  - 配對和非配對細胞的混合
  - 完全非配對的數據集

**基本用法**：
```python
# 準備帶有模態資訊的數據
# adata.X 應包含所有特徵（基因 + 峰值）
# adata.var["modality"] 指示「Gene」或「Peak」
# adata.obs["modality"] 指示每個細胞有哪種模態

scvi.model.MULTIVI.setup_anndata(
    adata,
    batch_key="batch",
    modality_key="modality"  # 指示細胞模態的欄位
)

model = scvi.model.MULTIVI(adata)
model.train()

# 獲取聯合潛在表示
latent = model.get_latent_representation()

# 插補缺失的模態
# 例如，為僅有 RNA 的細胞預測 ATAC
imputed_accessibility = model.get_accessibility_estimates(
    indices=rna_only_indices
)

# 獲取標準化的表達/可及性
rna_normalized = model.get_normalized_expression()
atac_normalized = model.get_accessibility_estimates()
```

**關鍵參數**：
- `n_genes`：基因特徵數
- `n_regions`：可及性區域數
- `n_latent`：潛在維度（預設：20）

**整合情境**：

**情境一：完全配對（10x Multiome）**：
```python
# 所有細胞都有 RNA 和 ATAC
# 單一模態鍵：「paired」
adata.obs["modality"] = "paired"
```

**情境二：部分配對**：
```python
# 部分細胞有兩種模態，部分僅有 RNA，部分僅有 ATAC
adata.obs["modality"] = ["RNA+ATAC", "RNA", "ATAC", ...]
```

**情境三：完全非配對**：
```python
# 分開的 RNA 和 ATAC 實驗
adata.obs["modality"] = ["RNA"] * n_rna + ["ATAC"] * n_atac
```

**進階用例**：

**跨模態預測**：
```python
# 從基因表達預測峰值
accessibility_from_rna = model.get_accessibility_estimates(
    indices=rna_only_cells
)

# 從可及性預測基因
expression_from_atac = model.get_normalized_expression(
    indices=atac_only_cells
)
```

**模態特定分析**：
```python
# 按模態分開分析
rna_subset = adata[adata.obs["modality"].str.contains("RNA")]
atac_subset = adata[adata.obs["modality"].str.contains("ATAC")]
```

## MrVI（Multi-resolution Variational Inference，多解析度變分推斷）

**目的**：考慮樣本特定和共享變異的多樣本分析。

**主要特點**：
- 同時分析多個樣本/條件
- 將變異分解為：
  - 共享變異（跨樣本通用）
  - 樣本特定變異
- 支援樣本層級比較
- 識別樣本特定的細胞狀態

**何時使用**：
- 比較多個生物樣本或條件
- 識別樣本特定 vs 共享的細胞狀態
- 疾病 vs 健康樣本比較
- 理解樣本間異質性
- 多供體研究

**基本用法**：
```python
scvi.model.MRVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    sample_key="sample"  # 關鍵：定義生物樣本
)

model = scvi.model.MRVI(adata, n_latent=10, n_latent_sample=5)
model.train()

# 獲取表示
shared_latent = model.get_latent_representation()  # 跨樣本共享
sample_specific = model.get_sample_specific_representation()

# 樣本距離矩陣
sample_distances = model.get_sample_distances()
```

**關鍵參數**：
- `n_latent`：共享潛在空間的維度
- `n_latent_sample`：樣本特定空間的維度
- `sample_key`：定義生物樣本的欄位

**分析工作流程**：
```python
# 1. 識別跨樣本的共享細胞類型
sc.pp.neighbors(adata, use_rep="X_MrVI_shared")
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="shared_clusters")

# 2. 分析樣本特定變異
sample_repr = model.get_sample_specific_representation()

# 3. 比較樣本
distances = model.get_sample_distances()

# 4. 找到樣本富集的基因
de_results = model.differential_expression(
    groupby="sample",
    group1="Disease",
    group2="Healthy"
)
```

**用例**：
- **多供體研究**：將供體效應與細胞類型變異分開
- **疾病研究**：識別疾病特定 vs 共享的生物學
- **時間序列**：將時間性與穩定變異分開
- **批次 + 生物學**：解開技術和生物學變異

## totalVI vs. MultiVI vs. MrVI：何時使用哪個？

### totalVI
**用於**：CITE-seq（RNA + 蛋白質，同一細胞）
- 配對測量
- 每個特徵單一模態類型
- 重點：蛋白質插補、聯合分析

### MultiVI
**用於**：多種模態（RNA + ATAC 等）
- 配對、非配對或混合
- 不同特徵類型
- 重點：跨模態整合和插補

### MrVI
**用於**：多樣本 RNA-seq
- 單一模態（RNA）
- 多個生物樣本
- 重點：樣本層級變異分解

## 整合最佳實踐

### 對於 CITE-seq（totalVI）
1. **蛋白質品質控制**：移除低品質抗體
2. **背景扣除**：使用經驗背景先驗
3. **聯合聚類**：使用聯合潛在空間，而非僅 RNA
4. **驗證**：在兩種模態中檢查已知標記

### 對於 Multiome/多模態（MultiVI）
1. **特徵過濾**：分別過濾基因和峰值
2. **平衡模態**：確保每種模態有合理的代表性
3. **模態權重**：考慮是否一種模態佔主導
4. **插補驗證**：仔細驗證插補值

### 對於多樣本（MrVI）
1. **樣本定義**：仔細定義生物樣本
2. **樣本量**：每個樣本需要足夠的細胞
3. **共變量處理**：正確處理批次 vs 樣本
4. **解釋**：區分技術和生物學變異

## 完整範例：使用 totalVI 的 CITE-seq 分析

```python
import scvi
import scanpy as sc

# 1. 載入 CITE-seq 數據
adata = sc.read_h5ad("cite_seq.h5ad")

# 2. QC 和過濾
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, n_top_genes=4000)

# 蛋白質 QC
protein_counts = adata.obsm["protein_expression"]
# 移除低品質蛋白質

# 3. 設定 totalVI
scvi.model.TOTALVI.setup_anndata(
    adata,
    layer="counts",
    protein_expression_obsm_key="protein_expression",
    batch_key="batch"
)

# 4. 訓練
model = scvi.model.TOTALVI(adata, n_latent=20)
model.train(max_epochs=400)

# 5. 提取聯合表示
latent = model.get_latent_representation()
adata.obsm["X_totalVI"] = latent

# 6. 在聯合空間上聚類
sc.pp.neighbors(adata, use_rep="X_totalVI")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# 7. 兩種模態的差異表達
rna_de = model.differential_expression(
    groupby="leiden",
    group1="0",
    group2="1"
)

protein_de = model.differential_expression(
    groupby="leiden",
    group1="0",
    group2="1",
    protein_expression=True
)

# 8. 儲存模型
model.save("totalvi_model")
```
