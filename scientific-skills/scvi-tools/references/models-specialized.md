# 特殊模態模型

本文檔涵蓋 scvi-tools 中用於特殊單細胞數據模態的模型。

## MethylVI / MethylANVI（甲基化分析）

**目的**：分析單細胞亞硫酸鹽定序（scBS-seq）數據用於 DNA 甲基化。

**主要特點**：
- 以單細胞解析度建模甲基化模式
- 處理甲基化數據的稀疏性
- 甲基化實驗的批次校正
- 標籤遷移（MethylANVI）用於細胞類型註釋

**何時使用**：
- 分析 scBS-seq 或類似的甲基化數據
- 研究跨細胞類型的 DNA 甲基化模式
- 跨批次整合甲基化數據
- 基於甲基化譜進行細胞類型註釋

**數據要求**：
- 甲基化計數矩陣（每個 CpG 位點的甲基化 vs 總讀數）
- 格式：細胞 × CpG 位點，包含甲基化比率或計數

### MethylVI（無監督）

**基本用法**：
```python
import scvi

# 設定甲基化數據
scvi.model.METHYLVI.setup_anndata(
    adata,
    layer="methylation_counts",  # 甲基化數據
    batch_key="batch"
)

model = scvi.model.METHYLVI(adata)
model.train()

# 獲取潛在表示
latent = model.get_latent_representation()

# 獲取標準化甲基化值
normalized_meth = model.get_normalized_methylation()
```

### MethylANVI（帶有細胞類型的半監督）

**基本用法**：
```python
# 設定帶有細胞類型標籤
scvi.model.METHYLANVI.setup_anndata(
    adata,
    layer="methylation_counts",
    batch_key="batch",
    labels_key="cell_type",
    unlabeled_category="Unknown"
)

model = scvi.model.METHYLANVI(adata)
model.train()

# 預測細胞類型
predictions = model.predict()
```

**關鍵參數**：
- `n_latent`：潛在維度
- `region_factors`：建模區域特異性效應

**用例**：
- 表觀遺傳異質性分析
- 通過甲基化進行細胞類型識別
- 與基因表達數據整合（分開分析）
- 差異甲基化分析

## CytoVI（流式和質譜細胞術）

**目的**：流式細胞術和質譜細胞術（CyTOF）數據的批次校正和整合。

**主要特點**：
- 處理基於抗體的蛋白質測量
- 校正細胞術數據中的批次效應
- 支援跨實驗整合
- 為高維蛋白質面板設計

**何時使用**：
- 分析流式細胞術或 CyTOF 數據
- 跨批次整合細胞術實驗
- 蛋白質面板的批次校正
- 跨研究細胞術整合

**數據要求**：
- 蛋白質表達矩陣（細胞 × 蛋白質）
- 流式細胞術或 CyTOF 測量
- 批次/實驗註釋

**基本用法**：
```python
scvi.model.CYTOVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein_expression",
    batch_key="batch"
)

model = scvi.model.CYTOVI(adata)
model.train()

# 獲取批次校正的表示
latent = model.get_latent_representation()

# 獲取標準化蛋白質值
normalized = model.get_normalized_expression()
```

**關鍵參數**：
- `n_latent`：潛在空間維度
- `n_layers`：網路深度

**典型工作流程**：
```python
import scanpy as sc

# 1. 載入細胞術數據
adata = sc.read_h5ad("cytof_data.h5ad")

# 2. 訓練 CytoVI
scvi.model.CYTOVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein",
    batch_key="experiment"
)
model = scvi.model.CYTOVI(adata)
model.train()

# 3. 獲取批次校正值
latent = model.get_latent_representation()
adata.obsm["X_CytoVI"] = latent

# 4. 下游分析
sc.pp.neighbors(adata, use_rep="X_CytoVI")
sc.tl.umap(adata)
sc.tl.leiden(adata)

# 5. 視覺化批次校正
sc.pl.umap(adata, color=["batch", "leiden"])
```

## SysVI（系統級整合）

**目的**：批次效應校正，強調保留生物學變異。

**主要特點**：
- 專門的批次整合方法
- 在移除技術效應的同時保留生物學信號
- 為大規模整合研究設計

**何時使用**：
- 大規模多批次整合
- 需要保留微妙的生物學變異
- 跨多個研究的系統級分析

**基本用法**：
```python
scvi.model.SYSVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch"
)

model = scvi.model.SYSVI(adata)
model.train()

latent = model.get_latent_representation()
```

## Decipher（軌跡推斷）

**目的**：單細胞數據的軌跡推斷和偽時間分析。

**主要特點**：
- 學習細胞軌跡和分化路徑
- 偽時間估計
- 考慮軌跡結構的不確定性
- 與 scVI 嵌入相容

**何時使用**：
- 研究細胞分化
- 時間序列或發育數據集
- 理解細胞狀態轉換
- 識別發育中的分支點

**基本用法**：
```python
# 通常在 scVI 獲得嵌入後使用
scvi_model = scvi.model.SCVI(adata)
scvi_model.train()

# Decipher 用於軌跡
scvi.model.DECIPHER.setup_anndata(adata)
decipher_model = scvi.model.DECIPHER(adata, scvi_model)
decipher_model.train()

# 獲取偽時間
pseudotime = decipher_model.get_pseudotime()
adata.obs["pseudotime"] = pseudotime
```

**視覺化**：
```python
import scanpy as sc

# 在 UMAP 上繪製偽時間
sc.pl.umap(adata, color="pseudotime", cmap="viridis")

# 沿偽時間的基因表達
sc.pl.scatter(adata, x="pseudotime", y="gene_of_interest")
```

## peRegLM（峰值調控線性模型）

**目的**：將染色質可及性與基因表達聯繫起來進行調控分析。

**主要特點**：
- 將 ATAC-seq 峰值與基因表達聯繫起來
- 識別調控關係
- 適用於配對的 multiome 數據

**何時使用**：
- Multiome 數據（來自同一細胞的 RNA + ATAC）
- 理解基因調控
- 將峰值連結到目標基因
- 調控網路構建

**基本用法**：
```python
# 需要配對的 RNA + ATAC 數據
scvi.model.PEREGLM.setup_anndata(
    multiome_adata,
    rna_layer="counts",
    atac_layer="atac_counts"
)

model = scvi.model.PEREGLM(multiome_adata)
model.train()

# 獲取峰值-基因聯繫
peak_gene_links = model.get_regulatory_links()
```

## 模型特定最佳實踐

### MethylVI/MethylANVI
1. **稀疏性**：甲基化數據本質上是稀疏的；模型會考慮這一點
2. **CpG 選擇**：過濾覆蓋度非常低的 CpG
3. **生物學解釋**：考慮基因組背景（啟動子、增強子）
4. **整合**：對於多組學，分開分析然後整合結果

### CytoVI
1. **蛋白質 QC**：移除低品質或無資訊的蛋白質
2. **補償**：分析前確保正確的光譜補償
3. **批次設計**：包含生物學和技術重複
4. **對照**：使用對照樣本驗證批次校正

### SysVI
1. **樣本量**：為大規模整合設計
2. **批次定義**：仔細定義批次結構
3. **生物學驗證**：驗證保留的生物學信號

### Decipher
1. **起點**：如果已知，定義軌跡起始細胞
2. **分支**：指定預期的分支數
3. **驗證**：使用已知標記驗證偽時間
4. **整合**：與 scVI 嵌入配合良好

## 與其他模型的整合

許多特殊模型可以很好地組合使用：

**甲基化 + 表達**：
```python
# 分開分析，然後整合
methylvi_model = scvi.model.METHYLVI(meth_adata)
scvi_model = scvi.model.SCVI(rna_adata)

# 在分析層面整合結果
# 例如，關聯甲基化和表達模式
```

**細胞術 + CITE-seq**：
```python
# CytoVI 用於流式/CyTOF
cyto_model = scvi.model.CYTOVI(cyto_adata)

# totalVI 用於 CITE-seq
cite_model = scvi.model.TOTALVI(cite_adata)

# 比較跨平台的蛋白質測量
```

**ATAC + RNA（Multiome）**：
```python
# MultiVI 用於聯合分析
multivi_model = scvi.model.MULTIVI(multiome_adata)

# peRegLM 用於調控連結
pereglm_model = scvi.model.PEREGLM(multiome_adata)
```

## 選擇特殊模型

### 決策樹

1. **什麼數據模態？**
   - 甲基化 → MethylVI/MethylANVI
   - 流式/CyTOF → CytoVI
   - 軌跡 → Decipher
   - 多批次整合 → SysVI
   - 調控連結 → peRegLM

2. **有標籤嗎？**
   - 有 → MethylANVI（甲基化）
   - 沒有 → MethylVI（甲基化）

3. **主要目標是什麼？**
   - 批次校正 → CytoVI、SysVI
   - 軌跡/偽時間 → Decipher
   - 峰值-基因連結 → peRegLM
   - 甲基化模式 → MethylVI/ANVI

## 範例：完整的甲基化分析

```python
import scvi
import scanpy as sc

# 1. 載入甲基化數據
meth_adata = sc.read_h5ad("methylation_data.h5ad")

# 2. QC：過濾低覆蓋度 CpG 位點
sc.pp.filter_genes(meth_adata, min_cells=10)

# 3. 設定 MethylVI
scvi.model.METHYLVI.setup_anndata(
    meth_adata,
    layer="methylation",
    batch_key="batch"
)

# 4. 訓練模型
model = scvi.model.METHYLVI(meth_adata, n_latent=15)
model.train(max_epochs=400)

# 5. 獲取潛在表示
latent = model.get_latent_representation()
meth_adata.obsm["X_MethylVI"] = latent

# 6. 聚類
sc.pp.neighbors(meth_adata, use_rep="X_MethylVI")
sc.tl.umap(meth_adata)
sc.tl.leiden(meth_adata)

# 7. 差異甲基化
dm_results = model.differential_methylation(
    groupby="leiden",
    group1="0",
    group2="1"
)

# 8. 儲存
model.save("methylvi_model")
meth_adata.write("methylation_analyzed.h5ad")
```

## 外部工具整合

一些特殊模型作為外部套件提供：

**SOLO**（雙細胞檢測）：
```python
from scvi.external import SOLO

solo = SOLO.from_scvi_model(scvi_model)
solo.train()
doublets = solo.predict()
```

**scArches**（參考映射）：
```python
from scvi.external import SCARCHES

# 用於遷移學習和查詢到參考映射
```

這些外部工具擴展了 scvi-tools 的功能以適用於特定用例。

## 總結表

| 模型 | 數據類型 | 主要用途 | 監督？ |
|------|----------|----------|--------|
| MethylVI | 甲基化 | 無監督分析 | 否 |
| MethylANVI | 甲基化 | 細胞類型註釋 | 半監督 |
| CytoVI | 細胞術 | 批次校正 | 否 |
| SysVI | scRNA-seq | 大規模整合 | 否 |
| Decipher | scRNA-seq | 軌跡推斷 | 否 |
| peRegLM | Multiome | 峰值-基因連結 | 否 |
| SOLO | scRNA-seq | 雙細胞檢測 | 半監督 |
