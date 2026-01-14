# 空間轉錄組學模型

本文檔涵蓋 scvi-tools 中用於分析空間解析轉錄組學數據的模型。

## DestVI（Deconvolution of Spatial Transcriptomics using Variational Inference，使用變分推斷進行空間轉錄組學反卷積）

**目的**：使用單細胞參考數據進行空間轉錄組學的多解析度反卷積。

**主要特點**：
- 估計每個空間位置的細胞類型比例
- 使用單細胞 RNA-seq 參考進行反卷積
- 多解析度方法（全局和局部模式）
- 考慮空間相關性
- 提供不確定性量化

**何時使用**：
- 反卷積 Visium 或類似的空間轉錄組學
- 擁有帶有細胞類型標籤的 scRNA-seq 參考數據
- 想要將細胞類型映射到空間位置
- 對細胞類型的空間組織感興趣
- 需要細胞類型豐度的機率估計

**數據要求**：
- **空間數據**：Visium 或類似的基於點的測量（目標數據）
- **單細胞參考**：帶有細胞類型註釋的 scRNA-seq
- 兩個數據集應共享基因

**基本用法**：
```python
import scvi

# 步驟 1：在單細胞參考上訓練 scVI
scvi.model.SCVI.setup_anndata(sc_adata, layer="counts")
sc_model = scvi.model.SCVI(sc_adata)
sc_model.train()

# 步驟 2：設定空間數據
scvi.model.DESTVI.setup_anndata(
    spatial_adata,
    layer="counts"
)

# 步驟 3：使用參考訓練 DestVI
model = scvi.model.DESTVI.from_rna_model(
    spatial_adata,
    sc_model,
    cell_type_key="cell_type"  # 參考中的細胞類型標籤
)
model.train(max_epochs=2500)

# 步驟 4：獲取細胞類型比例
proportions = model.get_proportions()
spatial_adata.obsm["proportions"] = proportions

# 步驟 5：獲取細胞類型特異性表達
# 每個點上特定細胞類型的基因表達
ct_expression = model.get_scale_for_ct("T cells")
```

**關鍵參數**：
- `amortization`：攤銷策略（「both」、「latent」、「proportion」）
- `n_latent`：潛在維度（從 scVI 模型繼承）

**輸出**：
- `get_proportions()`：每個點的細胞類型比例
- `get_scale_for_ct(cell_type)`：細胞類型特異性表達模式
- `get_gamma()`：比例特異性基因表達縮放

**視覺化**：
```python
import scanpy as sc
import matplotlib.pyplot as plt

# 空間視覺化特定細胞類型比例
sc.pl.spatial(
    spatial_adata,
    color="T cells",  # 如果比例已添加到 .obs
    spot_size=150
)

# 或直接使用 obsm
for ct in cell_types:
    plt.figure()
    sc.pl.spatial(
        spatial_adata,
        color=spatial_adata.obsm["proportions"][ct],
        title=f"{ct} 比例"
    )
```

## Stereoscope

**目的**：使用機率建模進行空間轉錄組學的細胞類型反卷積。

**主要特點**：
- 基於參考的反卷積
- 細胞類型比例的機率框架
- 適用於各種空間技術
- 處理基因選擇和標準化

**何時使用**：
- 類似於 DestVI 但更簡單的方法
- 使用參考進行空間數據反卷積
- 基本反卷積的更快替代方案

**基本用法**：
```python
scvi.model.STEREOSCOPE.setup_anndata(
    sc_adata,
    labels_key="cell_type",
    layer="counts"
)

# 在參考上訓練
ref_model = scvi.model.STEREOSCOPE(sc_adata)
ref_model.train()

# 設定空間數據
scvi.model.STEREOSCOPE.setup_anndata(spatial_adata, layer="counts")

# 遷移到空間
spatial_model = scvi.model.STEREOSCOPE.from_reference_model(
    spatial_adata,
    ref_model
)
spatial_model.train()

# 獲取比例
proportions = spatial_model.get_proportions()
```

## Tangram

**目的**：單細胞數據到空間位置的空間映射和整合。

**主要特點**：
- 將單細胞映射到空間座標
- 學習單細胞和空間數據之間的最優傳輸
- 空間位置的基因插補
- 細胞類型映射

**何時使用**：
- 將 scRNA-seq 細胞映射到空間位置
- 插補空間數據中未測量的基因
- 以單細胞解析度理解空間組織
- 整合 scRNA-seq 和空間轉錄組學

**數據要求**：
- 帶有註釋的單細胞 RNA-seq 數據
- 空間轉錄組學數據
- 模態之間的共享基因

**基本用法**：
```python
import tangram as tg

# 將細胞映射到空間位置
ad_map = tg.map_cells_to_space(
    adata_sc=sc_adata,
    adata_sp=spatial_adata,
    mode="cells",  # 或 "clusters" 用於細胞類型映射
    density_prior="rna_count_based"
)

# 獲取映射矩陣（細胞 × 點）
mapping = ad_map.X

# 將細胞註釋投射到空間
tg.project_cell_annotations(
    ad_map,
    spatial_adata,
    annotation="cell_type"
)

# 在空間數據中插補基因
genes_to_impute = ["CD3D", "CD8A", "CD4"]
tg.project_genes(ad_map, spatial_adata, genes=genes_to_impute)
```

**視覺化**：
```python
# 視覺化細胞類型映射
sc.pl.spatial(
    spatial_adata,
    color="cell_type_projected",
    spot_size=100
)
```

## gimVI（Gaussian Identity Multivi for Imputation，用於插補的高斯身份 MultiVI）

**目的**：空間和單細胞數據之間的跨模態插補。

**主要特點**：
- 空間和單細胞數據的聯合模型
- 插補空間數據中缺失的基因
- 支援跨數據集查詢
- 學習共享表示

**何時使用**：
- 插補空間數據中未測量的基因
- 空間和單細胞數據集的聯合分析
- 模態之間的映射

**基本用法**：
```python
# 合併數據集
combined_adata = sc.concat([sc_adata, spatial_adata])

scvi.model.GIMVI.setup_anndata(
    combined_adata,
    layer="counts"
)

model = scvi.model.GIMVI(combined_adata)
model.train()

# 插補空間數據中的基因
imputed = model.get_imputed_values(spatial_indices)
```

## scVIVA（Variation in Variational Autoencoders for Spatial，空間變分自編碼器中的變異）

**目的**：分析空間數據中的細胞-環境關係。

**主要特點**：
- 建模細胞鄰域和環境
- 識別環境相關的基因表達
- 考慮空間相關結構
- 細胞-細胞相互作用分析

**何時使用**：
- 理解空間背景如何影響細胞
- 識別生態位特異性基因程式
- 細胞-細胞相互作用研究
- 微環境分析

**數據要求**：
- 帶有座標的空間轉錄組學
- 細胞類型註釋（可選）

**基本用法**：
```python
scvi.model.SCVIVA.setup_anndata(
    spatial_adata,
    layer="counts",
    spatial_key="spatial"  # .obsm 中的座標
)

model = scvi.model.SCVIVA(spatial_adata)
model.train()

# 獲取環境表示
env_latent = model.get_environment_representation()

# 識別環境相關基因
env_genes = model.get_environment_specific_genes()
```

## ResolVI

**目的**：通過解析度感知建模處理空間轉錄組學噪音。

**主要特點**：
- 考慮空間解析度效應
- 空間數據去噪
- 多尺度分析
- 改善下游分析品質

**何時使用**：
- 噪音空間數據
- 多種空間解析度
- 分析前需要去噪
- 改善數據品質

**基本用法**：
```python
scvi.model.RESOLVI.setup_anndata(
    spatial_adata,
    layer="counts",
    spatial_key="spatial"
)

model = scvi.model.RESOLVI(spatial_adata)
model.train()

# 獲取去噪表達
denoised = model.get_denoised_expression()
```

## 空間轉錄組學模型選擇

### DestVI
**選擇情況**：
- 需要使用參考進行詳細反卷積
- 擁有高品質 scRNA-seq 參考
- 想要多解析度分析
- 需要不確定性量化

**最適合**：Visium、基於點的技術

### Stereoscope
**選擇情況**：
- 需要更簡單、更快的反卷積
- 基本細胞類型比例估計
- 計算資源有限

**最適合**：快速反卷積任務

### Tangram
**選擇情況**：
- 想要單細胞解析度映射
- 需要插補許多基因
- 對細胞定位感興趣
- 偏好最優傳輸方法

**最適合**：詳細空間映射

### gimVI
**選擇情況**：
- 需要雙向插補
- 空間和單細胞的聯合建模
- 跨數據集查詢

**最適合**：整合和插補

### scVIVA
**選擇情況**：
- 對細胞環境感興趣
- 細胞-細胞相互作用分析
- 鄰域效應

**最適合**：微環境研究

### ResolVI
**選擇情況**：
- 數據品質是問題
- 需要去噪
- 多尺度分析

**最適合**：噪音數據預處理

## 完整工作流程：使用 DestVI 進行空間反卷積

```python
import scvi
import scanpy as sc
import squidpy as sq

# ===== 第一部分：準備單細胞參考 =====
# 載入並處理 scRNA-seq 參考
sc_adata = sc.read_h5ad("reference_scrna.h5ad")

# QC 和過濾
sc.pp.filter_genes(sc_adata, min_cells=10)
sc.pp.highly_variable_genes(sc_adata, n_top_genes=4000)

# 在參考上訓練 scVI
scvi.model.SCVI.setup_anndata(
    sc_adata,
    layer="counts",
    batch_key="batch"
)

sc_model = scvi.model.SCVI(sc_adata)
sc_model.train(max_epochs=400)

# ===== 第二部分：載入空間數據 =====
spatial_adata = sc.read_visium("path/to/visium")
spatial_adata.var_names_make_unique()

# 空間數據 QC
sc.pp.filter_genes(spatial_adata, min_cells=10)

# ===== 第三部分：執行 DestVI =====
scvi.model.DESTVI.setup_anndata(
    spatial_adata,
    layer="counts"
)

destvi_model = scvi.model.DESTVI.from_rna_model(
    spatial_adata,
    sc_model,
    cell_type_key="cell_type"
)

destvi_model.train(max_epochs=2500)

# ===== 第四部分：提取結果 =====
# 獲取比例
proportions = destvi_model.get_proportions()
spatial_adata.obsm["proportions"] = proportions

# 將比例添加到 .obs 以便繪圖
for i, ct in enumerate(sc_model.adata.obs["cell_type"].cat.categories):
    spatial_adata.obs[f"prop_{ct}"] = proportions[:, i]

# ===== 第五部分：視覺化 =====
# 繪製特定細胞類型
cell_types = ["T cells", "B cells", "Macrophages"]

for ct in cell_types:
    sc.pl.spatial(
        spatial_adata,
        color=f"prop_{ct}",
        title=f"{ct} 比例",
        spot_size=150,
        cmap="viridis"
    )

# ===== 第六部分：空間分析 =====
# 計算空間鄰居
sq.gr.spatial_neighbors(spatial_adata)

# 細胞類型的空間自相關
for ct in cell_types:
    sq.gr.spatial_autocorr(
        spatial_adata,
        attr="obs",
        mode="moran",
        genes=[f"prop_{ct}"]
    )

# ===== 第七部分：儲存結果 =====
destvi_model.save("destvi_model")
spatial_adata.write("spatial_deconvolved.h5ad")
```

## 空間分析最佳實踐

1. **參考品質**：使用高品質、標註良好的 scRNA-seq 參考
2. **基因重疊**：確保參考和空間之間有足夠的共享基因
3. **空間座標**：正確在 `.obsm["spatial"]` 中註冊空間座標
4. **驗證**：使用已知標記基因驗證反卷積
5. **視覺化**：始終在空間上視覺化結果以檢查生物學合理性
6. **細胞類型粒度**：考慮適當的細胞類型解析度
7. **計算資源**：空間模型可能佔用大量記憶體
8. **品質控制**：分析前過濾低品質點
