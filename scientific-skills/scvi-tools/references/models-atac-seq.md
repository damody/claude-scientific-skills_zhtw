# ATAC-seq 與染色質可及性模型

本文檔涵蓋 scvi-tools 中用於分析單細胞 ATAC-seq 和染色質可及性數據的模型。

## PeakVI

**目的**：使用峰值計數分析和整合單細胞 ATAC-seq 數據。

**主要特點**：
- 專為 scATAC-seq 峰值數據設計的變分自編碼器（VAE）
- 學習染色質可及性的低維表示
- 跨樣本執行批次校正
- 支援差異可及性檢驗
- 整合多個 ATAC-seq 數據集

**何時使用**：
- 分析 scATAC-seq 峰值計數矩陣
- 整合多個 ATAC-seq 實驗
- 染色質可及性數據的批次校正
- ATAC-seq 的降維
- 細胞類型或條件之間的差異可及性分析

**數據要求**：
- 峰值計數矩陣（細胞 × 峰值）
- 峰值可及性的二元或計數數據
- 批次/樣本註釋（可選，用於批次校正）

**基本用法**：
```python
import scvi

# 準備數據（峰值應在 adata.X 中）
# 可選：過濾峰值
sc.pp.filter_genes(adata, min_cells=3)

# 設定數據
scvi.model.PEAKVI.setup_anndata(
    adata,
    batch_key="batch"
)

# 訓練模型
model = scvi.model.PEAKVI(adata)
model.train()

# 獲取潛在表示（已批次校正）
latent = model.get_latent_representation()
adata.obsm["X_PeakVI"] = latent

# 差異可及性
da_results = model.differential_accessibility(
    groupby="cell_type",
    group1="TypeA",
    group2="TypeB"
)
```

**關鍵參數**：
- `n_latent`：潛在空間維度（預設：10）
- `n_hidden`：每個隱藏層的節點數（預設：128）
- `n_layers`：隱藏層數（預設：1）
- `region_factors`：是否學習區域特定因子（預設：True）
- `latent_distribution`：潛在空間的分佈（「normal」或「ln」）

**輸出**：
- `get_latent_representation()`：細胞的低維嵌入
- `get_accessibility_estimates()`：標準化可及性值
- `differential_accessibility()`：差異峰值的統計檢驗
- `get_region_factors()`：峰值特定的縮放因子

**最佳實踐**：
1. 過濾低品質峰值（僅存在於極少數細胞中）
2. 如果整合多個樣本，請包含批次資訊
3. 使用潛在表示進行聚類和 UMAP 視覺化
4. 對於具有高技術變異的數據集，考慮使用 `region_factors=True`
5. 將潛在嵌入存儲在 `adata.obsm` 中，以便使用 scanpy 進行下游分析

## PoissonVI

**目的**：scATAC-seq 片段計數的定量分析（比峰值計數更詳細）。

**主要特點**：
- 直接建模片段計數（不僅僅是峰值存在/不存在）
- 計數數據使用泊松分佈
- 捕捉可及性的定量差異
- 實現對染色質狀態的精細分析

**何時使用**：
- 分析片段級別的 ATAC-seq 數據
- 需要定量可及性測量
- 比二元峰值調用更高解析度的分析
- 研究染色質可及性的漸變變化

**數據要求**：
- 片段計數矩陣（細胞 × 基因組區域）
- 計數數據（非二元）

**基本用法**：
```python
scvi.model.POISSONVI.setup_anndata(
    adata,
    batch_key="batch"
)

model = scvi.model.POISSONVI(adata)
model.train()

# 獲取結果
latent = model.get_latent_representation()
accessibility = model.get_accessibility_estimates()
```

**與 PeakVI 的主要區別**：
- **PeakVI**：最適合標準峰值計數矩陣，更快
- **PoissonVI**：最適合定量片段計數，更詳細

**何時選擇 PoissonVI 而不是 PeakVI**：
- 使用片段計數而非調用的峰值
- 需要捕捉定量差異
- 擁有高品質、高覆蓋率的數據
- 對微妙的可及性變化感興趣

## scBasset

**目的**：具有可解釋性和基序分析的深度學習方法用於 scATAC-seq 分析。

**主要特點**：
- 用於基於序列分析的卷積神經網路（CNN）架構
- 建模原始 DNA 序列，而不僅僅是峰值計數
- 支援基序發現和轉錄因子（TF）結合預測
- 提供可解釋的特徵重要性
- 執行批次校正

**何時使用**：
- 想要整合 DNA 序列資訊
- 對 TF 基序分析感興趣
- 需要可解釋的模型（哪些序列驅動可及性）
- 分析調控元件和 TF 結合位點
- 僅從序列預測可及性

**數據要求**：
- 峰值序列（從基因組提取）
- 峰值可及性矩陣
- 基因組參考（用於序列提取）

**基本用法**：
```python
# scBasset 需要序列資訊
# 首先，提取峰值的序列
from scbasset import utils
sequences = utils.fetch_sequences(adata, genome="hg38")

# 設定並訓練
scvi.model.SCBASSET.setup_anndata(
    adata,
    batch_key="batch"
)

model = scvi.model.SCBASSET(adata, sequences=sequences)
model.train()

# 獲取潛在表示
latent = model.get_latent_representation()

# 解釋模型：哪些序列/基序是重要的
importance_scores = model.get_feature_importance()
```

**關鍵參數**：
- `n_latent`：潛在空間維度
- `conv_layers`：卷積層數
- `n_filters`：每個卷積層的濾波器數
- `filter_size`：卷積濾波器大小

**進階功能**：
- **電腦模擬突變分析**：預測序列變化如何影響可及性
- **基序富集**：識別可及區域中富集的 TF 基序
- **批次校正**：類似於其他 scvi-tools 模型
- **遷移學習**：在新數據集上微調

**可解釋性工具**：
```python
# 獲取序列的重要性分數
importance = model.get_sequence_importance(region_indices=[0, 1, 2])

# 預測新序列的可及性
predictions = model.predict_accessibility(new_sequences)
```

## ATAC-seq 模型選擇

### PeakVI
**選擇情況**：
- 標準 scATAC-seq 分析工作流程
- 擁有峰值計數矩陣（最常見格式）
- 需要快速、高效的批次校正
- 想要直接的差異可及性分析
- 優先考慮計算效率

**優勢**：
- 訓練和推斷快速
- scATAC-seq 的成熟方法
- 易於與 scanpy 工作流程整合
- 穩健的批次校正

### PoissonVI
**選擇情況**：
- 擁有片段級別的計數數據
- 需要定量可及性測量
- 對微妙差異感興趣
- 擁有高覆蓋率、高品質數據

**優勢**：
- 更詳細的定量資訊
- 更適合漸變變化
- 適當的計數統計模型

### scBasset
**選擇情況**：
- 想要整合 DNA 序列
- 需要生物學解釋（基序、TF）
- 對調控機制感興趣
- 擁有 CNN 訓練的計算資源
- 想要對新序列的預測能力

**優勢**：
- 基於序列，生物學上可解釋
- 內建基序和 TF 分析
- 預測建模能力
- 電腦模擬擾動實驗

## 工作流程範例：完整的 ATAC-seq 分析

```python
import scvi
import scanpy as sc

# 1. 載入並預處理 ATAC-seq 數據
adata = sc.read_h5ad("atac_data.h5ad")

# 2. 過濾低品質峰值
sc.pp.filter_genes(adata, min_cells=10)

# 3. 設定並訓練 PeakVI
scvi.model.PEAKVI.setup_anndata(
    adata,
    batch_key="sample"
)

model = scvi.model.PEAKVI(adata, n_latent=20)
model.train(max_epochs=400)

# 4. 提取潛在表示
latent = model.get_latent_representation()
adata.obsm["X_PeakVI"] = latent

# 5. 下游分析
sc.pp.neighbors(adata, use_rep="X_PeakVI")
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters")

# 6. 差異可及性
da_results = model.differential_accessibility(
    groupby="clusters",
    group1="0",
    group2="1"
)

# 7. 儲存模型
model.save("peakvi_model")
```

## 與基因表達的整合（RNA+ATAC）

對於配對的多模態數據（來自同一細胞的 RNA+ATAC），請改用 **MultiVI**：

```python
# 對於 10x Multiome 或類似的配對數據
scvi.model.MULTIVI.setup_anndata(
    adata,
    batch_key="sample",
    modality_key="modality"  # "RNA" 或 "ATAC"
)

model = scvi.model.MULTIVI(adata)
model.train()

# 獲取聯合潛在空間
latent = model.get_latent_representation()
```

有關多模態整合的更多詳細資訊，請參見 `models-multimodal.md`。

## ATAC-seq 分析最佳實踐

1. **品質控制**：
   - 過濾峰值計數非常低或非常高的細胞
   - 移除僅存在於極少數細胞中的峰值
   - 如需要，過濾粒線體和性染色體峰值

2. **批次校正**：
   - 如果整合多個樣本，始終包含 `batch_key`
   - 考慮技術共變量（定序深度、TSS 富集）

3. **特徵選擇**：
   - 與 RNA-seq 不同，通常使用所有峰值
   - 考慮過濾非常罕見的峰值以提高效率

4. **潛在維度**：
   - 根據數據集複雜度從 `n_latent=10-30` 開始
   - 對於更異質的數據集使用較大值

5. **下游分析**：
   - 使用潛在表示進行聚類和視覺化
   - 將峰值連結到基因進行調控分析
   - 對聚類特異性峰值進行基序富集分析

6. **計算考量**：
   - ATAC-seq 矩陣通常非常大（許多峰值）
   - 考慮對峰值進行下採樣以進行初步探索
   - 對大數據集使用 GPU 加速
