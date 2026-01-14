# 單細胞 RNA-seq 模型

本文檔涵蓋 scvi-tools 中用於分析單細胞 RNA 定序數據的核心模型。

## scVI（Single-Cell Variational Inference，單細胞變分推斷）

**目的**：scRNA-seq 數據的無監督分析、降維和批次校正。

**主要特點**：
- 基於變分自編碼器（VAE）的深度生成模型
- 學習捕捉生物學變異的低維潛在表示
- 自動校正批次效應和技術共變量
- 支援標準化基因表達估計
- 支援差異表達分析

**何時使用**：
- scRNA-seq 數據集的初始探索和降維
- 整合多個批次或研究
- 生成批次校正的表達矩陣
- 執行機率性差異表達分析

**基本用法**：
```python
import scvi

# 設定數據
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch"
)

# 訓練模型
model = scvi.model.SCVI(adata, n_latent=30)
model.train()

# 提取結果
latent = model.get_latent_representation()
normalized = model.get_normalized_expression()
```

**關鍵參數**：
- `n_latent`：潛在空間維度（預設：10）
- `n_layers`：隱藏層數（預設：1）
- `n_hidden`：每個隱藏層的節點數（預設：128）
- `dropout_rate`：神經網路的丟棄率（預設：0.1）
- `dispersion`：基因特定或細胞特定離散度（「gene」或「gene-batch」）
- `gene_likelihood`：數據分佈（「zinb」、「nb」、「poisson」）

**輸出**：
- `get_latent_representation()`：批次校正的低維嵌入
- `get_normalized_expression()`：去噪、標準化的表達值
- `differential_expression()`：組間機率性 DE 檢驗
- `get_feature_correlation_matrix()`：基因-基因相關性估計

## scANVI（Single-Cell ANnotation using Variational Inference，使用變分推斷的單細胞註釋）

**目的**：使用有標籤和無標籤細胞進行半監督細胞類型註釋和整合。

**主要特點**：
- 用細胞類型標籤擴展 scVI
- 利用部分標記的數據集進行註釋遷移
- 同時執行批次校正和細胞類型預測
- 支援查詢到參考的映射

**何時使用**：
- 使用參考標籤註釋新數據集
- 從標註良好的數據集遷移學習到無標籤數據集
- 有標籤和無標籤細胞的聯合分析
- 建立帶有不確定性量化的細胞類型分類器

**基本用法**：
```python
# 選項 1：從頭訓練
scvi.model.SCANVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    labels_key="cell_type",
    unlabeled_category="Unknown"
)
model = scvi.model.SCANVI(adata)
model.train()

# 選項 2：從預訓練的 scVI 初始化
scvi_model = scvi.model.SCVI(adata)
scvi_model.train()
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    unlabeled_category="Unknown"
)
scanvi_model.train()

# 預測細胞類型
predictions = scanvi_model.predict()
```

**關鍵參數**：
- `labels_key`：`adata.obs` 中包含細胞類型標籤的欄位
- `unlabeled_category`：無註釋細胞的標籤
- 所有 scVI 參數也可用

**輸出**：
- `predict()`：所有細胞的細胞類型預測
- `predict_proba()`：預測機率
- `get_latent_representation()`：感知細胞類型的潛在空間

## AUTOZI

**目的**：自動識別和建模 scRNA-seq 數據中的零膨脹基因。

**主要特點**：
- 區分生物學零值和技術丟失
- 學習哪些基因表現出零膨脹
- 提供基因特定的零膨脹機率
- 通過考慮丟失來改善下游分析

**何時使用**：
- 檢測哪些基因受技術丟失影響
- 改善稀疏數據集的插補和標準化
- 理解數據中零膨脹的程度

**基本用法**：
```python
scvi.model.AUTOZI.setup_anndata(adata, layer="counts")
model = scvi.model.AUTOZI(adata)
model.train()

# 獲取每個基因的零膨脹機率
zi_probs = model.get_alphas_betas()
```

## VeloVI

**目的**：使用變分推斷進行 RNA 速度分析。

**主要特點**：
- 聯合建模剪接和未剪接 RNA 計數
- RNA 速度的機率估計
- 考慮技術噪音和批次效應
- 提供速度估計的不確定性量化

**何時使用**：
- 推斷細胞動態和分化軌跡
- 分析剪接/未剪接計數數據
- 帶有批次校正的 RNA 速度分析

**基本用法**：
```python
import scvelo as scv

# 準備速度數據
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)

# 訓練 VeloVI
scvi.model.VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
model = scvi.model.VELOVI(adata)
model.train()

# 獲取速度估計
latent_time = model.get_latent_time()
velocities = model.get_velocity()
```

## contrastiveVI

**目的**：從背景生物學變異中分離擾動特異性變異。

**主要特點**：
- 將共享變異（跨條件通用）與目標特異性變異分開
- 適用於擾動研究（藥物處理、基因擾動）
- 識別條件特異性基因程式
- 支援發現處理特異性效應

**何時使用**：
- 分析擾動實驗（藥物篩選、CRISPR 等）
- 識別特定響應處理的基因
- 將處理效應與背景變異分開
- 比較對照 vs 擾動條件

**基本用法**：
```python
scvi.model.CONTRASTIVEVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    categorical_covariate_keys=["condition"]  # 對照 vs 處理
)

model = scvi.model.CONTRASTIVEVI(
    adata,
    n_latent=10,        # 共享變異
    n_latent_target=5   # 目標特異性變異
)
model.train()

# 提取表示
shared = model.get_latent_representation(representation="shared")
target_specific = model.get_latent_representation(representation="target")
```

## CellAssign

**目的**：使用已知標記基因進行基於標記的細胞類型註釋。

**主要特點**：
- 使用細胞類型的標記基因先驗知識
- 細胞到類型的機率分配
- 處理標記基因重疊和歧義
- 提供帶有不確定性的軟分配

**何時使用**：
- 使用已知標記基因註釋細胞
- 利用現有生物學知識進行分類
- 有標記基因列表但沒有參考數據集的情況

**基本用法**：
```python
# 創建標記基因矩陣（細胞類型 × 基因）
marker_gene_mat = pd.DataFrame({
    "CD4 T cells": [1, 1, 0, 0],  # CD3D, CD4, CD8A, CD19
    "CD8 T cells": [1, 0, 1, 0],
    "B cells": [0, 0, 0, 1]
}, index=["CD3D", "CD4", "CD8A", "CD19"])

scvi.model.CELLASSIGN.setup_anndata(adata, layer="counts")
model = scvi.model.CELLASSIGN(adata, marker_gene_mat)
model.train()

predictions = model.predict()
```

## Solo（雙細胞檢測）

**目的**：識別 scRNA-seq 數據中的雙細胞（包含兩個或多個細胞）。

**主要特點**：
- 使用 scVI 嵌入的半監督雙細胞檢測
- 模擬人工雙細胞用於訓練
- 提供雙細胞機率分數
- 可應用於任何 scVI 模型

**何時使用**：
- scRNA-seq 數據集的品質控制
- 在下游分析前移除雙細胞
- 評估數據中的雙細胞率

**基本用法**：
```python
# 首先訓練 scVI 模型
scvi.model.SCVI.setup_anndata(adata, layer="counts")
scvi_model = scvi.model.SCVI(adata)
scvi_model.train()

# 訓練 Solo 進行雙細胞檢測
solo_model = scvi.external.SOLO.from_scvi_model(scvi_model)
solo_model.train()

# 預測雙細胞
predictions = solo_model.predict()
doublet_scores = predictions["doublet"]
adata.obs["doublet_score"] = doublet_scores
```

## Amortized LDA（主題模型）

**目的**：使用潛在狄利克雷分配（Latent Dirichlet Allocation）進行基因表達的主題建模。

**主要特點**：
- 發現基因表達程式（主題）
- 用於可擴展性的攤銷變分推斷
- 每個細胞是主題的混合
- 每個主題是基因上的分佈

**何時使用**：
- 發現基因程式或表達模組
- 理解表達的組成結構
- 替代性降維方法
- 表達模式的可解釋分解

**基本用法**：
```python
scvi.model.AMORTIZEDLDA.setup_anndata(adata, layer="counts")
model = scvi.model.AMORTIZEDLDA(adata, n_topics=10)
model.train()

# 獲取每個細胞的主題組成
topic_proportions = model.get_latent_representation()

# 獲取每個主題的基因載荷
topic_gene_loadings = model.get_topic_distribution()
```

## 模型選擇指南

**選擇 scVI 當**：
- 從無監督分析開始
- 需要批次校正和整合
- 想要標準化表達和 DE 分析

**選擇 scANVI 當**：
- 有一些有標籤的細胞用於訓練
- 需要細胞類型註釋
- 想要將標籤從參考遷移到查詢

**選擇 AUTOZI 當**：
- 擔心技術丟失
- 需要識別零膨脹基因
- 處理非常稀疏的數據集

**選擇 VeloVI 當**：
- 有剪接/未剪接計數數據
- 對細胞動態感興趣
- 需要帶有批次校正的 RNA 速度

**選擇 contrastiveVI 當**：
- 分析擾動實驗
- 需要分離處理效應
- 想要識別條件特異性程式

**選擇 CellAssign 當**：
- 有標記基因列表可用
- 想要基於標記的機率註釋
- 沒有參考數據集可用

**選擇 Solo 當**：
- 需要雙細胞檢測
- 已經使用 scVI 進行分析
- 想要機率性雙細胞分數
