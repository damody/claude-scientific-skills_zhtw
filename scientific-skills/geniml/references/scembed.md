# scEmbed：單細胞嵌入生成

## 概述

scEmbed 在單細胞 ATAC-seq 資料集上訓練 Region2Vec 模型，生成用於聚類和分析的細胞嵌入。它提供一個非監督式機器學習框架，用於在低維空間中表示和分析 scATAC-seq 資料。

## 使用時機

在處理以下情況時使用 scEmbed：
- 需要聚類的單細胞 ATAC-seq（scATAC-seq）資料
- 細胞類型註解任務
- 單細胞染色質可及性的降維
- 與 scanpy 工作流程整合進行下游分析

## 工作流程

### 步驟 1：資料準備

輸入資料必須為 AnnData 格式，`.var` 屬性包含峰的 `chr`、`start` 和 `end` 值。

**從原始資料開始**（barcodes.txt、peaks.bed、matrix.mtx）：

```python
import scanpy as sc
import pandas as pd
import scipy.io
import anndata

# 載入資料
barcodes = pd.read_csv('barcodes.txt', header=None, names=['barcode'])
peaks = pd.read_csv('peaks.bed', sep='\t', header=None,
                    names=['chr', 'start', 'end'])
matrix = scipy.io.mmread('matrix.mtx').tocsr()

# 建立 AnnData
adata = anndata.AnnData(X=matrix.T, obs=barcodes, var=peaks)
adata.write('scatac_data.h5ad')
```

### 步驟 2：預符記化

使用 gtars 公用程式將基因體區域轉換為符記。這會建立包含符記化細胞的 parquet 檔案以加快訓練：

```python
from geniml.io import tokenize_cells

tokenize_cells(
    adata='scatac_data.h5ad',
    universe_file='universe.bed',
    output='tokenized_cells.parquet'
)
```

**預符記化的好處：**
- 更快的訓練迭代
- 減少記憶體需求
- 可重複使用的符記化資料用於多次訓練執行

### 步驟 3：模型訓練

使用符記化資料訓練 scEmbed 模型：

```python
from geniml.scembed import ScEmbed
from geniml.region2vec import Region2VecDataset

# 載入符記化資料集
dataset = Region2VecDataset('tokenized_cells.parquet')

# 初始化和訓練模型
model = ScEmbed(
    embedding_dim=100,
    window_size=5,
    negative_samples=5
)

model.train(
    dataset=dataset,
    epochs=100,
    batch_size=256,
    learning_rate=0.025
)

# 儲存模型
model.save('scembed_model/')
```

### 步驟 4：生成細胞嵌入

使用訓練好的模型為細胞生成嵌入：

```python
from geniml.scembed import ScEmbed

# 載入訓練好的模型
model = ScEmbed.from_pretrained('scembed_model/')

# 為 AnnData 物件生成嵌入
embeddings = model.encode(adata)

# 添加到 AnnData 用於下游分析
adata.obsm['scembed_X'] = embeddings
```

### 步驟 5：下游分析

與 scanpy 整合進行聚類和視覺化：

```python
import scanpy as sc

# 使用 scEmbed 嵌入建立鄰域圖
sc.pp.neighbors(adata, use_rep='scembed_X')

# 聚類細胞
sc.tl.leiden(adata, resolution=0.5)

# 計算 UMAP 用於視覺化
sc.tl.umap(adata)

# 繪製結果
sc.pl.umap(adata, color='leiden')
```

## 關鍵參數

### 訓練參數

| 參數 | 描述 | 典型範圍 |
|------|------|----------|
| `embedding_dim` | 細胞嵌入的維度 | 50 - 200 |
| `window_size` | 訓練的上下文視窗 | 3 - 10 |
| `negative_samples` | 負樣本數量 | 5 - 20 |
| `epochs` | 訓練輪數 | 50 - 200 |
| `batch_size` | 訓練批次大小 | 128 - 512 |
| `learning_rate` | 初始學習率 | 0.01 - 0.05 |

### 符記化參數

- **Universe 檔案**：定義基因體詞彙表的參考 BED 檔案
- **重疊閾值**：峰-universe 匹配的最小重疊（通常 1e-9）

## 預訓練模型

常見參考資料集的預訓練 scEmbed 模型可在 Hugging Face 上取得。使用以下方式載入：

```python
from geniml.scembed import ScEmbed

# 載入預訓練模型
model = ScEmbed.from_pretrained('databio/scembed-pbmc-10k')

# 生成嵌入
embeddings = model.encode(adata)
```

## 最佳實踐

- **資料品質**：使用過濾後的峰-條碼矩陣，而非原始計數
- **預符記化**：始終預符記化以提高訓練效率
- **參數調整**：根據資料集大小調整 `embedding_dim` 和訓練輪數
- **驗證**：使用已知的細胞類型標記驗證聚類品質
- **整合**：與 scanpy 結合進行綜合單細胞分析
- **模型分享**：將訓練好的模型匯出到 Hugging Face 以確保可重現性

## 範例資料集

10x Genomics PBMC 10k 資料集（10,000 個外周血單核細胞）作為標準基準：
- 包含多樣的免疫細胞類型
- 特徵明確的細胞群體
- 可從 10x Genomics 網站取得

## 細胞類型註解

聚類後，使用 k-近鄰（KNN）與參考資料集進行細胞類型註解：

```python
from geniml.scembed import annotate_celltypes

# 使用參考進行註解
annotations = annotate_celltypes(
    query_adata=adata,
    reference_adata=reference,
    embedding_key='scembed_X',
    k=10
)

adata.obs['cell_type'] = annotations
```

## 輸出

scEmbed 產生：
- 低維細胞嵌入（儲存在 `adata.obsm` 中）
- 可重複使用的訓練模型檔案
- 與 scanpy 下游分析相容的格式
- 可選匯出到 Hugging Face 以供分享
