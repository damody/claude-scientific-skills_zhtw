# Region2Vec：基因體區域嵌入

## 概述

Region2Vec 從 BED 檔案生成基因體區域和區域集的非監督式嵌入。它將基因體區域對應到詞彙表，透過連接建立句子，並應用 word2vec 訓練來學習有意義的表示。

## 使用時機

在處理以下情況時使用 Region2Vec：
- 需要降維的 BED 檔案集合
- 基因體區域相似性分析
- 需要區域特徵向量的下游 ML 任務
- 跨多個基因體資料集的比較分析

## 工作流程

### 步驟 1：準備資料

在來源資料夾中收集 BED 檔案。可選擇指定檔案列表（預設使用目錄中的所有檔案）。準備 universe 檔案作為符記化的參考詞彙表。

### 步驟 2：符記化

執行硬符記化將基因體區域轉換為符記：

```python
from geniml.tokenization import hard_tokenization

src_folder = '/path/to/raw/bed/files'
dst_folder = '/path/to/tokenized_files'
universe_file = '/path/to/universe_file.bed'

hard_tokenization(src_folder, dst_folder, universe_file, 1e-9)
```

最後一個參數（1e-9）是符記化重疊顯著性的 p 值閾值。

### 步驟 3：訓練 Region2Vec 模型

在符記化檔案上執行 Region2Vec 訓練：

```python
from geniml.region2vec import region2vec

region2vec(
    token_folder=dst_folder,
    save_dir='./region2vec_model',
    num_shufflings=1000,
    embedding_dim=100,
    context_len=50,
    window_size=5,
    init_lr=0.025
)
```

## 關鍵參數

| 參數 | 描述 | 典型範圍 |
|------|------|----------|
| `init_lr` | 初始學習率 | 0.01 - 0.05 |
| `window_size` | 上下文視窗大小 | 3 - 10 |
| `num_shufflings` | 洗牌迭代次數 | 500 - 2000 |
| `embedding_dim` | 輸出嵌入的維度 | 50 - 300 |
| `context_len` | 訓練的上下文長度 | 30 - 100 |

## CLI 使用

```bash
geniml region2vec --token-folder /path/to/tokens \
  --save-dir ./region2vec_model \
  --num-shuffle 1000 \
  --embed-dim 100 \
  --context-len 50 \
  --window-size 5 \
  --init-lr 0.025
```

## 最佳實踐

- **參數調整**：經常調整 `init_lr`、`window_size`、`num_shufflings` 和 `embedding_dim` 以在您的特定資料集上獲得最佳效能
- **Universe 檔案**：使用涵蓋分析中所有感興趣區域的綜合 universe 檔案
- **驗證**：在進行訓練之前始終驗證符記化輸出
- **資源**：訓練可能計算密集；對於大型資料集監控記憶體使用

## 輸出

訓練好的模型儲存可用於以下目的的嵌入：
- 跨基因體區域的相似性搜尋
- 區域集聚類
- 下游 ML 任務的特徵向量
- 透過降維進行視覺化（t-SNE、UMAP）
