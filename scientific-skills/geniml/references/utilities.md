# Geniml 公用程式和附加工具

## BBClient：BED 檔案快取

### 概述

BBClient 提供從遠端來源高效快取 BED 檔案的功能，實現更快的重複存取和與 R 工作流程的整合。

### 使用時機

在以下情況使用 BBClient：
- 從遠端資料庫重複存取 BED 檔案
- 使用 BEDbase 儲存庫
- 將基因體資料與 R 管線整合
- 需要本地快取以提高效能

### Python 使用

```python
from geniml.bbclient import BBClient

# 初始化客戶端
client = BBClient(cache_folder='~/.bedcache')

# 取得並快取 BED 檔案
bed_file = client.load_bed(bed_id='GSM123456')

# 存取快取的檔案
regions = client.get_regions('GSM123456')
```

### R 整合

```r
library(reticulate)
geniml <- import("geniml.bbclient")

# 初始化客戶端
client <- geniml$BBClient(cache_folder='~/.bedcache')

# 載入 BED 檔案
bed_file <- client$load_bed(bed_id='GSM123456')
```

### 最佳實踐

- 設定具有足夠儲存空間的快取目錄
- 在分析中使用一致的快取位置
- 定期清理快取以移除未使用的檔案

---

## BEDshift：BED 檔案隨機化

### 概述

BEDshift 提供在保留基因體情境的同時隨機化 BED 檔案的工具，對於生成零分布和統計測試至關重要。

### 使用時機

在以下情況使用 BEDshift：
- 建立統計測試的零模型
- 生成對照資料集
- 評估基因體重疊的顯著性
- 基準測試分析方法

### 使用

```python
from geniml.bedshift import bedshift

# 隨機化 BED 檔案，保留染色體分布
randomized = bedshift(
    input_bed='peaks.bed',
    genome='hg38',
    preserve_chrom=True,
    n_iterations=100
)
```

### CLI 使用

```bash
geniml bedshift \
  --input peaks.bed \
  --genome hg38 \
  --preserve-chrom \
  --iterations 100 \
  --output randomized_peaks.bed
```

### 隨機化策略

**保留染色體分布：**
```python
bedshift(input_bed, genome, preserve_chrom=True)
```
維持區域在與原始相同的染色體上。

**保留距離分布：**
```python
bedshift(input_bed, genome, preserve_distance=True)
```
維持區域間距離。

**保留區域大小：**
```python
bedshift(input_bed, genome, preserve_size=True)
```
保持原始區域長度。

### 最佳實踐

- 選擇與零假設匹配的隨機化策略
- 生成多次迭代以獲得穩健的統計
- 驗證隨機化輸出維持所需屬性
- 記錄隨機化參數以確保可重現性

---

## Evaluation：模型評估工具

### 概述

Geniml 提供評估嵌入品質和模型效能的評估公用程式。

### 使用時機

在以下情況使用評估工具：
- 驗證訓練好的嵌入
- 比較不同模型
- 評估聚類品質
- 發表模型結果

### 嵌入評估

```python
from geniml.evaluation import evaluate_embeddings

# 評估 Region2Vec 嵌入
metrics = evaluate_embeddings(
    embeddings_file='region2vec_model/embeddings.npy',
    labels_file='metadata.csv',
    metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz']
)

print(f"輪廓係數：{metrics['silhouette']:.3f}")
print(f"Davies-Bouldin 指數：{metrics['davies_bouldin']:.3f}")
```

### 聚類指標

**輪廓係數：** 衡量聚類凝聚性和分離度（-1 到 1，越高越好）

**Davies-Bouldin 指數：** 聚類間的平均相似度（≥0，越低越好）

**Calinski-Harabasz 分數：** 聚類間/聚類內離散度比率（越高越好）

### scEmbed 細胞類型註解評估

```python
from geniml.evaluation import evaluate_annotation

# 評估細胞類型預測
results = evaluate_annotation(
    predicted=adata.obs['predicted_celltype'],
    true=adata.obs['true_celltype'],
    metrics=['accuracy', 'f1', 'confusion_matrix']
)

print(f"準確率：{results['accuracy']:.1%}")
print(f"F1 分數：{results['f1']:.3f}")
```

### 最佳實踐

- 使用多個互補指標
- 與基準模型比較
- 在保留的測試資料上報告指標
- 在指標旁邊視覺化嵌入（UMAP/t-SNE）

---

## Tokenization：區域符記化公用程式

### 概述

符記化使用參考 universe 將基因體區域轉換為離散符記，實現 word2vec 風格的訓練。

### 使用時機

符記化是以下操作的必要預處理步驟：
- Region2Vec 訓練
- scEmbed 模型訓練
- 任何需要離散符記的嵌入方法

### 硬符記化

嚴格的基於重疊的符記化：

```python
from geniml.tokenization import hard_tokenization

hard_tokenization(
    src_folder='bed_files/',
    dst_folder='tokenized/',
    universe_file='universe.bed',
    p_value_threshold=1e-9
)
```

**參數：**
- `p_value_threshold`：重疊的顯著性水準（通常 1e-9 或 1e-6）

### 軟符記化

允許部分匹配的機率符記化：

```python
from geniml.tokenization import soft_tokenization

soft_tokenization(
    src_folder='bed_files/',
    dst_folder='tokenized/',
    universe_file='universe.bed',
    overlap_threshold=0.5
)
```

**參數：**
- `overlap_threshold`：最小重疊比例（0-1）

### 基於 Universe 的符記化

使用自訂參數將區域對應到 universe 符記：

```python
from geniml.tokenization import universe_tokenization

universe_tokenization(
    bed_file='peaks.bed',
    universe_file='universe.bed',
    output_file='tokens.txt',
    method='hard',
    threshold=1e-9
)
```

### 最佳實踐

- **Universe 品質**：使用綜合、建構良好的 universes
- **閾值選擇**：更嚴格（較低 p 值）以獲得更高信心
- **驗證**：檢查符記化覆蓋率（多少比例的區域被符記化）
- **一致性**：在相關分析中使用相同的 universe 和參數

### 符記化覆蓋率

檢查區域符記化的效果：

```python
from geniml.tokenization import check_coverage

coverage = check_coverage(
    bed_file='peaks.bed',
    universe_file='universe.bed',
    threshold=1e-9
)

print(f"符記化覆蓋率：{coverage:.1%}")
```

為可靠的訓練，目標是 >80% 的覆蓋率。

---

## Text2BedNN：搜尋後端

### 概述

Text2BedNN 建立基於神經網路的搜尋後端，用於使用自然語言或元資料查詢基因體區域。

### 使用時機

在以下情況使用 Text2BedNN：
- 為基因體資料庫建構搜尋介面
- 在 BED 檔案上啟用自然語言查詢
- 建立元資料感知的搜尋系統
- 部署互動式基因體搜尋應用程式

### 工作流程

**步驟 1：準備嵌入**

使用元資料訓練 BEDspace 或 Region2Vec 模型。

**步驟 2：建構搜尋索引**

```python
from geniml.search import build_search_index

build_search_index(
    embeddings_file='bedspace_model/embeddings.npy',
    metadata_file='metadata.csv',
    output_dir='search_backend/'
)
```

**步驟 3：查詢索引**

```python
from geniml.search import SearchBackend

backend = SearchBackend.load('search_backend/')

# 自然語言查詢
results = backend.query(
    text="T cell regulatory regions",
    top_k=10
)

# 元資料查詢
results = backend.query(
    metadata={'cell_type': 'T_cell', 'tissue': 'blood'},
    top_k=10
)
```

### 最佳實踐

- 使用豐富的元資料訓練嵌入以獲得更好的搜尋
- 索引大型集合以獲得全面的覆蓋
- 在已知查詢上驗證搜尋相關性
- 使用 API 部署以實現互動式應用程式

---

## 附加工具

### I/O 公用程式

```python
from geniml.io import read_bed, write_bed, load_universe

# 讀取 BED 檔案
regions = read_bed('peaks.bed')

# 寫入 BED 檔案
write_bed(regions, 'output.bed')

# 載入 universe
universe = load_universe('universe.bed')
```

### 模型公用程式

```python
from geniml.models import save_model, load_model

# 儲存訓練好的模型
save_model(model, 'my_model/')

# 載入模型
model = load_model('my_model/')
```

### 常見模式

**管線工作流程：**
```python
# 1. 建構 universe
universe = build_universe(coverage_folder='coverage/', method='cc', cutoff=5)

# 2. 符記化
hard_tokenization(src_folder='beds/', dst_folder='tokens/',
                   universe_file='universe.bed', p_value_threshold=1e-9)

# 3. 訓練嵌入
region2vec(token_folder='tokens/', save_dir='model/', num_shufflings=1000)

# 4. 評估
metrics = evaluate_embeddings(embeddings_file='model/embeddings.npy',
                               labels_file='metadata.csv')
```

這種模組化設計允許靈活組合 geniml 工具以實現多樣化的基因體 ML 工作流程。
