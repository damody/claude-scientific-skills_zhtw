# BEDspace：區域和元資料的聯合嵌入

## 概述

BEDspace 將 StarSpace 模型應用於基因體資料，能夠同時訓練區域集和其元資料標籤的數值嵌入在共享的低維空間中。這允許跨區域和元資料進行豐富的查詢。

## 使用時機

在處理以下情況時使用 BEDspace：
- 具有相關元資料的區域集（細胞類型、組織、條件）
- 需要元資料感知相似性的搜尋任務
- 跨模態查詢（例如：「查找與標籤 X 相似的區域」）
- 基因體內容和實驗條件的聯合分析

## 工作流程

BEDspace 由四個順序操作組成：

### 1. 預處理

格式化基因體區間和元資料以進行 StarSpace 訓練：

```bash
geniml bedspace preprocess \
  --input /path/to/regions/ \
  --metadata labels.csv \
  --universe universe.bed \
  --labels "cell_type,tissue" \
  --output preprocessed.txt
```

**必要檔案：**
- **輸入資料夾**：包含 BED 檔案的目錄
- **元資料 CSV**：必須包含與 BED 檔案名稱匹配的 `file_name` 欄位，加上元資料欄位
- **Universe 檔案**：用於符記化的參考 BED 檔案
- **標籤**：逗號分隔的元資料欄位列表

預處理步驟為元資料添加 `__label__` 前綴，並將區域轉換為 StarSpace 相容格式。

### 2. 訓練

在預處理資料上執行 StarSpace 模型：

```bash
geniml bedspace train \
  --path-to-starspace /path/to/starspace \
  --input preprocessed.txt \
  --output model/ \
  --dim 100 \
  --epochs 50 \
  --lr 0.05
```

**主要訓練參數：**
- `--dim`：嵌入維度（典型值：50-200）
- `--epochs`：訓練輪數（典型值：20-100）
- `--lr`：學習率（典型值：0.01-0.1）

### 3. 距離

計算區域集和元資料標籤之間的距離指標：

```bash
geniml bedspace distances \
  --input model/ \
  --metadata labels.csv \
  --universe universe.bed \
  --output distances.pkl
```

此步驟建立相似性搜尋所需的距離矩陣。

### 4. 搜尋

在三種場景中擷取相似項目：

**區域到標籤（r2l）**：查詢區域集 → 擷取相似的元資料標籤
```bash
geniml bedspace search -t r2l -d distances.pkl -q query_regions.bed -n 10
```

**標籤到區域（l2r）**：查詢元資料標籤 → 擷取相似的區域集
```bash
geniml bedspace search -t l2r -d distances.pkl -q "T_cell" -n 10
```

**區域到區域（r2r）**：查詢區域集 → 擷取相似的區域集
```bash
geniml bedspace search -t r2r -d distances.pkl -q query_regions.bed -n 10
```

`-n` 參數控制回傳的結果數量。

## Python API

```python
from geniml.bedspace import BEDSpaceModel

# 載入訓練好的模型
model = BEDSpaceModel.load('model/')

# 查詢相似項目
results = model.search(
    query="T_cell",
    search_type="l2r",
    top_k=10
)
```

## 最佳實踐

- **元資料結構**：確保元資料 CSV 包含與 BED 檔案名稱完全匹配的 `file_name` 欄位（不含路徑）
- **標籤選擇**：選擇能捕捉感興趣生物變異的資訊性元資料欄位
- **Universe 一致性**：在預處理、距離計算和任何後續分析中使用相同的 universe 檔案
- **驗證**：在投入訓練之前預處理並檢查輸出格式
- **StarSpace 安裝**：StarSpace 需要單獨安裝，因為它是外部依賴項

## 輸出解釋

搜尋結果回傳在聯合嵌入空間中按相似性排序的項目：
- **r2l**：識別描述您查詢區域的元資料標籤
- **l2r**：查找與您的元資料標準匹配的區域集
- **r2r**：發現具有相似基因體內容的區域集

## 需求

BEDspace 需要單獨安裝 StarSpace。從以下網址下載：https://github.com/facebookresearch/StarSpace
