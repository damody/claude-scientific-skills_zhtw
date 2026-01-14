# CZ CELLxGENE Census 資料結構描述參考

## 概述

CZ CELLxGENE Census 是建立在 TileDB-SOMA 框架上的單細胞資料版本化集合。本參考文件記錄資料結構、可用的元資料欄位和查詢語法。

## 高階結構

Census 組織為一個 `SOMACollection`，包含兩個主要組件：

### 1. census_info
摘要資訊包含：
- **summary**：建置日期、細胞計數、資料集統計
- **datasets**：來自 CELLxGENE Discover 的所有資料集及其元資料
- **summary_cell_counts**：按元資料類別分層的細胞計數

### 2. census_data
物種特定的 `SOMAExperiment` 物件：
- **"homo_sapiens"**：人類單細胞資料
- **"mus_musculus"**：小鼠單細胞資料

## 每個物種的資料結構

每個物種實驗包含：

### obs（細胞元資料）
儲存為 `SOMADataFrame` 的細胞層級註釋。存取方式：
```python
census["census_data"]["homo_sapiens"].obs
```

### ms["RNA"]（測量）
RNA 測量資料包含：
- **X**：具有層級的資料矩陣：
  - `raw`：原始計數資料
  - `normalized`：（如果可用）標準化計數
- **var**：基因元資料
- **feature_dataset_presence_matrix**：稀疏布林陣列，顯示每個資料集中測量了哪些基因

## 細胞元資料欄位（obs）

### 必要/核心欄位

**身份與資料集：**
- `soma_joinid`：用於連接的唯一整數識別碼
- `dataset_id`：來源資料集識別碼
- `is_primary_data`：布林標誌（True = 唯一細胞，False = 跨資料集重複）

**細胞類型：**
- `cell_type`：人類可讀的細胞類型名稱
- `cell_type_ontology_term_id`：標準化本體論術語（例如 "CL:0000236"）

**組織：**
- `tissue`：特定組織名稱
- `tissue_general`：更廣泛的組織類別（適用於分組）
- `tissue_ontology_term_id`：標準化本體論術語

**測序技術：**
- `assay`：使用的定序技術
- `assay_ontology_term_id`：標準化本體論術語

**疾病：**
- `disease`：疾病狀態或病況
- `disease_ontology_term_id`：標準化本體論術語

**捐贈者：**
- `donor_id`：唯一捐贈者識別碼
- `sex`：生物性別（male、female、unknown）
- `self_reported_ethnicity`：族裔資訊
- `development_stage`：生命階段（adult、child、embryonic 等）
- `development_stage_ontology_term_id`：標準化本體論術語

**生物體：**
- `organism`：學名（Homo sapiens、Mus musculus）
- `organism_ontology_term_id`：標準化本體論術語

**技術：**
- `suspension_type`：樣品製備類型（cell、nucleus、na）

## 基因元資料欄位（var）

存取方式：
```python
census["census_data"]["homo_sapiens"].ms["RNA"].var
```

**可用欄位：**
- `soma_joinid`：用於連接的唯一整數識別碼
- `feature_id`：Ensembl 基因 ID（例如 "ENSG00000161798"）
- `feature_name`：基因符號（例如 "FOXP2"）
- `feature_length`：基因長度，以鹼基對為單位

## 值篩選語法

查詢使用類似 Python 的表達式進行篩選。語法由 TileDB-SOMA 處理。

### 比較運算子
- `==`：等於
- `!=`：不等於
- `<`、`>`、`<=`、`>=`：數值比較
- `in`：成員資格測試（例如 `feature_id in ['ENSG00000161798', 'ENSG00000188229']`）

### 邏輯運算子
- `and`、`&`：邏輯 AND
- `or`、`|`：邏輯 OR

### 範例

**單一條件：**
```python
value_filter="cell_type == 'B cell'"
```

**使用 AND 的多個條件：**
```python
value_filter="cell_type == 'B cell' and tissue_general == 'lung' and is_primary_data == True"
```

**使用 IN 匹配多個值：**
```python
value_filter="tissue in ['lung', 'liver', 'kidney']"
```

**複雜條件：**
```python
value_filter="(cell_type == 'neuron' or cell_type == 'astrocyte') and disease != 'normal'"
```

**篩選基因：**
```python
var_value_filter="feature_name in ['CD4', 'CD8A', 'CD19']"
```

## 資料納入標準

Census 包含來自 CZ CELLxGENE Discover 符合以下條件的所有資料：

1. **物種**：人類（*Homo sapiens*）或小鼠（*Mus musculus*）
2. **技術**：經批准的 RNA 定序技術
3. **計數類型**：僅原始計數（無處理過/僅標準化的資料）
4. **元資料**：遵循 CELLxGENE 結構描述進行標準化
5. **空間和非空間資料**：包含傳統和空間轉錄組學

## 重要資料特性

### 重複細胞
細胞可能出現在多個資料集中。在大多數分析中使用 `is_primary_data == True` 篩選唯一細胞。

### 計數類型
Census 包含：
- **分子計數**：來自基於 UMI 的方法
- **全基因定序讀數計數**：來自非 UMI 方法
這些可能需要不同的標準化方法。

### 版本控制
Census 版本有版本編號（例如 "2023-07-25"、"stable"）。始終指定版本以進行可重現的分析：
```python
census = cellxgene_census.open_soma(census_version="2023-07-25")
```

## 資料集存在矩陣

存取每個資料集中測量了哪些基因：
```python
presence_matrix = census["census_data"]["homo_sapiens"].ms["RNA"]["feature_dataset_presence_matrix"]
```

這個稀疏布林矩陣有助於了解：
- 跨資料集的基因覆蓋率
- 特定基因分析應包含哪些資料集
- 與基因覆蓋率相關的技術批次效應

## SOMA 物件類型

使用的核心 TileDB-SOMA 物件：
- **DataFrame**：表格資料（obs、var）
- **SparseNDArray**：稀疏矩陣（X 層、存在矩陣）
- **DenseNDArray**：密集陣列（較不常見）
- **Collection**：相關物件的容器
- **Experiment**：測量的頂層容器
- **SOMAScene**：空間轉錄組學場景
- **obs_spatial_presence**：空間資料可用性
