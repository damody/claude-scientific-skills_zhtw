# 常見查詢模式和最佳實務

## 查詢模式類別

### 1. 探索性查詢（僅元資料）

用於探索可用資料而不載入表現矩陣。

**模式：取得組織中的唯一細胞類型**
```python
import cellxgene_census

with cellxgene_census.open_soma() as census:
    cell_metadata = cellxgene_census.get_obs(
        census,
        "homo_sapiens",
        value_filter="tissue_general == 'brain' and is_primary_data == True",
        column_names=["cell_type"]
    )
    unique_cell_types = cell_metadata["cell_type"].unique()
    print(f"發現 {len(unique_cell_types)} 種唯一細胞類型")
```

**模式：按條件計算細胞數量**
```python
cell_metadata = cellxgene_census.get_obs(
    census,
    "homo_sapiens",
    value_filter="disease != 'normal' and is_primary_data == True",
    column_names=["disease", "tissue_general"]
)
counts = cell_metadata.groupby(["disease", "tissue_general"]).size()
```

**模式：探索資料集資訊**
```python
# 存取資料集表格
datasets = census["census_info"]["datasets"].read().concat().to_pandas()

# 篩選特定條件
covid_datasets = datasets[datasets["disease"].str.contains("COVID", na=False)]
```

### 2. 小到中等查詢（AnnData）

當結果可放入記憶體時（通常少於 100k 個細胞）使用 `get_anndata()`。

**模式：組織特定的細胞類型查詢**
```python
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="cell_type == 'B cell' and tissue_general == 'lung' and is_primary_data == True",
    obs_column_names=["assay", "disease", "sex", "donor_id"],
)
```

**模式：多基因的基因特定查詢**
```python
marker_genes = ["CD4", "CD8A", "CD19", "FOXP3"]

# 首先取得基因 ID
gene_metadata = cellxgene_census.get_var(
    census, "homo_sapiens",
    value_filter=f"feature_name in {marker_genes}",
    column_names=["feature_id", "feature_name"]
)
gene_ids = gene_metadata["feature_id"].tolist()

# 使用基因篩選進行查詢
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    var_value_filter=f"feature_id in {gene_ids}",
    obs_value_filter="cell_type == 'T cell' and is_primary_data == True",
)
```

**模式：多組織查詢**
```python
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="tissue_general in ['lung', 'liver', 'kidney'] and is_primary_data == True",
    obs_column_names=["cell_type", "tissue_general", "dataset_id"],
)
```

**模式：疾病特定查詢**
```python
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="disease == 'COVID-19' and tissue_general == 'lung' and is_primary_data == True",
)
```

### 3. 大型查詢（核外處理）

對超出可用 RAM 的查詢使用 `axis_query()`。

**模式：迭代處理**
```python
import pyarrow as pa

# 建立查詢
query = census["census_data"]["homo_sapiens"].axis_query(
    measurement_name="RNA",
    obs_query=soma.AxisQuery(
        value_filter="tissue_general == 'brain' and is_primary_data == True"
    ),
    var_query=soma.AxisQuery(
        value_filter="feature_name in ['FOXP2', 'TBR1', 'SATB2']"
    )
)

# 分塊迭代 X 矩陣
iterator = query.X("raw").tables()
for batch in iterator:
    # 處理批次（一個 pyarrow.Table）
    # 批次包含欄位：soma_data、soma_dim_0、soma_dim_1
    process_batch(batch)
```

**模式：增量統計（平均值/變異數）**
```python
# 使用 Welford 的線上演算法
n = 0
mean = 0
M2 = 0

iterator = query.X("raw").tables()
for batch in iterator:
    values = batch["soma_data"].to_numpy()
    for x in values:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2

variance = M2 / (n - 1) if n > 1 else 0
```

### 4. PyTorch 整合（機器學習）

使用 `experiment_dataloader()` 訓練模型。

**模式：建立訓練資料載入器**
```python
from cellxgene_census.experimental.ml import experiment_dataloader
import torch

with cellxgene_census.open_soma() as census:
    # 建立資料載入器
    dataloader = experiment_dataloader(
        census["census_data"]["homo_sapiens"],
        measurement_name="RNA",
        X_name="raw",
        obs_value_filter="tissue_general == 'liver' and is_primary_data == True",
        obs_column_names=["cell_type"],
        batch_size=128,
        shuffle=True,
    )

    # 訓練迴圈
    for epoch in range(num_epochs):
        for batch in dataloader:
            X = batch["X"]  # 基因表現
            labels = batch["obs"]["cell_type"]  # 細胞類型標籤
            # 訓練模型...
```

**模式：訓練/測試分割**
```python
from cellxgene_census.experimental.ml import ExperimentDataset

# 從查詢建立資料集
dataset = ExperimentDataset(
    experiment_axis_query,
    layer_name="raw",
    obs_column_names=["cell_type"],
    batch_size=128,
)

# 分割資料
train_dataset, test_dataset = dataset.random_split(
    split=[0.8, 0.2],
    seed=42
)

# 建立載入器
train_loader = experiment_dataloader(train_dataset)
test_loader = experiment_dataloader(test_dataset)
```

### 5. 整合工作流程

**模式：Scanpy 整合**
```python
import scanpy as sc

# 載入資料
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="cell_type == 'neuron' and is_primary_data == True",
)

# 標準 scanpy 工作流程
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["cell_type", "tissue_general"])
```

**模式：多資料集整合**
```python
# 分別查詢多個資料集
datasets_to_integrate = ["dataset_id_1", "dataset_id_2", "dataset_id_3"]

adatas = []
for dataset_id in datasets_to_integrate:
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=f"dataset_id == '{dataset_id}' and is_primary_data == True",
    )
    adatas.append(adata)

# 使用 scanorama、harmony 或其他工具整合
import scanpy.external as sce
sce.pp.scanorama_integrate(adatas)
```

## 最佳實務

### 1. 始終篩選主要資料
除非特別分析重複項，否則始終包含 `is_primary_data == True`：
```python
obs_value_filter="cell_type == 'B cell' and is_primary_data == True"
```

### 2. 指定 Census 版本
對於可重現的分析，始終指定 Census 版本：
```python
census = cellxgene_census.open_soma(census_version="2023-07-25")
```

### 3. 使用上下文管理器
始終使用上下文管理器以確保正確清理：
```python
with cellxgene_census.open_soma() as census:
    # 您的程式碼
```

### 4. 只選擇所需的欄位
透過只選擇所需的元資料欄位來最小化資料傳輸：
```python
obs_column_names=["cell_type", "tissue_general", "disease"]  # 不是所有欄位
```

### 5. 檢查基因查詢的資料集存在性
分析特定基因時，檢查哪些資料集測量了它們：
```python
presence = cellxgene_census.get_presence_matrix(
    census,
    "homo_sapiens",
    var_value_filter="feature_name in ['CD4', 'CD8A']"
)
```

### 6. 使用 tissue_general 進行更廣泛的查詢
`tissue_general` 提供比 `tissue` 更粗糙的分組，適用於跨組織分析：
```python
# 更適合廣泛查詢
obs_value_filter="tissue_general == 'immune system'"

# 需要時使用特定組織
obs_value_filter="tissue == 'peripheral blood mononuclear cell'"
```

### 7. 將元資料探索與表現查詢結合
先探索元資料以了解可用資料，然後查詢表現：
```python
# 步驟 1：探索
metadata = cellxgene_census.get_obs(
    census, "homo_sapiens",
    value_filter="disease == 'COVID-19'",
    column_names=["cell_type", "tissue_general"]
)
print(metadata.value_counts())

# 步驟 2：根據發現進行查詢
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="disease == 'COVID-19' and cell_type == 'T cell' and is_primary_data == True",
)
```

### 8. 大型查詢的記憶體管理
對於大型查詢，載入前先檢查估計大小：
```python
# 先取得細胞計數
metadata = cellxgene_census.get_obs(
    census, "homo_sapiens",
    value_filter="tissue_general == 'brain' and is_primary_data == True",
    column_names=["soma_joinid"]
)
n_cells = len(metadata)
print(f"查詢將回傳 {n_cells} 個細胞")

# 如果太大，使用核外處理或進一步篩選
```

### 9. 利用本體論術語確保一致性
盡可能使用本體論術語 ID 而非自由文字：
```python
# 比跨資料集使用 cell_type == 'B cell' 更可靠
obs_value_filter="cell_type_ontology_term_id == 'CL:0000236'"
```

### 10. 批次處理模式
對於跨多個條件的系統性分析：
```python
tissues = ["lung", "liver", "kidney", "heart"]
results = {}

for tissue in tissues:
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter=f"tissue_general == '{tissue}' and is_primary_data == True",
    )
    # 執行分析
    results[tissue] = analyze(adata)
```

## 應避免的常見陷阱

1. **未篩選 is_primary_data**：導致重複計算細胞
2. **載入過多資料**：先使用元資料查詢估計大小
3. **未使用上下文管理器**：可能導致資源洩漏
4. **版本控制不一致**：不指定版本會導致結果不可重現
5. **查詢過於寬泛**：從集中的查詢開始，根據需要擴展
6. **忽略資料集存在性**：某些基因並非在所有資料集中都有測量
7. **計數標準化錯誤**：注意 UMI 與讀數計數的差異
