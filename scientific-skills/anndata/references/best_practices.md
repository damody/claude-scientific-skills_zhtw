# 最佳實務

高效且有效使用 AnnData 的指南。

## 記憶體管理

### 對稀疏資料使用稀疏矩陣
```python
import numpy as np
from scipy.sparse import csr_matrix
import anndata as ad

# 檢查資料稀疏性
data = np.random.rand(1000, 2000)
sparsity = 1 - np.count_nonzero(data) / data.size
print(f"稀疏性：{sparsity:.2%}")

# 如果 >50% 為零則轉換為稀疏
if sparsity > 0.5:
    adata = ad.AnnData(X=csr_matrix(data))
else:
    adata = ad.AnnData(X=data)

# 好處：對稀疏基因體學資料減少 10-100 倍記憶體
```

### 將字串轉換為類別
```python
# 低效：字串欄位使用大量記憶體
adata.obs['cell_type'] = ['Type_A', 'Type_B', 'Type_C'] * 333 + ['Type_A']

# 高效：轉換為類別
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

# 轉換所有字串欄位
adata.strings_to_categoricals()

# 好處：對重複字串減少 10-50 倍記憶體
```

### 對大型資料集使用 backed 模式
```python
# 不要將整個資料集載入記憶體
adata = ad.read_h5ad('large_dataset.h5ad', backed='r')

# 處理中繼資料
filtered = adata[adata.obs['quality'] > 0.8]

# 僅載入過濾後的子集
adata_subset = filtered.to_memory()

# 好處：處理比 RAM 更大的資料集
```

## 視圖 vs 複製

### 理解視圖
```python
# 預設情況下子集會建立視圖
subset = adata[0:100, :]
print(subset.is_view)  # True

# 視圖不會複製資料（記憶體高效）
# 但修改可能影響原始資料

# 檢查物件是否為視圖
if adata.is_view:
    adata = adata.copy()  # 使其獨立
```

### 何時使用視圖
```python
# 好：對子集進行唯讀操作
mean_expr = adata[adata.obs['cell_type'] == 'T cell'].X.mean()

# 好：臨時分析
temp_subset = adata[:100, :]
result = analyze(temp_subset.X)
```

### 何時使用複製
```python
# 為修改建立獨立複製
adata_filtered = adata[keep_cells, :].copy()

# 可安全修改而不影響原始資料
adata_filtered.obs['new_column'] = values

# 始終在以下情況複製：
# - 儲存子集供以後使用
# - 修改子集資料
# - 傳遞給會修改資料的函式
```

## 資料儲存最佳實務

### 選擇正確的格式

**H5AD（HDF5）- 預設選擇**
```python
adata.write_h5ad('data.h5ad', compression='gzip')
```
- 快速隨機存取
- 支援 backed 模式
- 良好的壓縮
- 最適合：大多數使用案例

**Zarr - 雲端和並行存取**
```python
adata.write_zarr('data.zarr', chunks=(100, 100))
```
- 非常適合雲端儲存（S3、GCS）
- 支援並行 I/O
- 良好的壓縮
- 最適合：大型資料集、雲端工作流程、並行處理

**CSV - 互通性**
```python
adata.write_csvs('output_dir/')
```
- 人類可讀
- 與所有工具相容
- 檔案大、速度慢
- 最適合：與非 Python 工具共享、小型資料集

### 最佳化檔案大小
```python
# 儲存前最佳化：

# 1. 如適當則轉換為稀疏
from scipy.sparse import csr_matrix, issparse
if not issparse(adata.X):
    density = np.count_nonzero(adata.X) / adata.X.size
    if density < 0.5:
        adata.X = csr_matrix(adata.X)

# 2. 將字串轉換為類別
adata.strings_to_categoricals()

# 3. 使用壓縮
adata.write_h5ad('data.h5ad', compression='gzip', compression_opts=9)

# 典型結果：檔案大小減少 5-20 倍
```

## Backed 模式策略

### 唯讀分析
```python
# 以唯讀 backed 模式開啟
adata = ad.read_h5ad('data.h5ad', backed='r')

# 執行過濾而不載入資料
high_quality = adata[adata.obs['quality_score'] > 0.8]

# 僅載入過濾後的資料
adata_filtered = high_quality.to_memory()
```

### 讀寫修改
```python
# 以讀寫 backed 模式開啟
adata = ad.read_h5ad('data.h5ad', backed='r+')

# 修改中繼資料（寫入磁碟）
adata.obs['new_annotation'] = values

# X 保留在磁碟上，修改立即儲存
```

### 分塊處理
```python
# 分塊處理大型資料集
adata = ad.read_h5ad('huge_dataset.h5ad', backed='r')

results = []
chunk_size = 1000

for i in range(0, adata.n_obs, chunk_size):
    chunk = adata[i:i+chunk_size, :].to_memory()
    result = process(chunk)
    results.append(result)

final_result = combine(results)
```

## 效能最佳化

### 子集效能
```python
# 快速：使用陣列的布林索引
mask = np.array(adata.obs['quality'] > 0.5)
subset = adata[mask, :]

# 慢：使用 Series 的布林索引（建立視圖鏈）
subset = adata[adata.obs['quality'] > 0.5, :]

# 最快：整數索引
indices = np.where(adata.obs['quality'] > 0.5)[0]
subset = adata[indices, :]
```

### 避免重複子集
```python
# 低效：多次子集操作
for cell_type in ['A', 'B', 'C']:
    subset = adata[adata.obs['cell_type'] == cell_type]
    process(subset)

# 高效：分組並處理
groups = adata.obs.groupby('cell_type').groups
for cell_type, indices in groups.items():
    subset = adata[indices, :]
    process(subset)
```

### 對大型矩陣使用分塊操作
```python
# 分塊處理 X
for chunk in adata.chunked_X(chunk_size=1000):
    result = compute(chunk)

# 比載入完整 X 更節省記憶體
```

## 處理原始資料

### 過濾前儲存原始資料
```python
# 包含所有基因的原始資料
adata = ad.AnnData(X=counts)

# 過濾前儲存原始資料
adata.raw = adata.copy()

# 過濾至高變異基因
adata = adata[:, adata.var['highly_variable']]

# 稍後：存取原始資料
original_expression = adata.raw.X
all_genes = adata.raw.var_names
```

### 何時使用 raw
```python
# 使用 raw 的情況：
# - 對過濾基因進行差異表達分析
# - 視覺化不在過濾集中的特定基因
# - 標準化後存取原始計數

# 存取原始資料
if adata.raw is not None:
    gene_expr = adata.raw[:, 'GENE_NAME'].X
else:
    gene_expr = adata[:, 'GENE_NAME'].X
```

## 中繼資料管理

### 命名慣例
```python
# 一致的命名提高可用性

# 觀測中繼資料（obs）：
# - cell_id, sample_id
# - cell_type, tissue, condition
# - n_genes, n_counts, percent_mito
# - cluster, leiden, louvain

# 變數中繼資料（var）：
# - gene_id, gene_name
# - highly_variable, n_cells
# - mean_expression, dispersion

# 嵌入（obsm）：
# - X_pca, X_umap, X_tsne
# - X_diffmap, X_draw_graph_fr

# 遵循 scanpy/scverse 生態系統的慣例
```

### 記錄中繼資料
```python
# 在 uns 中儲存中繼資料描述
adata.uns['metadata_descriptions'] = {
    'cell_type': '來自自動聚類的細胞類型註解',
    'quality_score': '來自 scrublet 的 QC 分數（0-1，越高越好）',
    'batch': '實驗批次識別碼'
}

# 儲存處理歷史
adata.uns['processing_steps'] = [
    '從 10X 載入原始計數',
    '過濾：n_genes > 200, n_counts < 50000',
    '標準化至每細胞 10000 計數',
    '對數轉換'
]
```

## 可重現性

### 設定隨機種子
```python
import numpy as np

# 設定種子以獲得可重現結果
np.random.seed(42)

# 在 uns 中記錄
adata.uns['random_seed'] = 42
```

### 儲存參數
```python
# 在 uns 中儲存分析參數
adata.uns['pca'] = {
    'n_comps': 50,
    'svd_solver': 'arpack',
    'random_state': 42
}

adata.uns['neighbors'] = {
    'n_neighbors': 15,
    'n_pcs': 50,
    'metric': 'euclidean',
    'method': 'umap'
}
```

### 版本追蹤
```python
import anndata
import scanpy
import numpy

# 儲存版本
adata.uns['versions'] = {
    'anndata': anndata.__version__,
    'scanpy': scanpy.__version__,
    'numpy': numpy.__version__,
    'python': sys.version
}
```

## 錯誤處理

### 檢查資料有效性
```python
# 驗證維度
assert adata.n_obs == len(adata.obs)
assert adata.n_vars == len(adata.var)
assert adata.X.shape == (adata.n_obs, adata.n_vars)

# 檢查 NaN 值
has_nan = np.isnan(adata.X.data).any() if issparse(adata.X) else np.isnan(adata.X).any()
if has_nan:
    print("警告：資料包含 NaN 值")

# 檢查負值（如預期為計數）
has_negative = (adata.X.data < 0).any() if issparse(adata.X) else (adata.X < 0).any()
if has_negative:
    print("警告：資料包含負值")
```

### 驗證中繼資料
```python
# 檢查遺失值
missing_obs = adata.obs.isnull().sum()
if missing_obs.any():
    print("obs 中的遺失值：")
    print(missing_obs[missing_obs > 0])

# 驗證索引唯一
assert adata.obs_names.is_unique, "觀測名稱不唯一"
assert adata.var_names.is_unique, "變數名稱不唯一"

# 檢查中繼資料對齊
assert len(adata.obs) == adata.n_obs
assert len(adata.var) == adata.n_vars
```

## 與其他工具整合

### Scanpy 整合
```python
import scanpy as sc

# AnnData 是 scanpy 的原生格式
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
```

### Pandas 整合
```python
import pandas as pd

# 轉換為 DataFrame
df = adata.to_df()

# 從 DataFrame 建立
adata = ad.AnnData(df)

# 將中繼資料作為 DataFrames 處理
adata.obs = adata.obs.merge(external_metadata, left_index=True, right_index=True)
```

### PyTorch 整合
```python
from anndata.experimental import AnnLoader

# 建立 PyTorch DataLoader
dataloader = AnnLoader(adata, batch_size=128, shuffle=True)

# 在訓練迴圈中迭代
for batch in dataloader:
    X = batch.X
    # 對批次訓練模型
```

## 常見陷阱

### 陷阱 1：修改視圖
```python
# 錯誤：修改視圖可能影響原始資料
subset = adata[:100, :]
subset.X = new_data  # 可能修改 adata.X！

# 正確：修改前複製
subset = adata[:100, :].copy()
subset.X = new_data  # 獨立複製
```

### 陷阱 2：索引未對齊
```python
# 錯誤：假設順序匹配
external_data = pd.read_csv('data.csv')
adata.obs['new_col'] = external_data['values']  # 可能未對齊！

# 正確：在索引上對齊
adata.obs['new_col'] = external_data.set_index('cell_id').loc[adata.obs_names, 'values']
```

### 陷阱 3：混合稀疏和密集
```python
# 錯誤：將稀疏轉換為密集使用大量記憶體
result = adata.X + 1  # 將稀疏轉換為密集！

# 正確：使用稀疏操作
from scipy.sparse import issparse
if issparse(adata.X):
    result = adata.X.copy()
    result.data += 1
```

### 陷阱 4：未處理視圖
```python
# 錯誤：假設子集是獨立的
subset = adata[mask, :]
del adata  # 子集可能變得無效！

# 正確：需要時複製
subset = adata[mask, :].copy()
del adata  # 子集保持有效
```

### 陷阱 5：忽略記憶體限制
```python
# 錯誤：將大型資料集載入記憶體
adata = ad.read_h5ad('100GB_file.h5ad')  # OOM 錯誤！

# 正確：使用 backed 模式
adata = ad.read_h5ad('100GB_file.h5ad', backed='r')
subset = adata[adata.obs['keep']].to_memory()
```

## 工作流程範例

完整的最佳實務工作流程：

```python
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix

# 1. 如果大型則使用 backed 模式載入
adata = ad.read_h5ad('data.h5ad', backed='r')

# 2. 不載入資料快速檢查中繼資料
print(f"資料集：{adata.n_obs} 個細胞 × {adata.n_vars} 個基因")

# 3. 基於中繼資料過濾
high_quality = adata[adata.obs['quality_score'] > 0.8]

# 4. 將過濾後的子集載入記憶體
adata = high_quality.to_memory()

# 5. 轉換為最佳儲存類型
adata.strings_to_categoricals()
if not issparse(adata.X):
    density = np.count_nonzero(adata.X) / adata.X.size
    if density < 0.5:
        adata.X = csr_matrix(adata.X)

# 6. 過濾基因前儲存原始資料
adata.raw = adata.copy()

# 7. 過濾至高變異基因
adata = adata[:, adata.var['highly_variable']].copy()

# 8. 記錄處理
adata.uns['processing'] = {
    'filtered': 'quality_score > 0.8',
    'n_hvg': adata.n_vars,
    'date': '2025-11-03'
}

# 9. 最佳化儲存
adata.write_h5ad('processed.h5ad', compression='gzip')
```
