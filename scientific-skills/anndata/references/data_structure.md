# AnnData 物件結構

AnnData 物件儲存帶有相關註解的資料矩陣，為管理實驗資料和中繼資料提供靈活的框架。

## 核心組件

### X（資料矩陣）
形狀為 (n_obs, n_vars) 的主要資料矩陣，儲存實驗測量值。

```python
import anndata as ad
import numpy as np

# 使用密集陣列建立
adata = ad.AnnData(X=np.random.rand(100, 2000))

# 使用稀疏矩陣建立（推薦用於大型稀疏資料）
from scipy.sparse import csr_matrix
sparse_data = csr_matrix(np.random.rand(100, 2000))
adata = ad.AnnData(X=sparse_data)
```

存取資料：
```python
# 完整矩陣（大型資料集需謹慎）
full_data = adata.X

# 單個觀測值
obs_data = adata.X[0, :]

# 所有觀測值的單個變數
var_data = adata.X[:, 0]
```

### obs（觀測註解）
儲存觀測值（列）中繼資料的 DataFrame。每列對應 X 中的一個觀測值。

```python
import pandas as pd

# 建立附帶觀測中繼資料的 AnnData
obs_df = pd.DataFrame({
    'cell_type': ['T cell', 'B cell', 'Monocyte'],
    'treatment': ['control', 'treated', 'control'],
    'timepoint': [0, 24, 24]
}, index=['cell_1', 'cell_2', 'cell_3'])

adata = ad.AnnData(X=np.random.rand(3, 100), obs=obs_df)

# 存取觀測中繼資料
print(adata.obs['cell_type'])
print(adata.obs.loc['cell_1'])
```

### var（變數註解）
儲存變數（欄）中繼資料的 DataFrame。每列對應 X 中的一個變數。

```python
# 建立附帶變數中繼資料的 AnnData
var_df = pd.DataFrame({
    'gene_name': ['ACTB', 'GAPDH', 'TP53'],
    'chromosome': ['7', '12', '17'],
    'highly_variable': [True, False, True]
}, index=['ENSG00001', 'ENSG00002', 'ENSG00003'])

adata = ad.AnnData(X=np.random.rand(100, 3), var=var_df)

# 存取變數中繼資料
print(adata.var['gene_name'])
print(adata.var.loc['ENSG00001'])
```

### layers（替代資料表示）
儲存與 X 具有相同維度的替代矩陣的字典。

```python
# 儲存原始計數、標準化資料和縮放資料
adata = ad.AnnData(X=np.random.rand(100, 2000))
adata.layers['raw_counts'] = np.random.randint(0, 100, (100, 2000))
adata.layers['normalized'] = adata.X / np.sum(adata.X, axis=1, keepdims=True)
adata.layers['scaled'] = (adata.X - adata.X.mean()) / adata.X.std()

# 存取圖層
raw_data = adata.layers['raw_counts']
normalized_data = adata.layers['normalized']
```

常見圖層用途：
- `raw_counts`：標準化前的原始計數資料
- `normalized`：對數標準化或 TPM 值
- `scaled`：用於分析的 Z 分數值
- `imputed`：填補後的資料

### obsm（多維觀測註解）
儲存與觀測值對齊的多維陣列的字典。

```python
# 儲存 PCA 座標和 UMAP 嵌入
adata.obsm['X_pca'] = np.random.rand(100, 50)  # 50 個主成分
adata.obsm['X_umap'] = np.random.rand(100, 2)  # 2D UMAP 座標
adata.obsm['X_tsne'] = np.random.rand(100, 2)  # 2D t-SNE 座標

# 存取嵌入
pca_coords = adata.obsm['X_pca']
umap_coords = adata.obsm['X_umap']
```

常見 obsm 用途：
- `X_pca`：主成分座標
- `X_umap`：UMAP 嵌入座標
- `X_tsne`：t-SNE 嵌入座標
- `X_diffmap`：擴散圖座標
- `protein_expression`：蛋白質豐度測量值（CITE-seq）

### varm（多維變數註解）
儲存與變數對齊的多維陣列的字典。

```python
# 儲存 PCA 載荷
adata.varm['PCs'] = np.random.rand(2000, 50)  # 50 個成分的載荷
adata.varm['gene_modules'] = np.random.rand(2000, 10)  # 基因模組分數

# 存取載荷
pc_loadings = adata.varm['PCs']
```

常見 varm 用途：
- `PCs`：主成分載荷
- `gene_modules`：基因共表達模組分配

### obsp（成對觀測關係）
儲存表示觀測值之間關係的稀疏矩陣的字典。

```python
from scipy.sparse import csr_matrix

# 儲存 k 最近鄰圖
n_obs = 100
knn_graph = csr_matrix(np.random.rand(n_obs, n_obs) > 0.95)
adata.obsp['connectivities'] = knn_graph
adata.obsp['distances'] = csr_matrix(np.random.rand(n_obs, n_obs))

# 存取圖
knn_connections = adata.obsp['connectivities']
distances = adata.obsp['distances']
```

常見 obsp 用途：
- `connectivities`：細胞-細胞鄰域圖
- `distances`：細胞之間的成對距離

### varp（成對變數關係）
儲存表示變數之間關係的稀疏矩陣的字典。

```python
# 儲存基因-基因相關矩陣
n_vars = 2000
gene_corr = csr_matrix(np.random.rand(n_vars, n_vars) > 0.99)
adata.varp['correlations'] = gene_corr

# 存取相關性
gene_correlations = adata.varp['correlations']
```

### uns（非結構化註解）
儲存任意非結構化中繼資料的字典。

```python
# 儲存分析參數和結果
adata.uns['experiment_date'] = '2025-11-03'
adata.uns['pca'] = {
    'variance_ratio': [0.15, 0.10, 0.08],
    'params': {'n_comps': 50}
}
adata.uns['neighbors'] = {
    'params': {'n_neighbors': 15, 'method': 'umap'},
    'connectivities_key': 'connectivities'
}

# 存取非結構化資料
exp_date = adata.uns['experiment_date']
pca_params = adata.uns['pca']['params']
```

常見 uns 用途：
- 分析參數和設定
- 繪圖用色彩調色板
- 群集資訊
- 工具特定中繼資料

### raw（原始資料快照）
保留過濾前原始資料矩陣和變數註解的可選屬性。

```python
# 建立 AnnData 並儲存原始狀態
adata = ad.AnnData(X=np.random.rand(100, 5000))
adata.var['gene_name'] = [f'Gene_{i}' for i in range(5000)]

# 過濾前儲存原始狀態
adata.raw = adata.copy()

# 過濾至高變異基因
highly_variable_mask = np.random.rand(5000) > 0.5
adata = adata[:, highly_variable_mask]

# 存取原始資料
original_matrix = adata.raw.X
original_var = adata.raw.var
```

## 物件屬性

```python
# 維度
n_observations = adata.n_obs
n_variables = adata.n_vars
shape = adata.shape  # (n_obs, n_vars)

# 索引資訊
obs_names = adata.obs_names  # 觀測識別碼
var_names = adata.var_names  # 變數識別碼

# 儲存模式
is_view = adata.is_view  # 如果這是另一個物件的視圖則為 True
is_backed = adata.isbacked  # 如果由磁碟儲存支援則為 True
filename = adata.filename  # 支援檔案的路徑（如果是 backed）
```

## 建立 AnnData 物件

### 從陣列和 DataFrames
```python
import anndata as ad
import numpy as np
import pandas as pd

# 最小建立
X = np.random.rand(100, 2000)
adata = ad.AnnData(X)

# 附帶中繼資料
obs = pd.DataFrame({'cell_type': ['A', 'B'] * 50}, index=[f'cell_{i}' for i in range(100)])
var = pd.DataFrame({'gene_name': [f'Gene_{i}' for i in range(2000)]}, index=[f'ENSG{i:05d}' for i in range(2000)])
adata = ad.AnnData(X=X, obs=obs, var=var)

# 附帶所有組件
adata = ad.AnnData(
    X=X,
    obs=obs,
    var=var,
    layers={'raw': np.random.randint(0, 100, (100, 2000))},
    obsm={'X_pca': np.random.rand(100, 50)},
    uns={'experiment': 'test'}
)
```

### 從 DataFrame
```python
# 從 pandas DataFrame 建立（基因為欄，細胞為列）
df = pd.DataFrame(
    np.random.rand(100, 50),
    columns=[f'Gene_{i}' for i in range(50)],
    index=[f'Cell_{i}' for i in range(100)]
)
adata = ad.AnnData(df)
```

## 資料存取模式

### 向量擷取
```python
# 取得觀測註解為陣列
cell_types = adata.obs_vector('cell_type')

# 取得跨觀測值的變數值
gene_expression = adata.obs_vector('ACTB')  # 如果 ACTB 在 var_names 中

# 取得變數註解為陣列
gene_names = adata.var_vector('gene_name')
```

### 子集
```python
# 按索引
subset = adata[0:10, 0:100]  # 前 10 個觀測值，前 100 個變數

# 按名稱
subset = adata[['cell_1', 'cell_2'], ['ACTB', 'GAPDH']]

# 按布林遮罩
high_count_cells = adata.obs['total_counts'] > 1000
subset = adata[high_count_cells, :]

# 按觀測中繼資料
t_cells = adata[adata.obs['cell_type'] == 'T cell']
```

## 記憶體考量

AnnData 結構設計為記憶體高效：
- 稀疏矩陣減少稀疏資料的記憶體
- 視圖盡可能避免複製資料
- Backed 模式能夠處理比 RAM 更大的資料
- 類別註解減少離散值的記憶體

```python
# 將字串轉換為類別（更節省記憶體）
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
adata.strings_to_categoricals()

# 檢查物件是否為視圖（不擁有資料）
if adata.is_view:
    adata = adata.copy()  # 建立獨立複製
```
