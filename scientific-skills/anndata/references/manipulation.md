# 資料操作

用於轉換、子集和操作 AnnData 物件的操作。

## 子集

### 按索引
```python
import anndata as ad
import numpy as np

adata = ad.AnnData(X=np.random.rand(1000, 2000))

# 整數索引
subset = adata[0:100, 0:500]  # 前 100 個觀測值，前 500 個變數

# 索引列表
obs_indices = [0, 10, 20, 30, 40]
var_indices = [0, 1, 2, 3, 4]
subset = adata[obs_indices, var_indices]

# 單個觀測值或變數
single_obs = adata[0, :]
single_var = adata[:, 0]
```

### 按名稱
```python
import pandas as pd

# 使用命名索引建立
obs_names = [f'cell_{i}' for i in range(1000)]
var_names = [f'gene_{i}' for i in range(2000)]
adata = ad.AnnData(
    X=np.random.rand(1000, 2000),
    obs=pd.DataFrame(index=obs_names),
    var=pd.DataFrame(index=var_names)
)

# 按觀測名稱子集
subset = adata[['cell_0', 'cell_1', 'cell_2'], :]

# 按變數名稱子集
subset = adata[:, ['gene_0', 'gene_10', 'gene_20']]

# 兩個軸
subset = adata[['cell_0', 'cell_1'], ['gene_0', 'gene_1']]
```

### 按布林遮罩
```python
# 建立布林遮罩
high_count_obs = np.random.rand(1000) > 0.5
high_var_genes = np.random.rand(2000) > 0.7

# 使用遮罩子集
subset = adata[high_count_obs, :]
subset = adata[:, high_var_genes]
subset = adata[high_count_obs, high_var_genes]
```

### 按中繼資料條件
```python
# 添加中繼資料
adata.obs['cell_type'] = np.random.choice(['A', 'B', 'C'], 1000)
adata.obs['quality_score'] = np.random.rand(1000)
adata.var['highly_variable'] = np.random.rand(2000) > 0.8

# 按細胞類型過濾
t_cells = adata[adata.obs['cell_type'] == 'A']

# 按多個條件過濾
high_quality_a_cells = adata[
    (adata.obs['cell_type'] == 'A') &
    (adata.obs['quality_score'] > 0.7)
]

# 按變數中繼資料過濾
hv_genes = adata[:, adata.var['highly_variable']]

# 複雜條件
filtered = adata[
    (adata.obs['quality_score'] > 0.5) &
    (adata.obs['cell_type'].isin(['A', 'B'])),
    adata.var['highly_variable']
]
```

## 轉置

```python
# 轉置 AnnData 物件（交換觀測值和變數）
adata_T = adata.T

# 形狀改變
print(adata.shape)    # (1000, 2000)
print(adata_T.shape)  # (2000, 1000)

# obs 和 var 交換
print(adata.obs.head())   # 觀測中繼資料
print(adata_T.var.head()) # 相同資料，現在作為變數中繼資料

# 當資料方向相反時有用
# 常見於某些基因為列的檔案格式
```

## 複製

### 完整複製
```python
# 建立獨立複製
adata_copy = adata.copy()

# 對複製的修改不影響原始
adata_copy.obs['new_column'] = 1
print('new_column' in adata.obs.columns)  # False
```

### 淺複製
```python
# 視圖（不複製資料，修改影響原始）
adata_view = adata[0:100, :]

# 檢查物件是否為視圖
print(adata_view.is_view)  # True

# 將視圖轉換為獨立複製
adata_independent = adata_view.copy()
print(adata_independent.is_view)  # False
```

## 重新命名

### 重新命名觀測值和變數
```python
# 重新命名所有觀測值
adata.obs_names = [f'new_cell_{i}' for i in range(adata.n_obs)]

# 重新命名所有變數
adata.var_names = [f'new_gene_{i}' for i in range(adata.n_vars)]

# 使名稱唯一（為重複項添加後綴）
adata.obs_names_make_unique()
adata.var_names_make_unique()
```

### 重新命名類別
```python
# 建立類別欄位
adata.obs['cell_type'] = pd.Categorical(['A', 'B', 'C'] * 333 + ['A'])

# 重新命名類別
adata.rename_categories('cell_type', ['Type_A', 'Type_B', 'Type_C'])

# 或使用字典
adata.rename_categories('cell_type', {
    'Type_A': 'T_cell',
    'Type_B': 'B_cell',
    'Type_C': 'Monocyte'
})
```

## 類型轉換

### 字串轉類別
```python
# 將字串欄位轉換為類別（更節省記憶體）
adata.obs['cell_type'] = ['TypeA', 'TypeB'] * 500
adata.obs['tissue'] = ['brain', 'liver'] * 500

# 將所有字串欄位轉換為類別
adata.strings_to_categoricals()

print(adata.obs['cell_type'].dtype)  # category
print(adata.obs['tissue'].dtype)     # category
```

### 稀疏與密集互轉
```python
from scipy.sparse import csr_matrix

# 密集轉稀疏
if not isinstance(adata.X, csr_matrix):
    adata.X = csr_matrix(adata.X)

# 稀疏轉密集
if isinstance(adata.X, csr_matrix):
    adata.X = adata.X.toarray()

# 轉換圖層
adata.layers['normalized'] = csr_matrix(adata.layers['normalized'])
```

## 分塊操作

分塊處理大型資料集：

```python
# 分塊遍歷資料
chunk_size = 100
for chunk in adata.chunked_X(chunk_size):
    # 處理區塊
    result = process_chunk(chunk)
```

## 擷取向量

### 取得觀測向量
```python
# 取得觀測中繼資料為陣列
cell_types = adata.obs_vector('cell_type')

# 取得跨觀測值的基因表達
actb_expression = adata.obs_vector('ACTB')  # 如果 ACTB 在 var_names 中
```

### 取得變數向量
```python
# 取得變數中繼資料為陣列
gene_names = adata.var_vector('gene_name')
```

## 添加/修改資料

### 添加觀測值
```python
# 建立新觀測值
new_obs = ad.AnnData(X=np.random.rand(100, adata.n_vars))
new_obs.var_names = adata.var_names

# 與現有連接
adata_extended = ad.concat([adata, new_obs], axis=0)
```

### 添加變數
```python
# 建立新變數
new_vars = ad.AnnData(X=np.random.rand(adata.n_obs, 100))
new_vars.obs_names = adata.obs_names

# 與現有連接
adata_extended = ad.concat([adata, new_vars], axis=1)
```

### 添加中繼資料欄位
```python
# 添加觀測註解
adata.obs['new_score'] = np.random.rand(adata.n_obs)

# 添加變數註解
adata.var['new_label'] = ['label'] * adata.n_vars

# 從外部資料添加
external_data = pd.read_csv('metadata.csv', index_col=0)
adata.obs['external_info'] = external_data.loc[adata.obs_names, 'column']
```

### 添加圖層
```python
# 添加新圖層
adata.layers['raw_counts'] = np.random.randint(0, 100, adata.shape)
adata.layers['log_transformed'] = np.log1p(adata.X)

# 替換圖層
adata.layers['normalized'] = new_normalized_data
```

### 添加嵌入
```python
# 添加 PCA
adata.obsm['X_pca'] = np.random.rand(adata.n_obs, 50)

# 添加 UMAP
adata.obsm['X_umap'] = np.random.rand(adata.n_obs, 2)

# 添加多個嵌入
adata.obsm['X_tsne'] = np.random.rand(adata.n_obs, 2)
adata.obsm['X_diffmap'] = np.random.rand(adata.n_obs, 10)
```

### 添加成對關係
```python
from scipy.sparse import csr_matrix

# 添加最近鄰圖
n_obs = adata.n_obs
knn_graph = csr_matrix(np.random.rand(n_obs, n_obs) > 0.95)
adata.obsp['connectivities'] = knn_graph

# 添加距離矩陣
adata.obsp['distances'] = csr_matrix(np.random.rand(n_obs, n_obs))
```

### 添加非結構化資料
```python
# 添加分析參數
adata.uns['pca'] = {
    'variance': [0.2, 0.15, 0.1],
    'variance_ratio': [0.4, 0.3, 0.2],
    'params': {'n_comps': 50}
}

# 添加色彩方案
adata.uns['cell_type_colors'] = ['#FF0000', '#00FF00', '#0000FF']
```

## 移除資料

### 移除觀測值或變數
```python
# 僅保留特定觀測值
keep_obs = adata.obs['quality_score'] > 0.5
adata = adata[keep_obs, :]

# 移除特定變數
remove_vars = adata.var['low_count']
adata = adata[:, ~remove_vars]
```

### 移除中繼資料欄位
```python
# 移除觀測欄位
adata.obs.drop('unwanted_column', axis=1, inplace=True)

# 移除變數欄位
adata.var.drop('unwanted_column', axis=1, inplace=True)
```

### 移除圖層
```python
# 移除特定圖層
del adata.layers['unwanted_layer']

# 移除所有圖層
adata.layers = {}
```

### 移除嵌入
```python
# 移除特定嵌入
del adata.obsm['X_tsne']

# 移除所有嵌入
adata.obsm = {}
```

### 移除非結構化資料
```python
# 移除特定鍵
del adata.uns['unwanted_key']

# 移除所有非結構化資料
adata.uns = {}
```

## 重新排序

### 排序觀測值
```python
# 按觀測中繼資料排序
adata = adata[adata.obs.sort_values('quality_score').index, :]

# 按觀測名稱排序
adata = adata[sorted(adata.obs_names), :]
```

### 排序變數
```python
# 按變數中繼資料排序
adata = adata[:, adata.var.sort_values('gene_name').index]

# 按變數名稱排序
adata = adata[:, sorted(adata.var_names)]
```

### 重新排序以匹配外部列表
```python
# 重新排序觀測值以匹配外部列表
desired_order = ['cell_10', 'cell_5', 'cell_20', ...]
adata = adata[desired_order, :]

# 重新排序變數
desired_genes = ['TP53', 'ACTB', 'GAPDH', ...]
adata = adata[:, desired_genes]
```

## 資料轉換

### 標準化
```python
# 總計數標準化（類似 CPM/TPM）
total_counts = adata.X.sum(axis=1)
adata.layers['normalized'] = adata.X / total_counts[:, np.newaxis] * 1e6

# 對數轉換
adata.layers['log1p'] = np.log1p(adata.X)

# Z 分數標準化
mean = adata.X.mean(axis=0)
std = adata.X.std(axis=0)
adata.layers['scaled'] = (adata.X - mean) / std
```

### 過濾
```python
# 按總計數過濾細胞
total_counts = np.array(adata.X.sum(axis=1)).flatten()
adata.obs['total_counts'] = total_counts
adata = adata[adata.obs['total_counts'] > 1000, :]

# 按檢測率過濾基因
detection_rate = (adata.X > 0).sum(axis=0) / adata.n_obs
adata.var['detection_rate'] = np.array(detection_rate).flatten()
adata = adata[:, adata.var['detection_rate'] > 0.01]
```

## 處理視圖

視圖是資料子集的輕量參照，不複製底層矩陣：

```python
# 建立視圖
view = adata[0:100, 0:500]
print(view.is_view)  # True

# 視圖允許讀取存取
data = view.X

# 修改視圖資料會影響原始
# （要小心！）

# 將視圖轉換為獨立複製
independent = view.copy()

# 強制 AnnData 為複製而非視圖
adata = adata.copy()
```

## 合併中繼資料

```python
# 合併外部中繼資料
external_metadata = pd.read_csv('additional_metadata.csv', index_col=0)

# 連接中繼資料（在索引上內連接）
adata.obs = adata.obs.join(external_metadata)

# 左連接（保留所有 adata 觀測值）
adata.obs = adata.obs.merge(
    external_metadata,
    left_index=True,
    right_index=True,
    how='left'
)
```

## 常見操作模式

### 品質控制過濾
```python
# 計算 QC 指標
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)
adata.var['n_cells'] = (adata.X > 0).sum(axis=0)

# 過濾低品質細胞
adata = adata[adata.obs['n_genes'] > 200, :]
adata = adata[adata.obs['total_counts'] < 50000, :]

# 過濾很少檢測到的基因
adata = adata[:, adata.var['n_cells'] >= 3]
```

### 選擇高變異基因
```python
# 標記高變異基因
gene_variance = np.var(adata.X, axis=0)
adata.var['variance'] = np.array(gene_variance).flatten()
adata.var['highly_variable'] = adata.var['variance'] > np.percentile(gene_variance, 90)

# 子集至高變異基因
adata_hvg = adata[:, adata.var['highly_variable']].copy()
```

### 降抽樣
```python
# 觀測值的隨機抽樣
np.random.seed(42)
n_sample = 500
sample_indices = np.random.choice(adata.n_obs, n_sample, replace=False)
adata_downsampled = adata[sample_indices, :].copy()

# 按細胞類型分層抽樣
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(
    range(adata.n_obs),
    test_size=0.2,
    stratify=adata.obs['cell_type']
)
adata_train = adata[train_idx, :].copy()
adata_test = adata[test_idx, :].copy()
```

### 分割訓練/測試
```python
# 隨機訓練/測試分割
np.random.seed(42)
n_obs = adata.n_obs
train_size = int(0.8 * n_obs)
indices = np.random.permutation(n_obs)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

adata_train = adata[train_indices, :].copy()
adata_test = adata[test_indices, :].copy()
```
