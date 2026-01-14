# 連接 AnnData 物件

沿觀測值或變數軸合併多個 AnnData 物件。

## 基本連接

### 沿觀測值連接（堆疊細胞/樣本）
```python
import anndata as ad
import numpy as np

# 建立多個 AnnData 物件
adata1 = ad.AnnData(X=np.random.rand(100, 50))
adata2 = ad.AnnData(X=np.random.rand(150, 50))
adata3 = ad.AnnData(X=np.random.rand(200, 50))

# 沿觀測值連接（axis=0，預設）
adata_combined = ad.concat([adata1, adata2, adata3], axis=0)

print(adata_combined.shape)  # (450, 50)
```

### 沿變數連接（堆疊基因/特徵）
```python
# 建立具有相同觀測值、不同變數的物件
adata1 = ad.AnnData(X=np.random.rand(100, 50))
adata2 = ad.AnnData(X=np.random.rand(100, 30))
adata3 = ad.AnnData(X=np.random.rand(100, 70))

# 沿變數連接（axis=1）
adata_combined = ad.concat([adata1, adata2, adata3], axis=1)

print(adata_combined.shape)  # (100, 150)
```

## 連接類型

### 內連接（交集）
僅保留所有物件中都存在的變數/觀測值。

```python
import pandas as pd

# 建立具有不同變數的物件
adata1 = ad.AnnData(
    X=np.random.rand(100, 50),
    var=pd.DataFrame(index=[f'Gene_{i}' for i in range(50)])
)
adata2 = ad.AnnData(
    X=np.random.rand(150, 60),
    var=pd.DataFrame(index=[f'Gene_{i}' for i in range(10, 70)])
)

# 內連接：僅保留基因 10-49（重疊部分）
adata_inner = ad.concat([adata1, adata2], join='inner')
print(adata_inner.n_vars)  # 40 個基因（重疊）
```

### 外連接（聯集）
保留所有變數/觀測值，填充遺失值。

```python
# 外連接：保留所有基因
adata_outer = ad.concat([adata1, adata2], join='outer')
print(adata_outer.n_vars)  # 70 個基因（聯集）

# 遺失值以適當的預設值填充：
# - 稀疏矩陣為 0
# - 密集矩陣為 NaN
```

### 外連接的填充值
```python
# 指定遺失資料的填充值
adata_filled = ad.concat([adata1, adata2], join='outer', fill_value=0)
```

## 追蹤資料來源

### 添加批次標籤
```python
# 標記每個觀測值來自哪個物件
adata_combined = ad.concat(
    [adata1, adata2, adata3],
    label='batch',  # 標籤的欄位名稱
    keys=['batch1', 'batch2', 'batch3']  # 每個物件的標籤
)

print(adata_combined.obs['batch'].value_counts())
# batch1    100
# batch2    150
# batch3    200
```

### 自動批次標籤
```python
# 如果未提供 keys，使用整數索引
adata_combined = ad.concat(
    [adata1, adata2, adata3],
    label='dataset'
)
# dataset 欄位包含：0, 1, 2
```

## 合併策略

使用 `merge` 參數控制不同物件的中繼資料如何合併。

### merge=None（觀測值的預設值）
排除非連接軸上的中繼資料。

```python
# 連接觀測值時，var 中繼資料必須匹配
adata1.var['gene_type'] = 'protein_coding'
adata2.var['gene_type'] = 'protein_coding'

# 僅當所有物件相同時才保留 var
adata_combined = ad.concat([adata1, adata2], merge=None)
```

### merge='same'
保留所有物件中相同的中繼資料。

```python
adata1.var['chromosome'] = ['chr1'] * 25 + ['chr2'] * 25
adata2.var['chromosome'] = ['chr1'] * 25 + ['chr2'] * 25
adata1.var['type'] = 'protein_coding'
adata2.var['type'] = 'lncRNA'  # 不同

# 'chromosome' 保留（相同），'type' 被排除（不同）
adata_combined = ad.concat([adata1, adata2], merge='same')
```

### merge='unique'
保留每個鍵只有一個值的中繼資料欄位。

```python
adata1.var['gene_id'] = [f'ENSG{i:05d}' for i in range(50)]
adata2.var['gene_id'] = [f'ENSG{i:05d}' for i in range(50)]

# gene_id 保留（每個鍵的唯一值）
adata_combined = ad.concat([adata1, adata2], merge='unique')
```

### merge='first'
取包含每個鍵的第一個物件的值。

```python
adata1.var['description'] = ['Desc1'] * 50
adata2.var['description'] = ['Desc2'] * 50

# 使用 adata1 的描述
adata_combined = ad.concat([adata1, adata2], merge='first')
```

### merge='only'
保留僅出現在一個物件中的中繼資料。

```python
adata1.var['adata1_specific'] = [1] * 50
adata2.var['adata2_specific'] = [2] * 50

# 兩個中繼資料欄位都保留
adata_combined = ad.concat([adata1, adata2], merge='only')
```

## 處理索引衝突

### 使索引唯一
```python
import pandas as pd

# 建立具有重疊觀測名稱的物件
adata1 = ad.AnnData(
    X=np.random.rand(3, 10),
    obs=pd.DataFrame(index=['cell_1', 'cell_2', 'cell_3'])
)
adata2 = ad.AnnData(
    X=np.random.rand(3, 10),
    obs=pd.DataFrame(index=['cell_1', 'cell_2', 'cell_3'])
)

# 通過附加批次鍵使索引唯一
adata_combined = ad.concat(
    [adata1, adata2],
    label='batch',
    keys=['batch1', 'batch2'],
    index_unique='_'  # 使索引唯一的分隔符
)

print(adata_combined.obs_names)
# ['cell_1_batch1', 'cell_2_batch1', 'cell_3_batch1',
#  'cell_1_batch2', 'cell_2_batch2', 'cell_3_batch2']
```

## 連接圖層

```python
# 具有圖層的物件
adata1 = ad.AnnData(X=np.random.rand(100, 50))
adata1.layers['normalized'] = np.random.rand(100, 50)
adata1.layers['scaled'] = np.random.rand(100, 50)

adata2 = ad.AnnData(X=np.random.rand(150, 50))
adata2.layers['normalized'] = np.random.rand(150, 50)
adata2.layers['scaled'] = np.random.rand(150, 50)

# 如果所有物件中都存在，圖層會自動連接
adata_combined = ad.concat([adata1, adata2])

print(adata_combined.layers.keys())
# dict_keys(['normalized', 'scaled'])
```

## 連接多維註解

### obsm/varm
```python
# 具有嵌入的物件
adata1.obsm['X_pca'] = np.random.rand(100, 50)
adata2.obsm['X_pca'] = np.random.rand(150, 50)

# obsm 沿觀測軸連接
adata_combined = ad.concat([adata1, adata2])
print(adata_combined.obsm['X_pca'].shape)  # (250, 50)
```

### obsp/varp（成對註解）
```python
from scipy.sparse import csr_matrix

# 成對矩陣
adata1.obsp['connectivities'] = csr_matrix((100, 100))
adata2.obsp['connectivities'] = csr_matrix((150, 150))

# 預設情況下，obsp 不連接（設定 pairwise=True 以包含）
adata_combined = ad.concat([adata1, adata2])
# adata_combined.obsp 為空

# 包含成對資料（建立區塊對角矩陣）
adata_combined = ad.concat([adata1, adata2], pairwise=True)
print(adata_combined.obsp['connectivities'].shape)  # (250, 250)
```

## 連接 uns（非結構化）

非結構化中繼資料遞迴合併：

```python
adata1.uns['experiment'] = {'date': '2025-01-01', 'batch': 'A'}
adata2.uns['experiment'] = {'date': '2025-01-01', 'batch': 'B'}

# 對 uns 使用 merge='unique'
adata_combined = ad.concat([adata1, adata2], uns_merge='unique')
# 'date' 保留（相同值），'batch' 可能被排除（不同值）
```

## 延遲連接（AnnCollection）

對於非常大的資料集，使用不載入所有資料的延遲連接：

```python
from anndata.experimental import AnnCollection

# 從檔案路徑建立集合（不載入資料）
files = ['data1.h5ad', 'data2.h5ad', 'data3.h5ad']
collection = AnnCollection(
    files,
    join_obs='outer',
    join_vars='inner',
    label='dataset',
    keys=['dataset1', 'dataset2', 'dataset3']
)

# 延遲存取資料
print(collection.n_obs)  # 總觀測值
print(collection.obs.head())  # 載入中繼資料，而非 X

# 需要時轉換為常規 AnnData（載入所有資料）
adata = collection.to_adata()
```

### 使用 AnnCollection
```python
# 不載入資料進行子集
subset = collection[collection.obs['cell_type'] == 'T cell']

# 遍歷資料集
for adata in collection:
    print(adata.shape)

# 存取特定資料集
first_dataset = collection[0]
```

## 磁碟上連接

對於記憶體容納不下的資料集，直接在磁碟上連接：

```python
from anndata.experimental import concat_on_disk

# 不載入記憶體進行連接
concat_on_disk(
    ['data1.h5ad', 'data2.h5ad', 'data3.h5ad'],
    'combined.h5ad',
    join='outer'
)

# 以 backed 模式載入結果
adata = ad.read_h5ad('combined.h5ad', backed='r')
```

## 常見連接模式

### 合併技術重複
```python
# 相同樣本的多次運行
replicates = [adata_run1, adata_run2, adata_run3]
adata_combined = ad.concat(
    replicates,
    label='technical_replicate',
    keys=['rep1', 'rep2', 'rep3'],
    join='inner'  # 僅保留所有運行中測量的基因
)
```

### 合併實驗批次
```python
# 不同的實驗批次
batches = [adata_batch1, adata_batch2, adata_batch3]
adata_combined = ad.concat(
    batches,
    label='batch',
    keys=['batch1', 'batch2', 'batch3'],
    join='outer'  # 保留所有基因
)

# 稍後：應用批次校正
```

### 合併多模態資料
```python
# 不同的測量模態（例如 RNA + 蛋白質）
adata_rna = ad.AnnData(X=np.random.rand(100, 2000))
adata_protein = ad.AnnData(X=np.random.rand(100, 50))

# 沿變數連接以合併模態
adata_multimodal = ad.concat([adata_rna, adata_protein], axis=1)

# 添加標籤以區分模態
adata_multimodal.var['modality'] = ['RNA'] * 2000 + ['protein'] * 50
```

## 最佳實務

1. **連接前檢查相容性**
```python
# 驗證形狀相容
print([adata.n_vars for adata in [adata1, adata2, adata3]])

# 檢查變數名稱匹配
print([set(adata.var_names) for adata in [adata1, adata2, adata3]])
```

2. **使用適當的連接類型**
- `inner`：當您需要所有樣本中相同的特徵時（最嚴格）
- `outer`：當您想保留所有特徵時（最包容）

3. **追蹤資料來源**
始終使用 `label` 和 `keys` 追蹤哪些觀測值來自哪個資料集。

4. **考慮記憶體使用**
- 對於大型資料集，使用 `AnnCollection` 或 `concat_on_disk`
- 考慮對結果使用 backed 模式

5. **處理批次效應**
連接合併資料但不校正批次效應。連接後應用批次校正：
```python
# 連接後應用批次校正
import scanpy as sc
sc.pp.combat(adata_combined, key='batch')
```

6. **驗證結果**
```python
# 檢查維度
print(adata_combined.shape)

# 檢查批次分布
print(adata_combined.obs['batch'].value_counts())

# 驗證中繼資料完整性
print(adata_combined.var.head())
print(adata_combined.obs.head())
```
