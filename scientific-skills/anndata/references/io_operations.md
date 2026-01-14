# 輸入/輸出操作

AnnData 提供全面的 I/O 功能，用於讀取和寫入各種格式的資料。

## 原生格式

### H5AD（基於 HDF5）
AnnData 物件的推薦原生格式，提供高效儲存和快速存取。

#### 寫入 H5AD 檔案
```python
import anndata as ad

# 寫入檔案
adata.write_h5ad('data.h5ad')

# 附帶壓縮寫入
adata.write_h5ad('data.h5ad', compression='gzip')

# 使用特定壓縮等級寫入（0-9，越高 = 越多壓縮）
adata.write_h5ad('data.h5ad', compression='gzip', compression_opts=9)
```

#### 讀取 H5AD 檔案
```python
# 將整個檔案讀入記憶體
adata = ad.read_h5ad('data.h5ad')

# 以 backed 模式讀取（大型檔案的延遲載入）
adata = ad.read_h5ad('data.h5ad', backed='r')  # 唯讀
adata = ad.read_h5ad('data.h5ad', backed='r+')  # 讀寫

# Backed 模式能夠處理比 RAM 更大的資料集
# 僅存取的資料會載入記憶體
```

#### Backed 模式操作
```python
# 以 backed 模式開啟
adata = ad.read_h5ad('large_dataset.h5ad', backed='r')

# 存取中繼資料而不將 X 載入記憶體
print(adata.obs.head())
print(adata.var.head())

# 子集操作建立視圖
subset = adata[:100, :500]  # 視圖，未載入資料

# 將特定資料載入記憶體
X_subset = subset.X[:]  # 現在載入此子集

# 將整個 backed 物件轉換為記憶體
adata_memory = adata.to_memory()
```

### Zarr
階層式陣列儲存格式，最佳化用於雲端儲存和並行 I/O。

#### 寫入 Zarr
```python
# 寫入 Zarr 儲存
adata.write_zarr('data.zarr')

# 使用特定區塊寫入（對效能重要）
adata.write_zarr('data.zarr', chunks=(100, 100))
```

#### 讀取 Zarr
```python
# 讀取 Zarr 儲存
adata = ad.read_zarr('data.zarr')
```

#### 遠端 Zarr 存取
```python
import fsspec

# 從 S3 存取 Zarr
store = fsspec.get_mapper('s3://bucket-name/data.zarr')
adata = ad.read_zarr(store)

# 從 URL 存取 Zarr
store = fsspec.get_mapper('https://example.com/data.zarr')
adata = ad.read_zarr(store)
```

## 替代輸入格式

### CSV/TSV
```python
# 讀取 CSV（基因為欄，細胞為列）
adata = ad.read_csv('data.csv')

# 使用自訂分隔符讀取
adata = ad.read_csv('data.tsv', delimiter='\t')

# 指定第一欄為列名
adata = ad.read_csv('data.csv', first_column_names=True)
```

### Excel
```python
# 讀取 Excel 檔案
adata = ad.read_excel('data.xlsx')

# 讀取特定工作表
adata = ad.read_excel('data.xlsx', sheet='Sheet1')
```

### Matrix Market（MTX）
基因體學中稀疏矩陣的常見格式。

```python
# 讀取 MTX 及相關檔案
# 需要：matrix.mtx、genes.tsv、barcodes.tsv
adata = ad.read_mtx('matrix.mtx')

# 使用自訂基因和條碼檔案讀取
adata = ad.read_mtx(
    'matrix.mtx',
    var_names='genes.tsv',
    obs_names='barcodes.tsv'
)

# 如需要則轉置（MTX 通常基因為列）
adata = adata.T
```

### 10X Genomics 格式
```python
# 讀取 10X h5 格式
adata = ad.read_10x_h5('filtered_feature_bc_matrix.h5')

# 讀取 10X MTX 目錄
adata = ad.read_10x_mtx('filtered_feature_bc_matrix/')

# 如果存在多個基因組則指定
adata = ad.read_10x_h5('data.h5', genome='GRCh38')
```

### Loom
```python
# 讀取 Loom 檔案
adata = ad.read_loom('data.loom')

# 使用特定觀測和變數註解讀取
adata = ad.read_loom(
    'data.loom',
    obs_names='CellID',
    var_names='Gene'
)
```

### 文字檔案
```python
# 讀取通用文字檔案
adata = ad.read_text('data.txt', delimiter='\t')

# 使用自訂參數讀取
adata = ad.read_text(
    'data.txt',
    delimiter=',',
    first_column_names=True,
    dtype='float32'
)
```

### UMI 工具
```python
# 讀取 UMI 工具格式
adata = ad.read_umi_tools('counts.tsv')
```

### HDF5（通用）
```python
# 從 HDF5 檔案讀取（非 h5ad 格式）
adata = ad.read_hdf('data.h5', key='dataset')
```

## 替代輸出格式

### CSV
```python
# 寫入 CSV 檔案（建立多個檔案）
adata.write_csvs('output_dir/')

# 這會建立：
# - output_dir/X.csv（表達矩陣）
# - output_dir/obs.csv（觀測註解）
# - output_dir/var.csv（變數註解）
# - output_dir/uns.csv（非結構化註解，如果可能）

# 跳過某些組件
adata.write_csvs('output_dir/', skip_data=True)  # 跳過 X 矩陣
```

### Loom
```python
# 寫入 Loom 格式
adata.write_loom('output.loom')
```

## 讀取特定元素

對於細粒度控制，從儲存讀取特定元素：

```python
from anndata import read_elem

# 僅讀取觀測註解
obs = read_elem('data.h5ad/obs')

# 讀取特定圖層
layer = read_elem('data.h5ad/layers/normalized')

# 讀取非結構化資料元素
params = read_elem('data.h5ad/uns/pca_params')
```

## 寫入特定元素

```python
from anndata import write_elem
import h5py

# 寫入元素到現有檔案
with h5py.File('data.h5ad', 'a') as f:
    write_elem(f, 'new_layer', adata.X.copy())
```

## 延遲操作

對於非常大的資料集，使用延遲讀取以避免載入整個資料集：

```python
from anndata.experimental import read_elem_lazy

# 延遲讀取（回傳 dask 陣列或類似物）
X_lazy = read_elem_lazy('large_data.h5ad/X')

# 僅在需要時計算
subset = X_lazy[:100, :100].compute()
```

## 常見 I/O 模式

### 格式轉換
```python
# MTX 到 H5AD
adata = ad.read_mtx('matrix.mtx').T
adata.write_h5ad('data.h5ad')

# CSV 到 H5AD
adata = ad.read_csv('data.csv')
adata.write_h5ad('data.h5ad')

# H5AD 到 Zarr
adata = ad.read_h5ad('data.h5ad')
adata.write_zarr('data.zarr')
```

### 載入中繼資料而不載入資料
```python
# Backed 模式允許在不載入 X 的情況下檢查中繼資料
adata = ad.read_h5ad('large_file.h5ad', backed='r')
print(f"資料集包含 {adata.n_obs} 個觀測值和 {adata.n_vars} 個變數")
print(adata.obs.columns)
print(adata.var.columns)
# X 未載入記憶體
```

### 附加到現有檔案
```python
# 以讀寫模式開啟
adata = ad.read_h5ad('data.h5ad', backed='r+')

# 修改中繼資料
adata.obs['new_column'] = values

# 變更寫入磁碟
```

### 從 URL 下載
```python
import anndata as ad

# 直接從 URL 讀取（用於 h5ad 檔案）
url = 'https://example.com/data.h5ad'
adata = ad.read_h5ad(url, backed='r')  # 串流存取

# 對於其他格式，先下載
import urllib.request
urllib.request.urlretrieve(url, 'local_file.h5ad')
adata = ad.read_h5ad('local_file.h5ad')
```

## 效能提示

### 讀取
- 對於僅需查詢的大型檔案使用 `backed='r'`
- 如需修改中繼資料而不載入所有資料使用 `backed='r+'`
- H5AD 格式通常最適合隨機存取
- Zarr 更適合雲端儲存和並行存取
- 考慮壓縮以節省儲存空間，但注意可能會減慢讀取速度

### 寫入
- 使用壓縮進行長期儲存：`compression='gzip'` 或 `compression='lzf'`
- LZF 壓縮更快但壓縮率比 GZIP 低
- 對於 Zarr，根據存取模式調整區塊大小：
  - 順序讀取使用較大區塊
  - 隨機存取使用較小區塊
- 寫入前將字串欄位轉換為類別（較小的檔案）

### 記憶體管理
```python
# 將字串轉換為類別（減少檔案大小和記憶體）
adata.strings_to_categoricals()
adata.write_h5ad('data.h5ad')

# 對稀疏資料使用稀疏矩陣
from scipy.sparse import csr_matrix
if isinstance(adata.X, np.ndarray):
    density = np.count_nonzero(adata.X) / adata.X.size
    if density < 0.5:  # 如果超過 50% 為零
        adata.X = csr_matrix(adata.X)
```

## 處理大型資料集

### 策略 1：Backed 模式
```python
# 處理比 RAM 更大的資料集
adata = ad.read_h5ad('100GB_file.h5ad', backed='r')

# 基於中繼資料過濾（快速，不載入資料）
filtered = adata[adata.obs['quality_score'] > 0.8]

# 將過濾後的子集載入記憶體
adata_memory = filtered.to_memory()
```

### 策略 2：分塊處理
```python
# 分塊處理資料
adata = ad.read_h5ad('large_file.h5ad', backed='r')

chunk_size = 1000
results = []

for i in range(0, adata.n_obs, chunk_size):
    chunk = adata[i:i+chunk_size, :].to_memory()
    # 處理區塊
    result = process(chunk)
    results.append(result)
```

### 策略 3：使用 AnnCollection
```python
from anndata.experimental import AnnCollection

# 不載入資料建立集合
adatas = [f'dataset_{i}.h5ad' for i in range(10)]
collection = AnnCollection(
    adatas,
    join_obs='inner',
    join_vars='inner'
)

# 延遲處理集合
# 僅在存取時載入資料
```

## 常見問題和解決方案

### 問題：讀取時記憶體不足
**解決方案**：使用 backed 模式或分塊讀取
```python
adata = ad.read_h5ad('file.h5ad', backed='r')
```

### 問題：從雲端儲存讀取緩慢
**解決方案**：使用 Zarr 格式並適當分塊
```python
adata.write_zarr('data.zarr', chunks=(1000, 1000))
```

### 問題：檔案大小過大
**解決方案**：使用壓縮並轉換為稀疏/類別
```python
adata.strings_to_categoricals()
from scipy.sparse import csr_matrix
adata.X = csr_matrix(adata.X)
adata.write_h5ad('compressed.h5ad', compression='gzip')
```

### 問題：無法修改 backed 物件
**解決方案**：載入到記憶體或以 'r+' 模式開啟
```python
# 選項 1：載入到記憶體
adata = adata.to_memory()

# 選項 2：以讀寫模式開啟
adata = ad.read_h5ad('file.h5ad', backed='r+')
```
