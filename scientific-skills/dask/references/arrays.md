# Dask Arrays

## 概述

Dask Array 使用分塊演算法實作 NumPy 的 ndarray 介面。它協調許多排列成網格的 NumPy 陣列，以實現對超過可用記憶體的資料集進行運算，並利用多核心的平行處理。

## 核心概念

Dask Array 被分成多個分塊（blocks）：
- 每個分塊是一個常規的 NumPy 陣列
- 操作以平行方式應用於每個分塊
- 結果自動組合
- 支援核心外運算（資料大於 RAM）

## 主要功能

### Dask Arrays 支援的功能

**數學操作**：
- 算術運算（+、-、*、/）
- 純量函數（指數、對數、三角函數）
- 逐元素操作

**歸約操作**：
- `sum()`、`mean()`、`std()`、`var()`
- 沿指定軸的歸約
- `min()`、`max()`、`argmin()`、`argmax()`

**線性代數**：
- 張量收縮
- 點積和矩陣乘法
- 部分分解（SVD、QR）

**資料操作**：
- 轉置
- 切片（標準和花式索引）
- 重塑
- 串接和堆疊

**陣列協定**：
- 通用函數（ufuncs）
- NumPy 協定以實現互操作性

## 何時使用 Dask Arrays

**使用 Dask Arrays 的情況**：
- 陣列超過可用 RAM
- 運算可以跨分塊平行化
- 使用 NumPy 風格的數值操作
- 需要將 NumPy 程式碼擴展到更大的資料集

**繼續使用 NumPy 的情況**：
- 陣列可以輕鬆放入記憶體
- 操作需要資料的全域視圖
- 使用 Dask 中未提供的專門函數
- NumPy 的效能已經足夠

## 重要限制

Dask Arrays 刻意不實作某些 NumPy 功能：

**未實作**：
- 大多數 `np.linalg` 函數（僅提供基本操作）
- 難以平行化的操作（如完整排序）
- 記憶體效率低的操作（轉換為列表、透過迴圈迭代）
- 許多專門函數（由社群需求驅動）

**替代方案**：對於不支援的操作，考慮使用 `map_blocks` 搭配自訂 NumPy 程式碼。

## 建立 Dask Arrays

### 從 NumPy 陣列
```python
import dask.array as da
import numpy as np

# 從指定分塊的 NumPy 陣列建立
x = np.arange(10000)
dx = da.from_array(x, chunks=1000)  # 建立 10 個各 1000 個元素的分塊
```

### 隨機陣列
```python
# 建立指定分塊的隨機陣列
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# 其他隨機函數
x = da.random.normal(10, 0.1, size=(10000, 10000), chunks=(1000, 1000))
```

### 零、一和空陣列
```python
# 建立填充常數的陣列
zeros = da.zeros((10000, 10000), chunks=(1000, 1000))
ones = da.ones((10000, 10000), chunks=(1000, 1000))
empty = da.empty((10000, 10000), chunks=(1000, 1000))
```

### 從函數建立
```python
# 從函數建立陣列
def create_block(block_id):
    return np.random.random((1000, 1000)) * block_id[0]

x = da.from_delayed(
    [[dask.delayed(create_block)((i, j)) for j in range(10)] for i in range(10)],
    shape=(10000, 10000),
    dtype=float
)
```

### 從磁碟載入
```python
# 從 HDF5 載入
import h5py
f = h5py.File('myfile.hdf5', mode='r')
x = da.from_array(f['/data'], chunks=(1000, 1000))

# 從 Zarr 載入
import zarr
z = zarr.open('myfile.zarr', mode='r')
x = da.from_array(z, chunks=(1000, 1000))
```

## 常見操作

### 算術操作
```python
import dask.array as da

x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = da.random.random((10000, 10000), chunks=(1000, 1000))

# 逐元素操作（延遲）
z = x + y
z = x * y
z = da.exp(x)
z = da.log(y)

# 計算結果
result = z.compute()
```

### 歸約操作
```python
# 沿軸歸約
total = x.sum().compute()
mean = x.mean().compute()
std = x.std().compute()

# 沿特定軸歸約
row_means = x.mean(axis=1).compute()
col_sums = x.sum(axis=0).compute()
```

### 切片和索引
```python
# 標準切片（回傳 Dask Array）
subset = x[1000:5000, 2000:8000]

# 花式索引
indices = [0, 5, 10, 15]
selected = x[indices, :]

# 布林索引
mask = x > 0.5
filtered = x[mask]
```

### 矩陣操作
```python
# 矩陣乘法
A = da.random.random((10000, 5000), chunks=(1000, 1000))
B = da.random.random((5000, 8000), chunks=(1000, 1000))
C = da.matmul(A, B)
result = C.compute()

# 點積
dot_product = da.dot(A, B)

# 轉置
AT = A.T
```

### 線性代數
```python
# SVD（奇異值分解）
U, s, Vt = da.linalg.svd(A)
U_computed, s_computed, Vt_computed = dask.compute(U, s, Vt)

# QR 分解
Q, R = da.linalg.qr(A)
Q_computed, R_computed = dask.compute(Q, R)

# 注意：僅部分線性代數操作可用
```

### 重塑和操作
```python
# 重塑
x = da.random.random((10000, 10000), chunks=(1000, 1000))
reshaped = x.reshape(5000, 20000)

# 轉置
transposed = x.T

# 串接
x1 = da.random.random((5000, 10000), chunks=(1000, 1000))
x2 = da.random.random((5000, 10000), chunks=(1000, 1000))
combined = da.concatenate([x1, x2], axis=0)

# 堆疊
stacked = da.stack([x1, x2], axis=0)
```

## 分塊策略

分塊對於 Dask Array 效能至關重要。

### 分塊大小指南

**良好的分塊大小**：
- 每個分塊：約 10-100 MB（壓縮後）
- 數值資料每個分塊約 100 萬個元素
- 在平行性和開銷之間取得平衡

**計算範例**：
```python
# 對於 float64 資料（每個元素 8 bytes）
# 目標 100 MB 分塊：100 MB / 8 bytes = 1250 萬個元素

# 對於 2D 陣列 (10000, 10000)：
x = da.random.random((10000, 10000), chunks=(1000, 1000))  # 每個分塊約 8 MB
```

### 查看分塊結構
```python
# 檢查分塊
print(x.chunks)  # ((1000, 1000, ...), (1000, 1000, ...))

# 分塊數量
print(x.npartitions)

# 分塊大小（bytes）
print(x.nbytes / x.npartitions)
```

### 重新分塊
```python
# 變更分塊大小
x = da.random.random((10000, 10000), chunks=(500, 500))
x_rechunked = x.rechunk((2000, 2000))

# 重新分塊特定維度
x_rechunked = x.rechunk({0: 2000, 1: 'auto'})
```

## 使用 map_blocks 進行自訂操作

對於 Dask 中未提供的操作，使用 `map_blocks`：

```python
import dask.array as da
import numpy as np

def custom_function(block):
    # 應用自訂 NumPy 操作
    return np.fft.fft2(block)

x = da.random.random((10000, 10000), chunks=(1000, 1000))
result = da.map_blocks(custom_function, x, dtype=x.dtype)

# 計算
output = result.compute()
```

### 具有不同輸出形狀的 map_blocks
```python
def reduction_function(block):
    # 為每個分塊回傳純量
    return np.array([block.mean()])

result = da.map_blocks(
    reduction_function,
    x,
    dtype='float64',
    drop_axis=[0, 1],  # 輸出沒有來自輸入的軸
    new_axis=0,        # 輸出有新軸
    chunks=(1,)        # 每個分塊一個元素
)
```

## 延遲求值與運算

### 延遲操作
```python
# 所有操作都是延遲的（即時、無運算）
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = x + 100
z = y.mean(axis=0)
result = z * 2

# 尚未運算，只是建構了任務圖
```

### 觸發運算
```python
# 計算單一結果
final = result.compute()

# 高效計算多個結果
result1, result2 = dask.compute(operation1, operation2)
```

### 持久化到記憶體
```python
# 將中間結果保留在記憶體中
x_cached = x.persist()

# 重複使用快取的結果
y1 = (x_cached + 10).compute()
y2 = (x_cached * 2).compute()
```

## 儲存結果

### 轉為 NumPy
```python
# 轉換為 NumPy（載入所有到記憶體）
numpy_array = dask_array.compute()
```

### 存到磁碟
```python
# 儲存到 HDF5
import h5py
with h5py.File('output.hdf5', mode='w') as f:
    dset = f.create_dataset('/data', shape=x.shape, dtype=x.dtype)
    da.store(x, dset)

# 儲存到 Zarr
import zarr
z = zarr.open('output.zarr', mode='w', shape=x.shape, dtype=x.dtype, chunks=x.chunks)
da.store(x, z)
```

## 效能考量

### 高效操作
- 逐元素操作：非常高效
- 可平行化操作的歸約：高效
- 沿分塊邊界切片：高效
- 具有良好分塊對齊的矩陣操作：高效

### 昂貴操作
- 跨多個分塊切片：需要資料移動
- 需要全域排序的操作：支援不佳
- 極不規則的存取模式：效能差
- 分塊對齊不佳的操作：需要重新分塊

### 優化技巧

**1. 選擇良好的分塊大小**
```python
# 目標平衡的分塊
# 良好：每個分塊約 100 MB
x = da.random.random((100000, 10000), chunks=(10000, 10000))
```

**2. 為操作對齊分塊**
```python
# 確保分塊為操作對齊
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = da.random.random((10000, 10000), chunks=(1000, 1000))  # 對齊
z = x + y  # 高效
```

**3. 使用適當的排程器**
```python
# 陣列在執行緒排程器下運作良好（預設）
# 共享記憶體存取是高效的
result = x.compute()  # 預設使用執行緒
```

**4. 最小化資料傳輸**
```python
# 較佳：在每個分塊上計算，然後傳輸結果
means = x.mean(axis=1).compute()  # 傳輸較少資料

# 較差：傳輸所有資料然後計算
x_numpy = x.compute()
means = x_numpy.mean(axis=1)  # 傳輸較多資料
```

## 常見模式

### 影像處理
```python
import dask.array as da

# 載入大型影像堆疊
images = da.from_zarr('images.zarr')

# 應用濾波
def apply_gaussian(block):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(block, sigma=2)

filtered = da.map_blocks(apply_gaussian, images, dtype=images.dtype)

# 計算統計
mean_intensity = filtered.mean().compute()
```

### 科學運算
```python
# 大規模數值模擬
x = da.random.random((100000, 100000), chunks=(10000, 10000))

# 應用迭代運算
for i in range(num_iterations):
    x = da.exp(-x) * da.sin(x)
    x = x.persist()  # 為下一次迭代保留在記憶體中

# 最終結果
result = x.compute()
```

### 資料分析
```python
# 載入大型資料集
data = da.from_zarr('measurements.zarr')

# 計算統計
mean = data.mean(axis=0)
std = data.std(axis=0)
normalized = (data - mean) / std

# 儲存標準化資料
da.to_zarr(normalized, 'normalized.zarr')
```

## 與其他工具整合

### XArray
```python
import xarray as xr
import dask.array as da

# XArray 以標記維度包裝 Dask 陣列
data = da.random.random((1000, 2000, 3000), chunks=(100, 200, 300))
dataset = xr.DataArray(
    data,
    dims=['time', 'y', 'x'],
    coords={'time': range(1000), 'y': range(2000), 'x': range(3000)}
)
```

### Scikit-learn（透過 Dask-ML）
```python
# 部分與 scikit-learn 相容的操作
from dask_ml.preprocessing import StandardScaler

X = da.random.random((10000, 100), chunks=(1000, 100))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 除錯技巧

### 視覺化任務圖
```python
# 視覺化運算圖（適用於小陣列）
x = da.random.random((100, 100), chunks=(10, 10))
y = x + 1
y.visualize(filename='graph.png')
```

### 檢查陣列屬性
```python
# 計算前檢查
print(f"Shape: {x.shape}")
print(f"Dtype: {x.dtype}")
print(f"Chunks: {x.chunks}")
print(f"Number of tasks: {len(x.__dask_graph__())}")
```

### 先在小陣列上測試
```python
# 在小陣列上測試邏輯
small_x = da.random.random((100, 100), chunks=(50, 50))
result_small = computation(small_x).compute()

# 驗證後，再擴展
large_x = da.random.random((100000, 100000), chunks=(10000, 10000))
result_large = computation(large_x).compute()
```
