# Zarr Python 快速參考

本參考提供常用 Zarr 函數、參數和模式的簡明概述，方便在開發過程中快速查閱。

## 陣列建立函數

### `zarr.zeros()` / `zarr.ones()` / `zarr.empty()`
```python
zarr.zeros(shape, chunks=None, dtype='f8', store=None, compressor='default',
           fill_value=0, order='C', filters=None)
```
建立填充零、一或空值（未初始化）的陣列。

**關鍵參數：**
- `shape`：定義陣列維度的元組（例如 `(1000, 1000)`）
- `chunks`：定義區塊維度的元組（例如 `(100, 100)`），或 `None` 表示不分塊
- `dtype`：NumPy 資料型別（例如 `'f4'`、`'i8'`、`'bool'`）
- `store`：儲存位置（字串路徑、Store 物件，或 `None` 表示記憶體）
- `compressor`：壓縮編解碼器或 `None` 表示不壓縮

### `zarr.create_array()` / `zarr.create()`
```python
zarr.create_array(store, shape, chunks, dtype='f8', compressor='default',
                  fill_value=0, order='C', filters=None, overwrite=False)
```
建立新陣列，可明確控制所有參數。

### `zarr.array()`
```python
zarr.array(data, chunks=None, dtype=None, compressor='default', store=None)
```
從現有資料（NumPy 陣列、列表等）建立陣列。

**範例：**
```python
import numpy as np
data = np.random.random((1000, 1000))
z = zarr.array(data, chunks=(100, 100), store='data.zarr')
```

### `zarr.open_array()` / `zarr.open()`
```python
zarr.open_array(store, mode='a', shape=None, chunks=None, dtype=None,
                compressor='default', fill_value=0)
```
開啟現有陣列或建立新陣列。

**模式選項：**
- `'r'`：唯讀
- `'r+'`：讀寫，檔案必須存在
- `'a'`：讀寫，不存在則建立（預設）
- `'w'`：建立新檔，存在則覆寫
- `'w-'`：建立新檔，存在則失敗

## 儲存類別

### LocalStore（預設）
```python
from zarr.storage import LocalStore

store = LocalStore('path/to/data.zarr')
z = zarr.open_array(store=store, mode='w', shape=(1000, 1000), chunks=(100, 100))
```

### MemoryStore
```python
from zarr.storage import MemoryStore

store = MemoryStore()  # 資料僅在記憶體中
z = zarr.open_array(store=store, mode='w', shape=(1000, 1000), chunks=(100, 100))
```

### ZipStore
```python
from zarr.storage import ZipStore

# 寫入
store = ZipStore('data.zip', mode='w')
z = zarr.open_array(store=store, mode='w', shape=(1000, 1000), chunks=(100, 100))
z[:] = data
store.close()  # 必須關閉

# 讀取
store = ZipStore('data.zip', mode='r')
z = zarr.open_array(store=store)
data = z[:]
store.close()
```

### 雲端儲存（S3/GCS）
```python
# S3
import s3fs
s3 = s3fs.S3FileSystem(anon=False)
store = s3fs.S3Map(root='bucket/path/data.zarr', s3=s3)

# GCS
import gcsfs
gcs = gcsfs.GCSFileSystem(project='my-project')
store = gcsfs.GCSMap(root='bucket/path/data.zarr', gcs=gcs)
```

## 壓縮編解碼器

### Blosc 編解碼器（預設）
```python
from zarr.codecs.blosc import BloscCodec

codec = BloscCodec(
    cname='zstd',      # 壓縮器：'blosclz'、'lz4'、'lz4hc'、'snappy'、'zlib'、'zstd'
    clevel=5,          # 壓縮等級：0-9
    shuffle='shuffle'  # Shuffle 過濾器：'noshuffle'、'shuffle'、'bitshuffle'
)

z = zarr.create_array(store='data.zarr', shape=(1000, 1000), chunks=(100, 100),
                      dtype='f4', codecs=[codec])
```

**Blosc 壓縮器特性：**
- `'lz4'`：最快壓縮，較低壓縮比
- `'zstd'`：平衡（預設），良好的壓縮比和速度
- `'zlib'`：良好相容性，中等性能
- `'lz4hc'`：比 lz4 更好的壓縮比，較慢
- `'snappy'`：快速，中等壓縮比
- `'blosclz'`：Blosc 的預設

### 其他編解碼器
```python
from zarr.codecs import GzipCodec, ZstdCodec, BytesCodec

# Gzip 壓縮（最大壓縮比，較慢）
GzipCodec(level=6)  # 等級 0-9

# Zstandard 壓縮
ZstdCodec(level=3)  # 等級 1-22

# 無壓縮
BytesCodec()
```

## 陣列索引與選擇

### 基本索引（NumPy 風格）
```python
z = zarr.zeros((1000, 1000), chunks=(100, 100))

# 讀取
row = z[0, :]           # 單行
col = z[:, 0]           # 單列
block = z[10:20, 50:60] # 切片
element = z[5, 10]      # 單一元素

# 寫入
z[0, :] = 42
z[10:20, 50:60] = np.random.random((10, 10))
```

### 進階索引
```python
# 座標索引（點選擇）
z.vindex[[0, 5, 10], [2, 8, 15]]  # 特定座標

# 正交索引（外積）
z.oindex[0:10, [5, 10, 15]]  # 第 0-9 行，第 5、10、15 列

# 區塊/chunk 索引
z.blocks[0, 0]  # 第一個 chunk
z.blocks[0:2, 0:2]  # 前四個 chunks
```

## 群組與階層結構

### 建立群組
```python
# 建立根群組
root = zarr.group(store='data.zarr')

# 建立巢狀群組
grp1 = root.create_group('group1')
grp2 = grp1.create_group('subgroup')

# 在群組中建立陣列
arr = grp1.create_array(name='data', shape=(1000, 1000),
                        chunks=(100, 100), dtype='f4')

# 透過路徑存取
arr2 = root['group1/data']
```

### 群組方法
```python
root = zarr.group('data.zarr')

# h5py 相容方法
dataset = root.create_dataset('data', shape=(1000, 1000), chunks=(100, 100))
subgrp = root.require_group('subgroup')  # 不存在則建立

# 視覺化結構
print(root.tree())

# 列出內容
print(list(root.keys()))
print(list(root.groups()))
print(list(root.arrays()))
```

## 陣列屬性與元資料

### 處理屬性
```python
z = zarr.zeros((1000, 1000), chunks=(100, 100))

# 設定屬性
z.attrs['units'] = 'meters'
z.attrs['description'] = 'Temperature data'
z.attrs['created'] = '2024-01-15'
z.attrs['version'] = 1.2
z.attrs['tags'] = ['climate', 'temperature']

# 讀取屬性
print(z.attrs['units'])
print(dict(z.attrs))  # 所有屬性轉為字典

# 更新/刪除
z.attrs['version'] = 2.0
del z.attrs['tags']
```

**注意：** 屬性必須是 JSON 可序列化的。

## 陣列屬性與方法

### 屬性
```python
z = zarr.zeros((1000, 1000), chunks=(100, 100), dtype='f4')

z.shape          # (1000, 1000)
z.chunks         # (100, 100)
z.dtype          # dtype('float32')
z.size           # 1000000
z.nbytes         # 4000000（未壓縮大小，位元組）
z.nbytes_stored  # 磁碟上實際壓縮大小
z.nchunks        # 100（chunk 數量）
z.cdata_shape    # 以 chunks 表示的形狀：(10, 10)
```

### 方法
```python
# 資訊
print(z.info)  # 陣列的詳細資訊
print(z.info_items())  # 資訊以元組列表表示

# 調整大小
z.resize(1500, 1500)  # 改變維度

# 附加
z.append(new_data, axis=0)  # 沿軸新增資料

# 複製
z2 = z.copy(store='new_location.zarr')
```

## 分塊指南

### Chunk 大小計算
```python
# 對於 float32（每元素 4 位元組）：
# 1 MB = 262,144 元素
# 10 MB = 2,621,440 元素

# 1 MB chunks 範例：
(512, 512)      # 對於 2D：512 × 512 × 4 = 1,048,576 位元組
(128, 128, 128) # 對於 3D：128 × 128 × 128 × 4 = 8,388,608 位元組 ≈ 8 MB
(64, 256, 256)  # 對於 3D：64 × 256 × 256 × 4 = 16,777,216 位元組 ≈ 16 MB
```

### 依存取模式的分塊策略

**時間序列（沿第一維循序存取）：**
```python
chunks=(1, 720, 1440)  # 每個 chunk 一個時間步
```

**逐行存取：**
```python
chunks=(10, 10000)  # 小行，跨越列
```

**逐列存取：**
```python
chunks=(10000, 10)  # 跨越行，小列
```

**隨機存取：**
```python
chunks=(500, 500)  # 平衡的方形 chunks
```

**3D 體積資料：**
```python
chunks=(64, 64, 64)  # 用於各向同性存取的立方體 chunks
```

## 整合 API

### NumPy 整合
```python
import numpy as np

z = zarr.zeros((1000, 1000), chunks=(100, 100))

# 使用 NumPy 函數
result = np.sum(z, axis=0)
mean = np.mean(z)
std = np.std(z)

# 轉換為 NumPy
arr = z[:]  # 將整個陣列載入記憶體
```

### Dask 整合
```python
import dask.array as da

# 將 Zarr 載入為 Dask 陣列
dask_array = da.from_zarr('data.zarr')

# 平行計算運算
result = dask_array.mean(axis=0).compute()

# 將 Dask 陣列寫入 Zarr
large_array = da.random.random((100000, 100000), chunks=(1000, 1000))
da.to_zarr(large_array, 'output.zarr')
```

### Xarray 整合
```python
import xarray as xr

# 將 Zarr 開啟為 Xarray Dataset
ds = xr.open_zarr('data.zarr')

# 將 Xarray 寫入 Zarr
ds.to_zarr('output.zarr')

# 使用座標建立
ds = xr.Dataset(
    {'temperature': (['time', 'lat', 'lon'], data)},
    coords={
        'time': pd.date_range('2024-01-01', periods=365),
        'lat': np.arange(-90, 91, 1),
        'lon': np.arange(-180, 180, 1)
    }
)
ds.to_zarr('climate.zarr')
```

## 平行運算

### 同步器
```python
from zarr import ThreadSynchronizer, ProcessSynchronizer

# 多執行緒寫入
sync = ThreadSynchronizer()
z = zarr.open_array('data.zarr', mode='r+', synchronizer=sync)

# 多程序寫入
sync = ProcessSynchronizer('sync.sync')
z = zarr.open_array('data.zarr', mode='r+', synchronizer=sync)
```

**注意：** 同步僅在以下情況需要：
- 可能跨越 chunk 邊界的並發寫入
- 讀取不需要（始終安全）
- 如果每個程序寫入不同的 chunks 則不需要

## 元資料合併

```python
# 合併元資料（在建立所有陣列/群組後）
zarr.consolidate_metadata('data.zarr')

# 使用合併的元資料開啟（更快，特別是在雲端）
root = zarr.open_consolidated('data.zarr')
```

**優點：**
- 將 I/O 從 N 次操作減少到 1 次
- 對雲端儲存至關重要（減少延遲）
- 加速階層結構遍歷

**注意事項：**
- 如果資料更新可能會過時
- 修改後需重新合併
- 不適用於頻繁更新的資料集

## 常見模式

### 具有增長資料的時間序列
```python
# 以空的第一維開始
z = zarr.open('timeseries.zarr', mode='a',
              shape=(0, 720, 1440),
              chunks=(1, 720, 1440),
              dtype='f4')

# 附加新的時間步
for new_timestep in data_stream:
    z.append(new_timestep, axis=0)
```

### 分塊處理大型陣列
```python
z = zarr.open('large_data.zarr', mode='r')

# 不載入整個陣列進行處理
for i in range(0, z.shape[0], 1000):
    chunk = z[i:i+1000, :]
    result = process(chunk)
    save(result)
```

### 格式轉換流程
```python
# HDF5 → Zarr
import h5py
with h5py.File('data.h5', 'r') as h5:
    z = zarr.array(h5['dataset'][:], chunks=(1000, 1000), store='data.zarr')

# Zarr → NumPy 檔案
z = zarr.open('data.zarr', mode='r')
np.save('data.npy', z[:])

# Zarr → NetCDF（透過 Xarray）
ds = xr.open_zarr('data.zarr')
ds.to_netcdf('data.nc')
```

## 性能優化快速檢核表

1. **Chunk 大小**：每個 chunk 1-10 MB
2. **Chunk 形狀**：與存取模式對齊
3. **壓縮**：
   - 快速：`BloscCodec(cname='lz4', clevel=1)`
   - 平衡：`BloscCodec(cname='zstd', clevel=5)`
   - 最大壓縮：`GzipCodec(level=9)`
4. **雲端儲存**：
   - 較大的 chunks（5-100 MB）
   - 合併元資料
   - 考慮 sharding
5. **平行 I/O**：大型操作使用 Dask
6. **記憶體**：分塊處理，不要載入整個陣列

## 除錯與效能分析

```python
z = zarr.open('data.zarr', mode='r')

# 詳細資訊
print(z.info)

# 大小統計
print(f"未壓縮：{z.nbytes / 1e6:.2f} MB")
print(f"已壓縮：{z.nbytes_stored / 1e6:.2f} MB")
print(f"壓縮比：{z.nbytes / z.nbytes_stored:.1f}x")

# Chunk 資訊
print(f"Chunks：{z.chunks}")
print(f"Chunk 數量：{z.nchunks}")
print(f"Chunk 網格：{z.cdata_shape}")
```

## 常見資料型別

```python
# 整數
'i1', 'i2', 'i4', 'i8'  # 有號：8、16、32、64 位元
'u1', 'u2', 'u4', 'u8'  # 無號：8、16、32、64 位元

# 浮點數
'f2', 'f4', 'f8'  # 16、32、64 位元（半精度、單精度、雙精度）

# 其他
'bool'     # 布林
'c8', 'c16'  # 複數：64、128 位元
'S10'      # 固定長度字串（10 位元組）
'U10'      # Unicode 字串（10 字元）
```

## 版本相容性

Zarr-Python 3.x 版本支援：
- **Zarr v2 格式**：舊版格式，廣泛相容
- **Zarr v3 格式**：新格式，具有 sharding、改進的元資料

檢查格式版本：
```python
# Zarr 自動偵測格式版本
z = zarr.open('data.zarr', mode='r')
# 格式資訊可在元資料中取得
```

## 錯誤處理

```python
try:
    z = zarr.open_array('data.zarr', mode='r')
except zarr.errors.PathNotFoundError:
    print("陣列不存在")
except zarr.errors.ReadOnlyError:
    print("無法寫入唯讀陣列")
except Exception as e:
    print(f"意外錯誤：{e}")
```

