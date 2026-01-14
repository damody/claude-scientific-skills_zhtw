# Dask 最佳實踐

## 效能優化原則

### 先從較簡單的解決方案開始

在使用 Dask 實作平行運算之前，探索這些替代方案：
- 針對特定問題的更好演算法
- 高效的檔案格式（用 Parquet、HDF5、Zarr 代替 CSV）
- 透過 Numba 或 Cython 編譯程式碼
- 用於開發和測試的資料抽樣

這些替代方案通常比分散式系統提供更好的回報，應該在擴展到平行運算之前充分探索。

### 分塊大小策略

**關鍵規則**：分塊應該足夠小，讓多個分塊可以同時放入 worker 的可用記憶體中。

**建議目標**：調整分塊大小，使 worker 可以在每個核心上容納 10 個分塊而不超出可用記憶體。

**重要性**：
- 分塊太大：記憶體溢出和平行化效率低
- 分塊太小：過多的排程開銷

**計算範例**：
- 8 核心，32 GB RAM
- 目標：每個分塊約 400 MB（32 GB / 8 核心 / 10 個分塊）

### 使用儀表板監控

Dask 儀表板提供對以下方面的重要可見性：
- Worker 狀態和資源使用率
- 任務進度和瓶頸
- 記憶體使用模式
- 效能特徵

存取儀表板以了解平行工作負載中實際緩慢的部分，而不是猜測優化方向。

## 要避免的關鍵陷阱

### 1. 不要在 Dask 之前在本地建立大型物件

**錯誤方法**：
```python
import pandas as pd
import dask.dataframe as dd

# 首先將整個資料集載入記憶體
df = pd.read_csv('large_file.csv')
ddf = dd.from_pandas(df, npartitions=10)
```

**正確方法**：
```python
import dask.dataframe as dd

# 讓 Dask 處理載入
ddf = dd.read_csv('large_file.csv')
```

**原因**：先用 pandas 或 NumPy 載入資料會強制排程器在任務圖中序列化和嵌入這些物件，這違背了平行運算的目的。

**關鍵原則**：使用 Dask 方法來載入資料，並使用 Dask 來控制結果。

### 2. 避免重複呼叫 compute()

**錯誤方法**：
```python
results = []
for item in items:
    result = dask_computation(item).compute()  # 每次 compute 是獨立的
    results.append(result)
```

**正確方法**：
```python
computations = [dask_computation(item) for item in items]
results = dask.compute(*computations)  # 單次 compute 處理所有
```

**原因**：在迴圈中呼叫 compute 會阻止 Dask：
- 平行化不同的運算
- 共享中間結果
- 優化整體任務圖

### 3. 不要建構過大的任務圖

**症狀**：
- 單一運算中有數百萬個任務
- 嚴重的排程開銷
- 運算開始前的長時間延遲

**解決方案**：
- 增加分塊大小以減少任務數量
- 使用 `map_partitions` 或 `map_blocks` 來融合操作
- 將運算分解為較小的部分並進行中間持久化
- 考慮問題是否真的需要分散式運算

**使用 map_partitions 的範例**：
```python
# 不是對每一行應用函數
ddf['result'] = ddf.apply(complex_function, axis=1)  # 很多任務

# 而是一次對整個分區應用
ddf = ddf.map_partitions(lambda df: df.assign(result=complex_function(df)))
```

## 基礎架構考量

### 排程器選擇

**使用執行緒的情況**：
- 使用釋放 GIL 的函式庫進行數值運算（NumPy、Pandas、scikit-learn）
- 受益於共享記憶體的操作
- 使用陣列/dataframe 操作的單機工作負載

**使用程序的情況**：
- 文字處理和 Python 集合操作
- 受 GIL 限制的純 Python 程式碼
- 需要程序隔離的操作

**使用分散式排程器的情況**：
- 多機器叢集
- 需要診斷儀表板
- 非同步 API
- 更好的資料局部性處理

### 執行緒配置

**建議**：數值工作負載中，目標是每個程序約 4 個執行緒。

**理由**：
- 平行性和開銷之間的平衡
- 允許 CPU 核心的有效使用
- 減少上下文切換成本

### 記憶體管理

**策略性持久化**：
```python
# 持久化重複使用的中間結果
intermediate = expensive_computation(data).persist()
result1 = intermediate.operation1().compute()
result2 = intermediate.operation2().compute()
```

**完成後清理記憶體**：
```python
# 明確刪除大型物件
del intermediate
```

## 資料載入最佳實踐

### 使用適當的檔案格式

**用於表格資料**：
- Parquet：列式、壓縮、快速篩選
- CSV：僅用於小資料或初始載入

**用於陣列資料**：
- HDF5：適合數值陣列
- Zarr：雲端原生、適合平行
- NetCDF：帶有中繼資料的科學資料

### 優化資料載入

**高效讀取多個檔案**：
```python
# 使用 glob 模式平行讀取多個檔案
ddf = dd.read_parquet('data/year=2024/month=*/day=*.parquet')
```

**儘早指定有用的欄位**：
```python
# 只讀取需要的欄位
ddf = dd.read_parquet('data.parquet', columns=['col1', 'col2', 'col3'])
```

## 常見模式和解決方案

### 模式：embarrassingly parallel 問題

對於獨立的運算，使用 Futures：
```python
from dask.distributed import Client

client = Client()
futures = [client.submit(func, arg) for arg in args]
results = client.gather(futures)
```

### 模式：資料前處理管道

使用 Bags 進行初始 ETL，然後轉換為結構化格式：
```python
import dask.bag as db

# 處理原始 JSON
bag = db.read_text('logs/*.json').map(json.loads)
bag = bag.filter(lambda x: x['status'] == 'success')

# 轉換為 DataFrame 進行分析
ddf = bag.to_dataframe()
```

### 模式：迭代演算法

在迭代之間持久化資料：
```python
data = dd.read_parquet('data.parquet')
data = data.persist()  # 跨迭代保持在記憶體中

for iteration in range(num_iterations):
    data = update_function(data)
    data = data.persist()  # 持久化更新後的版本
```

## 除錯技巧

### 使用單執行緒排程器

用於 pdb 除錯或詳細錯誤檢查：
```python
import dask

dask.config.set(scheduler='synchronous')
result = computation.compute()  # 在單執行緒中執行以便除錯
```

### 檢查任務圖大小

計算前，檢查任務數量：
```python
print(len(ddf.__dask_graph__()))  # 應該合理，不是數百萬
```

### 先在小資料上驗證

在擴展前在小子集上測試邏輯：
```python
# 在第一個分區上測試
sample = ddf.head(1000)
# 驗證結果
# 然後擴展到完整資料集
```

## 效能故障排除

### 症狀：運算啟動緩慢

**可能原因**：任務圖太大
**解決方案**：增加分塊大小或使用 map_partitions

### 症狀：記憶體錯誤

**可能原因**：
- 分塊太大
- 中間結果太多
- 使用者函數中的記憶體洩漏

**解決方案**：
- 減小分塊大小
- 策略性使用 persist() 並在完成後刪除
- 分析使用者函數的記憶體問題

### 症狀：平行化效果差

**可能原因**：
- 資料依賴阻止平行化
- 分塊太大（任務不夠）
- 在 Python 程式碼上使用執行緒時的 GIL 競爭

**解決方案**：
- 重構運算以減少依賴
- 增加分區數量
- 對 Python 程式碼切換到多程序排程器
