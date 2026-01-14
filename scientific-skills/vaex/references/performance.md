# 效能和最佳化

本參考文件涵蓋 Vaex 的效能功能，包括惰性求值、快取、記憶體管理、非同步操作，以及處理海量資料集的最佳化策略。

## 理解惰性求值

惰性求值（lazy evaluation）是 Vaex 效能的基礎：

### 惰性求值的運作方式

```python
import vaex

df = vaex.open('large_file.hdf5')

# 這裡不會發生計算 - 只是定義要計算什麼
df['total'] = df.price * df.quantity
df['log_price'] = df.price.log()
mean_expr = df.total.mean()

# 這裡才發生計算（當需要結果時）
result = mean_expr  # 現在才實際計算平均值
```

**關鍵概念：**
- **表達式**是惰性的 - 它們定義計算但不執行
- **實體化（materialization）**發生在存取結果時
- **查詢最佳化**在執行前自動進行

### 何時觸發求值？

```python
# 這些會觸發求值：
print(df.x.mean())                    # 存取值
array = df.x.values                   # 取得 NumPy 陣列
pdf = df.to_pandas_df()              # 轉換為 pandas
df.export_hdf5('output.hdf5')       # 匯出

# 這些不會觸發求值：
df['new_col'] = df.x + df.y          # 建立虛擬欄
expr = df.x.mean()                    # 建立表達式
df_filtered = df[df.x > 10]          # 建立篩選視圖
```

## 使用 delay=True 批次操作

將多個操作一起執行以獲得更好的效能：

### 基本延遲執行

```python
# 不使用 delay - 每個操作都處理整個資料集
mean_x = df.x.mean()      # 第 1 次遍歷資料
std_x = df.x.std()        # 第 2 次遍歷資料
max_x = df.x.max()        # 第 3 次遍歷資料

# 使用 delay - 單次遍歷資料集
mean_x = df.x.mean(delay=True)
std_x = df.x.std(delay=True)
max_x = df.x.max(delay=True)
results = vaex.execute([mean_x, std_x, max_x])  # 單次遍歷！

print(results[0])  # 平均值
print(results[1])  # 標準差
print(results[2])  # 最大值
```

### 多欄位的延遲執行

```python
# 高效計算多個欄位的統計
stats = {}
delayed_results = []

for column in ['sales', 'quantity', 'profit', 'cost']:
    mean = df[column].mean(delay=True)
    std = df[column].std(delay=True)
    delayed_results.extend([mean, std])

# 一次執行全部
results = vaex.execute(delayed_results)

# 處理結果
for i, column in enumerate(['sales', 'quantity', 'profit', 'cost']):
    stats[column] = {
        'mean': results[i*2],
        'std': results[i*2 + 1]
    }
```

### 何時使用 delay=True

在以下情況使用 `delay=True`：
- 計算多個聚合
- 計算多個欄位的統計
- 建構儀表板或報告
- 任何需要多次遍歷資料的場景

```python
# 不好：4 次遍歷資料集
mean1 = df.col1.mean()
mean2 = df.col2.mean()
mean3 = df.col3.mean()
mean4 = df.col4.mean()

# 好：1 次遍歷資料集
results = vaex.execute([
    df.col1.mean(delay=True),
    df.col2.mean(delay=True),
    df.col3.mean(delay=True),
    df.col4.mean(delay=True)
])
```

## 非同步操作

使用 async/await 非同步處理資料：

### 使用 async/await 的非同步

```python
import vaex
import asyncio

async def compute_statistics(df):
    # 建立非同步任務
    mean_task = df.x.mean(delay=True)
    std_task = df.x.std(delay=True)

    # 非同步執行
    results = await vaex.async_execute([mean_task, std_task])

    return {'mean': results[0], 'std': results[1]}

# 執行非同步函數
async def main():
    df = vaex.open('large_file.hdf5')
    stats = await compute_statistics(df)
    print(stats)

asyncio.run(main())
```

### 使用 Promise/Future

```python
# 取得 future 物件
future = df.x.mean(delay=True)

# 做其他工作...

# 準備好時取得結果
result = future.get()  # 阻塞直到完成
```

## 虛擬欄 vs 實體化欄

理解差異對效能至關重要：

### 虛擬欄（偏好使用）

```python
# 虛擬欄 - 即時計算，零記憶體
df['total'] = df.price * df.quantity
df['log_sales'] = df.sales.log()
df['full_name'] = df.first_name + ' ' + df.last_name

# 檢查是否為虛擬
print(df.is_local('total'))  # False = 虛擬

# 優點：
# - 零記憶體開銷
# - 如果來源資料變更，總是最新的
# - 建立速度快
```

### 實體化欄

```python
# 實體化虛擬欄
df['total_materialized'] = df['total'].values

# 或使用 materialize 方法
df = df.materialize(df['total'], inplace=True)

# 檢查是否已實體化
print(df.is_local('total_materialized'))  # True = 已實體化

# 何時實體化：
# - 欄位被重複計算（分攤成本）
# - 複雜表達式用於多個操作
# - 需要匯出資料
```

### 決定：虛擬 vs 實體化

```python
# 虛擬較好的情況：
# - 欄位很簡單（x + y、x * 2 等）
# - 欄位很少使用
# - 記憶體有限

# 實體化的情況：
# - 複雜計算（多個操作）
# - 重複用於聚合
# - 拖慢其他操作

# 範例：複雜計算使用多次
df['complex'] = (df.x.log() * df.y.sqrt() + df.z ** 2).values  # 實體化
```

## 快取策略

Vaex 自動快取某些操作，但您可以進一步最佳化：

### 自動快取

```python
# 第一次呼叫計算並快取
mean1 = df.x.mean()  # 計算

# 第二次呼叫使用快取
mean2 = df.x.mean()  # 從快取（即時）

# 如果 DataFrame 變更，快取失效
df['new_col'] = df.x + 1
mean3 = df.x.mean()  # 重新計算
```

### 狀態管理

```python
# 儲存 DataFrame 狀態（包括虛擬欄）
df.state_write('state.json')

# 稍後載入狀態
df_new = vaex.open('data.hdf5')
df_new.state_load('state.json')  # 還原虛擬欄、選擇
```

### 檢查點模式

```python
# 匯出複雜管線的中間結果
df['processed'] = complex_calculation(df)

# 儲存檢查點
df.export_hdf5('checkpoint.hdf5')

# 從檢查點繼續
df = vaex.open('checkpoint.hdf5')
# 繼續處理...
```

## 記憶體管理

最佳化非常大型資料集的記憶體使用：

### 記憶體映射檔案

```python
# HDF5 和 Arrow 是記憶體映射的（最佳）
df = vaex.open('data.hdf5')  # 存取前不使用記憶體

# 檔案保留在磁碟，只有存取的部分載入 RAM
mean = df.x.mean()  # 串流處理資料，最小記憶體
```

### 分塊處理

```python
# 分塊處理大型 DataFrame
chunk_size = 1_000_000

for i1, i2, chunk in df.to_pandas_df(chunk_size=chunk_size):
    # 處理分塊（注意：這違背了 Vaex 的目的）
    process_chunk(chunk)

# 更好：直接使用 Vaex 操作（不需要分塊）
result = df.x.mean()  # 自動處理大型資料
```

### 監控記憶體使用

```python
# 檢查 DataFrame 記憶體佔用
print(df.byte_size())  # 實體化欄使用的位元組

# 檢查欄位記憶體
for col in df.get_column_names():
    if df.is_local(col):
        print(f"{col}: {df[col].nbytes / 1e9:.2f} GB")

# 分析操作
import vaex.profiler
with vaex.profiler():
    result = df.x.mean()
```

## 平行計算

Vaex 自動平行化操作：

### 多執行緒

```python
# Vaex 預設使用所有 CPU 核心
import vaex

# 檢查/設定執行緒數
print(vaex.multithreading.thread_count_default)
vaex.multithreading.thread_count_default = 8  # 使用 8 個執行緒

# 操作自動平行化
mean = df.x.mean()  # 使用所有執行緒
```

### 使用 Dask 的分散式計算

```python
# 轉換為 Dask 進行分散式處理
import vaex
import dask.dataframe as dd

# 建立 Vaex DataFrame
df_vaex = vaex.open('large_file.hdf5')

# 轉換為 Dask
df_dask = df_vaex.to_dask_dataframe()

# 使用 Dask 處理
result = df_dask.groupby('category')['value'].sum().compute()
```

## JIT 編譯

Vaex 可以對自訂操作使用即時（Just-In-Time）編譯：

### 使用 Numba

```python
import vaex
import numba

# 定義 JIT 編譯函數
@numba.jit
def custom_calculation(x, y):
    return x ** 2 + y ** 2

# 套用到 DataFrame
df['custom'] = df.apply(custom_calculation,
                        arguments=[df.x, df.y],
                        vectorize=True)
```

### 自訂聚合

```python
@numba.jit
def custom_sum(a):
    total = 0
    for val in a:
        total += val * 2  # 自訂邏輯
    return total

# 在聚合中使用
result = df.x.custom_agg(custom_sum)
```

## 最佳化策略

### 策略 1：最小化實體化

```python
# 不好：建立許多實體化欄
df['a'] = (df.x + df.y).values
df['b'] = (df.a * 2).values
df['c'] = (df.b + df.z).values

# 好：保持虛擬直到最終匯出
df['a'] = df.x + df.y
df['b'] = df.a * 2
df['c'] = df.b + df.z
# 只在匯出時實體化：
# df.export_hdf5('output.hdf5')
```

### 策略 2：使用選擇而非篩選

```python
# 效率較低：建立新 DataFrame
df_high = df[df.value > 100]
df_low = df[df.value <= 100]
mean_high = df_high.value.mean()
mean_low = df_low.value.mean()

# 更高效：使用選擇
df.select(df.value > 100, name='high')
df.select(df.value <= 100, name='low')
mean_high = df.value.mean(selection='high')
mean_low = df.value.mean(selection='low')
```

### 策略 3：批次聚合

```python
# 效率較低：多次遍歷
stats = {
    'mean': df.x.mean(),
    'std': df.x.std(),
    'min': df.x.min(),
    'max': df.x.max()
}

# 更高效：單次遍歷
delayed = [
    df.x.mean(delay=True),
    df.x.std(delay=True),
    df.x.min(delay=True),
    df.x.max(delay=True)
]
results = vaex.execute(delayed)
stats = dict(zip(['mean', 'std', 'min', 'max'], results))
```

### 策略 4：選擇最佳檔案格式

```python
# 慢：大型 CSV
df = vaex.from_csv('huge.csv')  # 可能需要幾分鐘

# 快：HDF5 或 Arrow
df = vaex.open('huge.hdf5')     # 即時
df = vaex.open('huge.arrow')    # 即時

# 一次性轉換
df = vaex.from_csv('huge.csv', convert='huge.hdf5')
# 未來載入：vaex.open('huge.hdf5')
```

### 策略 5：最佳化表達式

```python
# 效率較低：重複計算
df['result'] = df.x.log() + df.x.log() * 2

# 更高效：重用計算
df['log_x'] = df.x.log()
df['result'] = df.log_x + df.log_x * 2

# 更好：合併操作
df['result'] = df.x.log() * 3  # 簡化數學
```

## 效能分析

### 基本分析

```python
import time
import vaex

df = vaex.open('large_file.hdf5')

# 計時操作
start = time.time()
result = df.x.mean()
elapsed = time.time() - start
print(f"計算耗時 {elapsed:.2f} 秒")
```

### 詳細分析

```python
# 使用上下文管理器分析
with vaex.profiler():
    result = df.groupby('category').agg({'value': 'sum'})
# 列印詳細計時資訊
```

### 基準測試模式

```python
# 比較策略
def benchmark_operation(operation, name):
    start = time.time()
    result = operation()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}s")
    return result

# 測試不同方法
benchmark_operation(lambda: df.x.mean(), "直接平均")
benchmark_operation(lambda: df[df.x > 0].x.mean(), "篩選平均")
benchmark_operation(lambda: df.x.mean(selection='positive'), "選擇平均")
```

## 常見效能問題和解決方案

### 問題：聚合緩慢

```python
# 問題：多個分開的聚合
for col in df.column_names:
    print(f"{col}: {df[col].mean()}")

# 解決方案：使用 delay=True 批次處理
delayed = [df[col].mean(delay=True) for col in df.column_names]
results = vaex.execute(delayed)
for col, result in zip(df.column_names, results):
    print(f"{col}: {result}")
```

### 問題：高記憶體使用

```python
# 問題：實體化大型虛擬欄
df['large_col'] = (complex_expression).values

# 解決方案：保持虛擬，或實體化並匯出
df['large_col'] = complex_expression  # 虛擬
# 或：df.export_hdf5('with_new_col.hdf5')
```

### 問題：匯出緩慢

```python
# 問題：匯出時有許多虛擬欄
df.export_csv('output.csv')  # 如果有許多虛擬欄會很慢

# 解決方案：匯出為 HDF5 或 Arrow（更快）
df.export_hdf5('output.hdf5')
df.export_arrow('output.arrow')

# 或先實體化再匯出 CSV
df_materialized = df.materialize()
df_materialized.export_csv('output.csv')
```

### 問題：重複複雜計算

```python
# 問題：複雜虛擬欄重複使用
df['complex'] = df.x.log() * df.y.sqrt() + df.z ** 3
result1 = df.groupby('cat1').agg({'complex': 'mean'})
result2 = df.groupby('cat2').agg({'complex': 'sum'})
result3 = df.complex.std()

# 解決方案：實體化一次
df['complex'] = (df.x.log() * df.y.sqrt() + df.z ** 3).values
# 或：df = df.materialize('complex')
```

## 效能最佳實務總結

1. **使用 HDF5 或 Arrow 格式** - 比 CSV 快數個數量級
2. **利用惰性求值** - 除非必要，不要強制計算
3. **使用 delay=True 批次操作** - 最小化資料遍歷次數
4. **保持欄位為虛擬** - 只在有益時才實體化
5. **使用選擇而非篩選** - 對多個區段更高效
6. **分析您的程式碼** - 最佳化前先識別瓶頸
7. **避免 `.values` 和 `.to_pandas_df()`** - 將操作保持在 Vaex 中
8. **自然平行化** - Vaex 自動使用所有核心
9. **匯出為高效格式** - 為複雜管線建立檢查點
10. **最佳化表達式** - 簡化數學並重用計算

## 相關資源

- DataFrame 基礎：參見 `core_dataframes.md`
- 資料操作：參見 `data_processing.md`
- 檔案 I/O 最佳化：參見 `io_operations.md`
