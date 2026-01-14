# 核心 DataFrame 和資料載入

本參考文件涵蓋 Vaex DataFrame 基礎、從各種來源載入資料，以及理解 DataFrame 結構。

## DataFrame 基礎

Vaex DataFrame 是處理大型表格資料集的核心資料結構。與 pandas 不同，Vaex DataFrame：
- 使用**惰性求值（lazy evaluation）** - 操作直到需要時才執行
- **核外（out-of-core）**工作 - 資料不需要放入 RAM
- 支援**虛擬欄（virtual columns）** - 計算欄位無記憶體開銷
- 透過最佳化的 C++ 後端實現**每秒十億列**處理

## 開啟現有檔案

### 主要方法：`vaex.open()`

載入資料最常見的方式：

```python
import vaex

# 支援多種格式
df = vaex.open('data.hdf5')     # HDF5（推薦）
df = vaex.open('data.arrow')    # Apache Arrow（推薦）
df = vaex.open('data.parquet')  # Parquet
df = vaex.open('data.csv')      # CSV（大型檔案較慢）
df = vaex.open('data.fits')     # FITS（天文學）

# 可開啟多個檔案作為單一 DataFrame
df = vaex.open('data_*.hdf5')   # 支援萬用字元
```

**關鍵特性：**
- **HDF5/Arrow 即時載入** - 記憶體映射檔案，無載入時間
- **處理大型 CSV** - 自動分塊處理大型 CSV 檔案
- **立即回傳** - 惰性求值表示直到需要時才計算

### 格式特定載入器

```python
# 帶選項的 CSV
df = vaex.from_csv(
    'large_file.csv',
    chunk_size=5_000_000,      # 分塊處理
    convert=True,               # 自動轉換為 HDF5
    copy_index=False            # 不複製 pandas 索引（如果存在）
)

# Apache Arrow
df = vaex.open('data.arrow')    # 原生支援，非常快速

# HDF5（最佳格式）
df = vaex.open('data.hdf5')     # 透過記憶體映射即時載入
```

## 從其他來源建立 DataFrame

### 從 Pandas

```python
import pandas as pd
import vaex

# 轉換 pandas DataFrame
pdf = pd.read_csv('data.csv')
df = vaex.from_pandas(pdf, copy_index=False)

# 警告：這會將整個 pandas DataFrame 載入記憶體
# 對於大型資料，建議直接使用 vaex.from_csv()
```

### 從 NumPy 陣列

```python
import numpy as np
import vaex

# 單一陣列
x = np.random.rand(1_000_000)
df = vaex.from_arrays(x=x)

# 多個陣列
x = np.random.rand(1_000_000)
y = np.random.rand(1_000_000)
df = vaex.from_arrays(x=x, y=y)
```

### 從字典

```python
import vaex

# 列表/陣列的字典
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
df = vaex.from_dict(data)
```

### 從 Arrow 表格

```python
import pyarrow as pa
import vaex

# 從 Arrow Table
arrow_table = pa.table({
    'x': [1, 2, 3],
    'y': [4, 5, 6]
})
df = vaex.from_arrow_table(arrow_table)
```

## 範例資料集

Vaex 提供內建範例資料集供測試使用：

```python
import vaex

# NYC 計程車資料集（約 1GB，1100 萬列）
df = vaex.example()

# 較小的資料集
df = vaex.datasets.titanic()
df = vaex.datasets.iris()
```

## 檢視 DataFrame

### 基本資訊

```python
# 顯示首列和尾列
print(df)

# 形狀（列數，欄數）
print(df.shape)  # 回傳 (row_count, column_count)
print(len(df))   # 列數

# 欄位名稱
print(df.columns)
print(df.column_names)

# 資料類型
print(df.dtypes)

# 記憶體使用量（對於實體化欄位）
df.byte_size()
```

### 統計摘要

```python
# 所有數值欄位的快速統計
df.describe()

# 單一欄位統計
df.x.mean()
df.x.std()
df.x.min()
df.x.max()
df.x.sum()
df.x.count()

# 分位數
df.x.quantile(0.5)   # 中位數
df.x.quantile([0.25, 0.5, 0.75])  # 多個分位數
```

### 檢視資料

```python
# 首列/尾列（回傳 pandas DataFrame）
df.head(10)
df.tail(10)

# 隨機抽樣
df.sample(n=100)

# 轉換為 pandas（大型資料要小心！）
pdf = df.to_pandas_df()

# 只轉換特定欄位
pdf = df[['x', 'y']].to_pandas_df()
```

## DataFrame 結構

### 欄位

```python
# 以表達式存取欄位
x_column = df.x
y_column = df['y']

# 欄位操作回傳表達式（惰性）
sum_column = df.x + df.y    # 尚未計算

# 列出所有欄位
print(df.get_column_names())

# 檢查欄位類型
print(df.dtypes)

# 虛擬欄 vs 實體化欄
print(df.get_column_names(virtual=False))  # 僅實體化欄
print(df.get_column_names(virtual=True))   # 所有欄位
```

### 列

```python
# 列數
row_count = len(df)
row_count = df.count()

# 單一列（回傳字典）
row = df.row(0)
print(row['column_name'])

# 注意：在 Vaex 中不建議遍歷列
# 請改用向量化操作
```

## 使用表達式

表達式是 Vaex 表示尚未執行的計算的方式：

```python
# 建立表達式（無計算）
expr = df.x ** 2 + df.y

# 表達式可用於許多情境
mean_of_expr = expr.mean()          # 仍然是惰性的
df['new_col'] = expr                # 虛擬欄
filtered = df[expr > 10]            # 選擇

# 強制求值
result = expr.values  # 回傳 NumPy 陣列（謹慎使用！）
```

## DataFrame 操作

### 複製

```python
# 淺複製（共享資料）
df_copy = df.copy()

# 深複製（獨立資料）
df_deep = df.copy(deep=True)
```

### 裁切/切片

```python
# 選擇列範圍
df_subset = df[1000:2000]      # 第 1000-2000 列
df_subset = df[:1000]          # 前 1000 列
df_subset = df[-1000:]         # 最後 1000 列

# 注意：這建立視圖，不是複製（高效）
```

### 串接

```python
# 垂直串接（合併列）
df_combined = vaex.concat([df1, df2, df3])

# 水平串接（合併欄位）
# 使用 join 或直接指派欄位
df['new_col'] = other_df.some_column
```

## 最佳實務

1. **優先使用 HDF5 或 Arrow 格式** - 即時載入，最佳效能
2. **將大型 CSV 轉換為 HDF5** - 一次性轉換供重複使用
3. **避免對大型資料使用 `.to_pandas_df()`** - 違背 Vaex 的目的
4. **使用表達式而非 `.values`** - 保持操作惰性
5. **檢查資料類型** - 確保數值欄位不是字串類型
6. **使用虛擬欄** - 衍生資料零記憶體開銷

## 常見模式

### 模式：一次性 CSV 到 HDF5 轉換

```python
# 初始轉換（執行一次）
df = vaex.from_csv('large_data.csv', convert='large_data.hdf5')

# 未來載入（即時）
df = vaex.open('large_data.hdf5')
```

### 模式：檢視大型資料集

```python
import vaex

df = vaex.open('large_file.hdf5')

# 快速概覽
print(df)                    # 首列/尾列
print(df.shape)             # 維度
print(df.describe())        # 統計

# 抽樣進行詳細檢視
sample = df.sample(1000).to_pandas_df()
print(sample.head())
```

### 模式：載入多個檔案

```python
# 將多個檔案作為單一 DataFrame 載入
df = vaex.open('data_part*.hdf5')

# 或明確串接
df1 = vaex.open('data_2020.hdf5')
df2 = vaex.open('data_2021.hdf5')
df_all = vaex.concat([df1, df2])
```

## 常見問題和解決方案

### 問題：CSV 載入緩慢

```python
# 解決方案：先轉換為 HDF5
df = vaex.from_csv('large.csv', convert='large.hdf5')
# 未來載入：df = vaex.open('large.hdf5')
```

### 問題：欄位顯示為字串類型

```python
# 檢查類型
print(df.dtypes)

# 轉換為數值（建立虛擬欄）
df['age_numeric'] = df.age.astype('int64')
```

### 問題：小型操作記憶體不足

```python
# 可能使用了 .values 或 .to_pandas_df()
# 解決方案：使用惰性操作

# 不好（載入記憶體）
array = df.x.values

# 好（保持惰性）
mean = df.x.mean()
filtered = df[df.x > 10]
```

## 相關資源

- 資料操作和篩選：參見 `data_processing.md`
- 效能最佳化：參見 `performance.md`
- 檔案格式詳情：參見 `io_operations.md`
