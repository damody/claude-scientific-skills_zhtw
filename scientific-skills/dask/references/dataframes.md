# Dask DataFrames

## 概述

Dask DataFrames 透過將工作分散到多個 pandas DataFrames 來實現大型表格資料的平行處理。如文件所述，「Dask DataFrames 是多個 pandas DataFrames 的集合」，具有相同的 API，使得從 pandas 的轉換變得簡單直接。

## 核心概念

Dask DataFrame 沿著索引被分成多個 pandas DataFrames（分區）：
- 每個分區是一個常規的 pandas DataFrame
- 操作以平行方式應用於每個分區
- 結果自動組合

## 主要功能

### 規模
- 在筆記型電腦上處理 100 GiB
- 在叢集上處理 100 TiB
- 處理超過可用 RAM 的資料集

### 相容性
- 實作大部分 pandas API
- 從 pandas 程式碼輕鬆轉換
- 使用熟悉的操作

## 何時使用 Dask DataFrames

**使用 Dask 的情況**：
- 資料集超過可用 RAM
- 運算需要大量時間且 pandas 優化沒有幫助
- 需要從原型（pandas）擴展到生產環境（更大資料）
- 處理應該一起處理的多個檔案

**繼續使用 Pandas 的情況**：
- 資料可以輕鬆放入記憶體
- 運算在次秒級完成
- 沒有自訂 `.apply()` 函數的簡單操作
- 迭代開發和探索

## 讀取資料

Dask 鏡像 pandas 的讀取語法，並增加了對多個檔案的支援：

### 單一檔案
```python
import dask.dataframe as dd

# 讀取單一檔案
ddf = dd.read_csv('data.csv')
ddf = dd.read_parquet('data.parquet')
```

### 多個檔案
```python
# 使用 glob 模式讀取多個檔案
ddf = dd.read_csv('data/*.csv')
ddf = dd.read_parquet('s3://mybucket/data/*.parquet')

# 使用路徑結構讀取
ddf = dd.read_parquet('data/year=*/month=*/day=*.parquet')
```

### 優化
```python
# 指定要讀取的欄位（減少記憶體）
ddf = dd.read_parquet('data.parquet', columns=['col1', 'col2'])

# 控制分區
ddf = dd.read_csv('data.csv', blocksize='64MB')  # 建立 64MB 分區
```

## 常見操作

所有操作都是延遲的，直到呼叫 `.compute()`。

### 篩選
```python
# 與 pandas 相同
filtered = ddf[ddf['column'] > 100]
filtered = ddf.query('column > 100')
```

### 欄位操作
```python
# 新增欄位
ddf['new_column'] = ddf['col1'] + ddf['col2']

# 選擇欄位
subset = ddf[['col1', 'col2', 'col3']]

# 刪除欄位
ddf = ddf.drop(columns=['unnecessary_col'])
```

### 聚合
```python
# 標準聚合按預期運作
mean = ddf['column'].mean().compute()
sum_total = ddf['column'].sum().compute()
counts = ddf['category'].value_counts().compute()
```

### GroupBy
```python
# GroupBy 操作（可能需要 shuffle）
grouped = ddf.groupby('category')['value'].mean().compute()

# 多個聚合
agg_result = ddf.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'amount': 'sum'
}).compute()
```

### Join 和 Merge
```python
# 合併 DataFrames
merged = dd.merge(ddf1, ddf2, on='key', how='left')

# 在索引上 join
joined = ddf1.join(ddf2, on='key')
```

### 排序
```python
# 排序（昂貴操作，需要資料移動）
sorted_ddf = ddf.sort_values('column')
result = sorted_ddf.compute()
```

## 自訂操作

### 應用函數

**對分區應用（高效）**：
```python
# 對整個分區應用函數
def custom_partition_function(partition_df):
    # partition_df 是一個 pandas DataFrame
    return partition_df.assign(new_col=partition_df['col1'] * 2)

ddf = ddf.map_partitions(custom_partition_function)
```

**對行應用（效率較低）**：
```python
# 對每一行應用（建立很多任務）
ddf['result'] = ddf.apply(lambda row: custom_function(row), axis=1, meta=('result', 'float'))
```

**注意**：為了更好的效能，始終優先使用 `map_partitions` 而不是逐行的 `apply`。

### Meta 參數

當 Dask 無法推斷輸出結構時，指定 `meta` 參數：
```python
# 用於 apply 操作
ddf['new'] = ddf.apply(func, axis=1, meta=('new', 'float64'))

# 用於 map_partitions
ddf = ddf.map_partitions(func, meta=pd.DataFrame({
    'col1': pd.Series(dtype='float64'),
    'col2': pd.Series(dtype='int64')
}))
```

## 延遲求值與運算

### 延遲操作
```python
# 這些操作是延遲的（即時、無運算）
filtered = ddf[ddf['value'] > 100]
aggregated = filtered.groupby('category').mean()
final = aggregated[aggregated['value'] < 500]

# 尚未運算
```

### 觸發運算
```python
# 計算單一結果
result = final.compute()

# 高效計算多個結果
result1, result2, result3 = dask.compute(
    operation1,
    operation2,
    operation3
)
```

### 持久化到記憶體
```python
# 將結果保留在分散式記憶體中以供重複使用
ddf_cached = ddf.persist()

# 現在對 ddf_cached 的多個操作不會重新計算
result1 = ddf_cached.mean().compute()
result2 = ddf_cached.sum().compute()
```

## 索引管理

### 設定索引
```python
# 設定索引（高效 join 和某些操作所需）
ddf = ddf.set_index('timestamp', sorted=True)
```

### 索引屬性
- 排序的索引支援高效的篩選和 join
- 索引決定分區
- 某些操作在適當的索引下效能更好

## 寫入結果

### 寫入檔案
```python
# 寫入多個檔案（每個分區一個）
ddf.to_parquet('output/data.parquet')
ddf.to_csv('output/data-*.csv')

# 寫入單一檔案（強制運算和串接）
ddf.compute().to_csv('output/single_file.csv')
```

### 寫入記憶體（Pandas）
```python
# 轉換為 pandas（將所有資料載入記憶體）
pdf = ddf.compute()
```

## 效能考量

### 高效操作
- 欄位選擇和篩選：非常高效
- 簡單聚合（sum、mean、count）：高效
- 分區上的逐行操作：使用 `map_partitions` 時高效

### 昂貴操作
- 排序：需要跨 worker 的資料 shuffle
- 具有很多群組的 GroupBy：可能需要 shuffle
- 複雜 join：取決於資料分佈
- 逐行 apply：建立很多任務

### 優化技巧

**1. 儘早選擇欄位**
```python
# 較佳：只讀取需要的欄位
ddf = dd.read_parquet('data.parquet', columns=['col1', 'col2'])
```

**2. 在 GroupBy 之前篩選**
```python
# 較佳：在昂貴操作之前減少資料
result = ddf[ddf['year'] == 2024].groupby('category').sum().compute()
```

**3. 使用高效的檔案格式**
```python
# 使用 Parquet 代替 CSV 以獲得更好的效能
ddf.to_parquet('data.parquet')  # 更快、更小、列式
```

**4. 適當重新分區**
```python
# 如果分區太小
ddf = ddf.repartition(npartitions=10)

# 如果分區太大
ddf = ddf.repartition(partition_size='100MB')
```

## 常見模式

### ETL 管道
```python
import dask.dataframe as dd

# 讀取資料
ddf = dd.read_csv('raw_data/*.csv')

# 轉換
ddf = ddf[ddf['status'] == 'valid']
ddf['amount'] = ddf['amount'].astype('float64')
ddf = ddf.dropna(subset=['important_col'])

# 聚合
summary = ddf.groupby('category').agg({
    'amount': ['sum', 'mean'],
    'quantity': 'count'
})

# 寫入結果
summary.to_parquet('output/summary.parquet')
```

### 時間序列分析
```python
# 讀取時間序列資料
ddf = dd.read_parquet('timeseries/*.parquet')

# 設定時間戳索引
ddf = ddf.set_index('timestamp', sorted=True)

# 重新取樣（如果 Dask 版本支援）
hourly = ddf.resample('1H').mean()

# 計算統計
result = hourly.compute()
```

### 合併多個檔案
```python
# 將多個檔案讀取為單一 DataFrame
ddf = dd.read_csv('data/2024-*.csv')

# 處理合併的資料
result = ddf.groupby('category')['value'].sum().compute()
```

## 與 Pandas 的限制和差異

### 並非所有 Pandas 功能都可用
一些 pandas 操作在 Dask 中未實作：
- 一些字串方法
- 某些視窗函數
- 一些專門的統計函數

### 分區很重要
- 分區內的操作是高效的
- 跨分區操作可能很昂貴
- 基於索引的操作受益於排序的索引

### 延遲求值
- 操作在 `.compute()` 之前不會執行
- 需要注意運算觸發
- 不運算無法檢查中間結果

## 除錯技巧

### 檢查分區
```python
# 取得分區數量
print(ddf.npartitions)

# 計算單一分區
first_partition = ddf.get_partition(0).compute()

# 查看前幾行（計算第一個分區）
print(ddf.head())
```

### 在小資料上驗證操作
```python
# 先在小樣本上測試
sample = ddf.head(1000)
# 驗證邏輯有效
# 然後擴展到完整資料集
result = ddf.compute()
```

### 檢查 Dtypes
```python
# 驗證資料類型正確
print(ddf.dtypes)
```
