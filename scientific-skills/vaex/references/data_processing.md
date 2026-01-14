# 資料處理和操作

本參考文件涵蓋 Vaex 中的篩選、選擇、虛擬欄、表達式、聚合、groupby 操作和資料轉換。

## 篩選和選擇

Vaex 使用布林表達式高效篩選資料而無需複製：

### 基本篩選

```python
# 簡單篩選
df_filtered = df[df.age > 25]

# 多重條件
df_filtered = df[(df.age > 25) & (df.salary > 50000)]
df_filtered = df[(df.category == 'A') | (df.category == 'B')]

# 否定
df_filtered = df[~(df.age < 18)]
```

### 選擇物件

Vaex 可同時維護多個命名選擇：

```python
# 建立命名選擇
df.select(df.age > 30, name='adults')
df.select(df.salary > 100000, name='high_earners')

# 在操作中使用選擇
mean_age_adults = df.mean(df.age, selection='adults')
count_high_earners = df.count(selection='high_earners')

# 組合選擇
df.select((df.age > 30) & (df.salary > 100000), name='adult_high_earners')

# 列出所有選擇
print(df.selection_names())

# 刪除選擇
df.select_drop('adults')
```

### 進階篩選

```python
# 字串匹配
df_filtered = df[df.name.str.contains('John')]
df_filtered = df[df.name.str.startswith('A')]
df_filtered = df[df.email.str.endswith('@gmail.com')]

# 空值/缺失值篩選
df_filtered = df[df.age.isna()]      # 保留缺失
df_filtered = df[df.age.notna()]     # 移除缺失

# 值成員資格
df_filtered = df[df.category.isin(['A', 'B', 'C'])]

# 範圍篩選
df_filtered = df[df.age.between(25, 65)]
```

## 虛擬欄和表達式

虛擬欄即時計算，零記憶體開銷：

### 建立虛擬欄

```python
# 算術運算
df['total'] = df.price * df.quantity
df['price_squared'] = df.price ** 2

# 數學函數
df['log_price'] = df.price.log()
df['sqrt_value'] = df.value.sqrt()
df['abs_diff'] = (df.x - df.y).abs()

# 條件邏輯
df['is_adult'] = df.age >= 18
df['category'] = (df.score > 80).where('A', 'B')  # If-then-else
```

### 表達式方法

```python
# 數學
df.x.abs()          # 絕對值
df.x.sqrt()         # 平方根
df.x.log()          # 自然對數
df.x.log10()        # 以 10 為底對數
df.x.exp()          # 指數

# 三角函數
df.angle.sin()
df.angle.cos()
df.angle.tan()
df.angle.arcsin()

# 捨入
df.x.round(2)       # 捨入到小數點後 2 位
df.x.floor()        # 向下捨入
df.x.ceil()         # 向上捨入

# 類型轉換
df.x.astype('int64')
df.x.astype('float32')
df.x.astype('str')
```

### 條件表達式

```python
# where() 方法：condition.where(true_value, false_value)
df['status'] = (df.age >= 18).where('adult', 'minor')

# 使用巢狀 where 處理多重條件
df['grade'] = (df.score >= 90).where('A',
              (df.score >= 80).where('B',
              (df.score >= 70).where('C', 'F')))

# 使用 searchsorted 進行分箱
bins = [0, 18, 65, 100]
labels = ['minor', 'adult', 'senior']
df['age_group'] = df.age.searchsorted(bins).where(...)
```

## 字串操作

透過 `.str` 存取器使用字串方法：

### 基本字串方法

```python
# 大小寫轉換
df['upper_name'] = df.name.str.upper()
df['lower_name'] = df.name.str.lower()
df['title_name'] = df.name.str.title()

# 修剪
df['trimmed'] = df.text.str.strip()
df['ltrimmed'] = df.text.str.lstrip()
df['rtrimmed'] = df.text.str.rstrip()

# 搜尋
df['has_john'] = df.name.str.contains('John')
df['starts_with_a'] = df.name.str.startswith('A')
df['ends_with_com'] = df.email.str.endswith('.com')

# 切片
df['first_char'] = df.name.str.slice(0, 1)
df['last_three'] = df.name.str.slice(-3, None)

# 長度
df['name_length'] = df.name.str.len()
```

### 進階字串操作

```python
# 替換
df['clean_text'] = df.text.str.replace('bad', 'good')

# 分割（回傳第一部分）
df['first_name'] = df.full_name.str.split(' ')[0]

# 串接
df['full_name'] = df.first_name + ' ' + df.last_name

# 填充
df['padded'] = df.code.str.pad(10, '0', 'left')  # 零填充
```

## 日期時間操作

透過 `.dt` 存取器使用日期時間方法：

### 日期時間屬性

```python
# 解析字串為日期時間
df['date_parsed'] = df.date_string.astype('datetime64')

# 提取組件
df['year'] = df.timestamp.dt.year
df['month'] = df.timestamp.dt.month
df['day'] = df.timestamp.dt.day
df['hour'] = df.timestamp.dt.hour
df['minute'] = df.timestamp.dt.minute
df['second'] = df.timestamp.dt.second

# 星期幾
df['weekday'] = df.timestamp.dt.dayofweek  # 0=星期一
df['day_name'] = df.timestamp.dt.day_name  # 'Monday', 'Tuesday', ...

# 日期運算
df['tomorrow'] = df.date + pd.Timedelta(days=1)
df['next_week'] = df.date + pd.Timedelta(weeks=1)
```

## 聚合

Vaex 高效地對數十億列執行聚合：

### 基本聚合

```python
# 單一欄位
mean_age = df.age.mean()
std_age = df.age.std()
min_age = df.age.min()
max_age = df.age.max()
sum_sales = df.sales.sum()
count_rows = df.count()

# 使用選擇
mean_adult_age = df.age.mean(selection='adults')

# 使用 delay 同時計算多個
mean = df.age.mean(delay=True)
std = df.age.std(delay=True)
results = vaex.execute([mean, std])
```

### 可用的聚合函數

```python
# 集中趨勢
df.x.mean()
df.x.median_approx()  # 近似中位數（快速）

# 離散程度
df.x.std()           # 標準差
df.x.var()           # 變異數
df.x.min()
df.x.max()
df.x.minmax()        # 同時取得最小值和最大值

# 計數
df.count()           # 總列數
df.x.count()         # 非缺失值

# 總和與乘積
df.x.sum()
df.x.prod()

# 百分位數
df.x.quantile(0.5)           # 中位數
df.x.quantile([0.25, 0.75])  # 四分位數

# 相關性
df.correlation(df.x, df.y)
df.covar(df.x, df.y)

# 高階動差
df.x.kurtosis()
df.x.skew()

# 唯一值
df.x.nunique()       # 計算唯一值數量
df.x.unique()        # 取得唯一值（回傳陣列）
```

## GroupBy 操作

分組資料並計算每組的聚合：

### 基本 GroupBy

```python
# 單一欄位 groupby
grouped = df.groupby('category')

# 聚合
result = grouped.agg({'sales': 'sum'})
result = grouped.agg({'sales': 'sum', 'quantity': 'mean'})

# 對同一欄位進行多個聚合
result = grouped.agg({
    'sales': ['sum', 'mean', 'std'],
    'quantity': 'sum'
})
```

### 進階 GroupBy

```python
# 多個分組欄位
result = df.groupby(['category', 'region']).agg({
    'sales': 'sum',
    'quantity': 'mean'
})

# 自訂聚合函數
result = df.groupby('category').agg({
    'sales': lambda x: x.max() - x.min()
})

# 可用的聚合函數
# 'sum', 'mean', 'std', 'min', 'max', 'count', 'first', 'last'
```

### 帶分箱的 GroupBy

```python
# 分箱連續變數並聚合
result = df.groupby(vaex.vrange(0, 100, 10)).agg({
    'sales': 'sum'
})

# 日期時間分箱
result = df.groupby(df.timestamp.dt.year).agg({
    'sales': 'sum'
})
```

## 分箱和離散化

從連續變數建立分箱：

### 簡單分箱

```python
# 建立分箱
df['age_bin'] = df.age.digitize([18, 30, 50, 65, 100])

# 標籤分箱
bins = [0, 18, 30, 50, 65, 100]
labels = ['child', 'young_adult', 'adult', 'middle_age', 'senior']
df['age_group'] = df.age.digitize(bins)
# 注意：使用 where() 或映射套用標籤
```

### 統計分箱

```python
# 等寬分箱
df['value_bin'] = df.value.digitize(
    vaex.vrange(df.value.min(), df.value.max(), 10)
)

# 基於分位數的分箱
quantiles = df.value.quantile([0.25, 0.5, 0.75])
df['value_quartile'] = df.value.digitize(quantiles)
```

## 多維聚合

在網格上計算統計：

```python
# 2D 直方圖/熱圖資料
counts = df.count(binby=[df.x, df.y], limits=[[0, 10], [0, 10]], shape=(100, 100))

# 網格上的平均值
mean_z = df.mean(df.z, binby=[df.x, df.y], limits=[[0, 10], [0, 10]], shape=(50, 50))

# 網格上的多個統計
stats = df.mean(df.z, binby=[df.x, df.y], shape=(50, 50), delay=True)
counts = df.count(binby=[df.x, df.y], shape=(50, 50), delay=True)
results = vaex.execute([stats, counts])
```

## 處理缺失資料

處理缺失、空值和 NaN 值：

### 偵測缺失資料

```python
# 檢查缺失
df['age_missing'] = df.age.isna()
df['age_present'] = df.age.notna()

# 計算缺失
missing_count = df.age.isna().sum()
missing_pct = df.age.isna().mean() * 100
```

### 處理缺失資料

```python
# 篩選掉缺失
df_clean = df[df.age.notna()]

# 用值填充缺失
df['age_filled'] = df.age.fillna(0)
df['age_filled'] = df.age.fillna(df.age.mean())

# 前向/後向填充（用於時間序列）
df['age_ffill'] = df.age.fillna(method='ffill')
df['age_bfill'] = df.age.fillna(method='bfill')
```

### Vaex 中的缺失資料類型

Vaex 區分以下類型：
- **NaN** - IEEE 浮點數 Not-a-Number
- **NA** - Arrow 空值類型
- **Missing** - 缺失資料的通用術語

```python
# 檢查是哪種缺失類型
df.is_masked('column_name')  # 如果使用 Arrow null (NA) 則為 True

# 類型之間轉換
df['col_masked'] = df.col.as_masked()  # 轉換為 NA 表示
```

## 排序

```python
# 按單一欄位排序
df_sorted = df.sort('age')
df_sorted = df.sort('age', ascending=False)

# 按多個欄位排序
df_sorted = df.sort(['category', 'age'])

# 注意：排序會實體化一個帶有索引的新欄位
# 對於非常大的資料集，考慮是否真的需要排序
```

## 合併 DataFrame

基於鍵合併 DataFrame：

```python
# 內部合併
df_joined = df1.join(df2, on='key_column')

# 左合併
df_joined = df1.join(df2, on='key_column', how='left')

# 在不同欄位名稱上合併
df_joined = df1.join(
    df2,
    left_on='id',
    right_on='user_id',
    how='left'
)

# 多個鍵欄位
df_joined = df1.join(df2, on=['key1', 'key2'])
```

## 新增和移除欄位

### 新增欄位

```python
# 虛擬欄（無記憶體）
df['new_col'] = df.x + df.y

# 從外部陣列（必須匹配長度）
import numpy as np
new_data = np.random.rand(len(df))
df['random'] = new_data

# 常數值
df['constant'] = 42
```

### 移除欄位

```python
# 刪除單一欄位
df = df.drop('column_name')

# 刪除多個欄位
df = df.drop(['col1', 'col2', 'col3'])

# 選擇特定欄位（刪除其他）
df = df[['col1', 'col2', 'col3']]
```

### 重新命名欄位

```python
# 重新命名單一欄位
df = df.rename('old_name', 'new_name')

# 重新命名多個欄位
df = df.rename({
    'old_name1': 'new_name1',
    'old_name2': 'new_name2'
})
```

## 常見模式

### 模式：複雜特徵工程

```python
# 多個衍生特徵
df['log_price'] = df.price.log()
df['price_per_unit'] = df.price / df.quantity
df['is_discount'] = df.discount > 0
df['price_category'] = (df.price > 100).where('expensive', 'affordable')
df['revenue'] = df.price * df.quantity * (1 - df.discount)
```

### 模式：文字清理

```python
# 清理和標準化文字
df['email_clean'] = df.email.str.lower().str.strip()
df['has_valid_email'] = df.email_clean.str.contains('@')
df['domain'] = df.email_clean.str.split('@')[1]
```

### 模式：基於時間的分析

```python
# 提取時間特徵
df['year'] = df.timestamp.dt.year
df['month'] = df.timestamp.dt.month
df['day_of_week'] = df.timestamp.dt.dayofweek
df['is_weekend'] = df.day_of_week >= 5
df['quarter'] = ((df.month - 1) // 3) + 1
```

### 模式：分組統計

```python
# 按組計算統計
monthly_sales = df.groupby(df.timestamp.dt.month).agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': 'sum'
})

# 多層分組
category_region_sales = df.groupby(['category', 'region']).agg({
    'sales': 'sum',
    'profit': 'mean'
})
```

## 效能提示

1. **使用虛擬欄** - 它們即時計算，無記憶體成本
2. **使用 delay=True 批次操作** - 一次計算多個聚合
3. **避免 `.values` 或 `.to_pandas_df()`** - 盡可能保持操作惰性
4. **使用選擇** - 多個命名選擇比建立新 DataFrame 更高效
5. **利用表達式** - 它們啟用查詢最佳化
6. **最小化排序** - 排序在大型資料集上很昂貴

## 相關資源

- DataFrame 建立：參見 `core_dataframes.md`
- 效能最佳化：參見 `performance.md`
- 視覺化：參見 `visualization.md`
- 機器學習管線：參見 `machine_learning.md`
