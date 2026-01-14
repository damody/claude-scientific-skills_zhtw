# Dask Bags

## 概述

Dask Bag 在通用 Python 物件上實作函數式操作，包括 `map`、`filter`、`fold` 和 `groupby`。它透過 Python 迭代器平行處理資料，同時維持較小的記憶體佔用。Bags 的功能類似於「PyToolz 的平行版本或 PySpark RDD 的 Pythonic 版本」。

## 核心概念

Dask Bag 是分散在多個分區中的 Python 物件集合：
- 每個分區包含通用 Python 物件
- 操作使用函數式程式設計模式
- 處理使用串流/迭代器以提高記憶體效率
- 適合非結構化或半結構化資料

## 主要功能

### 函數式操作
- `map`：轉換每個元素
- `filter`：根據條件選擇元素
- `fold`：使用組合函數歸約元素
- `groupby`：按鍵分組元素
- `pluck`：從記錄中提取欄位
- `flatten`：展平巢狀結構

### 使用情境
- 文字處理和日誌分析
- JSON 記錄處理
- 非結構化資料的 ETL
- 結構化分析前的資料清理

## 何時使用 Dask Bags

**使用 Bags 的情況**：
- 處理需要靈活運算的通用 Python 物件
- 資料不適合結構化陣列或表格格式
- 處理文字、JSON 或自訂 Python 物件
- 需要初始資料清理和 ETL
- 記憶體高效的串流很重要

**使用其他集合的情況**：
- 資料是結構化的（改用 DataFrames）
- 數值運算（改用 Arrays）
- 操作需要複雜的 groupby 或 shuffle（改用 DataFrames）

**關鍵建議**：使用 Bag 來清理和處理資料，然後在進行需要 shuffle 步驟的更複雜操作之前將其轉換為陣列或 DataFrame。

## 重要限制

Bags 為了通用性犧牲了效能：
- 依賴多程序排程（非執行緒）
- 保持不可變（變更會建立新的 bags）
- 比陣列/DataFrame 等效操作慢
- `groupby` 處理效率低（盡可能使用 `foldby`）
- 需要大量工作者間通訊的操作較慢

## 建立 Bags

### 從序列建立
```python
import dask.bag as db

# 從 Python 列表建立
bag = db.from_sequence([1, 2, 3, 4, 5], partition_size=2)

# 從 range 建立
bag = db.from_sequence(range(10000), partition_size=1000)
```

### 從文字檔案建立
```python
# 單一檔案
bag = db.read_text('data.txt')

# 使用 glob 處理多個檔案
bag = db.read_text('data/*.txt')

# 指定編碼
bag = db.read_text('data/*.txt', encoding='utf-8')

# 自訂行處理
bag = db.read_text('logs/*.log', blocksize='64MB')
```

### 從 Delayed 物件建立
```python
import dask

@dask.delayed
def load_data(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

files = ['file1.txt', 'file2.txt', 'file3.txt']
partitions = [load_data(f) for f in files]
bag = db.from_delayed(partitions)
```

### 從自訂來源建立
```python
# 從任何產生可迭代物件的函數
def read_json_files():
    import json
    for filename in glob.glob('data/*.json'):
        with open(filename) as f:
            yield json.load(f)

# 從生成器建立 bag
bag = db.from_sequence(read_json_files(), partition_size=10)
```

## 常見操作

### Map（轉換）
```python
import dask.bag as db

bag = db.read_text('data/*.json')

# 解析 JSON
import json
parsed = bag.map(json.loads)

# 提取欄位
values = parsed.map(lambda x: x['value'])

# 複雜轉換
def process_record(record):
    return {
        'id': record['id'],
        'value': record['value'] * 2,
        'category': record.get('category', 'unknown')
    }

processed = parsed.map(process_record)
```

### Filter
```python
# 按條件篩選
valid = parsed.filter(lambda x: x['status'] == 'valid')

# 多個條件
filtered = parsed.filter(lambda x: x['value'] > 100 and x['year'] == 2024)

# 使用自訂函數篩選
def is_valid_record(record):
    return record.get('status') == 'valid' and record.get('value') is not None

valid_records = parsed.filter(is_valid_record)
```

### Pluck（提取欄位）
```python
# 提取單一欄位
ids = parsed.pluck('id')

# 提取多個欄位（建立元組）
key_pairs = parsed.pluck(['id', 'value'])
```

### Flatten
```python
# 展平巢狀列表
nested = db.from_sequence([[1, 2], [3, 4], [5, 6]])
flat = nested.flatten()  # [1, 2, 3, 4, 5, 6]

# 在 map 後展平
bag = db.read_text('data/*.txt')
words = bag.map(str.split).flatten()  # 所有檔案中的所有單詞
```

### GroupBy（昂貴操作）
```python
# 按鍵分組（需要 shuffle）
grouped = parsed.groupby(lambda x: x['category'])

# 分組後聚合
counts = grouped.map(lambda key_items: (key_items[0], len(list(key_items[1]))))
result = counts.compute()
```

### FoldBy（聚合的首選方式）
```python
# FoldBy 對於聚合比 groupby 更高效
def add(acc, item):
    return acc + item['value']

def combine(acc1, acc2):
    return acc1 + acc2

# 按類別加總值
sums = parsed.foldby(
    key='category',
    binop=add,
    initial=0,
    combine=combine
)

result = sums.compute()
```

### 歸約操作
```python
# 計算元素數量
count = bag.count().compute()

# 取得所有不重複值（需要記憶體）
distinct = bag.distinct().compute()

# 取前 n 個元素
first_ten = bag.take(10)

# Fold/reduce
total = bag.fold(
    lambda acc, x: acc + x['value'],
    initial=0,
    combine=lambda a, b: a + b
).compute()
```

## 轉換為其他集合

### 轉為 DataFrame
```python
import dask.bag as db
import dask.dataframe as dd

# 字典組成的 Bag
bag = db.read_text('data/*.json').map(json.loads)

# 轉換為 DataFrame
ddf = bag.to_dataframe()

# 指定明確的欄位
ddf = bag.to_dataframe(meta={'id': int, 'value': float, 'category': str})
```

### 轉為列表/Compute
```python
# 計算為 Python 列表（載入所有到記憶體）
result = bag.compute()

# 取樣本
sample = bag.take(100)
```

## 常見模式

### JSON 處理
```python
import dask.bag as db
import json

# 讀取並解析 JSON 檔案
bag = db.read_text('logs/*.json')
parsed = bag.map(json.loads)

# 篩選有效記錄
valid = parsed.filter(lambda x: x.get('status') == 'success')

# 提取相關欄位
processed = valid.map(lambda x: {
    'user_id': x['user']['id'],
    'timestamp': x['timestamp'],
    'value': x['metrics']['value']
})

# 轉換為 DataFrame 進行分析
ddf = processed.to_dataframe()

# 分析
summary = ddf.groupby('user_id')['value'].mean().compute()
```

### 日誌分析
```python
# 讀取日誌檔案
logs = db.read_text('logs/*.log')

# 解析日誌行
def parse_log_line(line):
    parts = line.split(' ')
    return {
        'timestamp': parts[0],
        'level': parts[1],
        'message': ' '.join(parts[2:])
    }

parsed_logs = logs.map(parse_log_line)

# 篩選錯誤
errors = parsed_logs.filter(lambda x: x['level'] == 'ERROR')

# 按訊息模式計數
error_counts = errors.foldby(
    key='message',
    binop=lambda acc, x: acc + 1,
    initial=0,
    combine=lambda a, b: a + b
)

result = error_counts.compute()
```

### 文字處理
```python
# 讀取文字檔案
text = db.read_text('documents/*.txt')

# 分割為單詞
words = text.map(str.lower).map(str.split).flatten()

# 計算單詞頻率
def increment(acc, word):
    return acc + 1

def combine_counts(a, b):
    return a + b

word_counts = words.foldby(
    key=lambda word: word,
    binop=increment,
    initial=0,
    combine=combine_counts
)

# 取得最常見的單詞
top_words = word_counts.compute()
sorted_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:100]
```

### 資料清理管道
```python
import dask.bag as db
import json

# 讀取原始資料
raw = db.read_text('raw_data/*.json').map(json.loads)

# 驗證函數
def is_valid(record):
    required_fields = ['id', 'timestamp', 'value']
    return all(field in record for field in required_fields)

# 清理函數
def clean_record(record):
    return {
        'id': int(record['id']),
        'timestamp': record['timestamp'],
        'value': float(record['value']),
        'category': record.get('category', 'unknown'),
        'tags': record.get('tags', [])
    }

# 管道
cleaned = (raw
    .filter(is_valid)
    .map(clean_record)
    .filter(lambda x: x['value'] > 0)
)

# 轉換為 DataFrame
ddf = cleaned.to_dataframe()

# 儲存清理後的資料
ddf.to_parquet('cleaned_data/')
```

## 效能考量

### 高效操作
- Map、filter、pluck：非常高效（串流）
- Flatten：高效
- FoldBy 搭配良好的鍵分佈：合理
- Take 和 head：高效（僅處理需要的分區）

### 昂貴操作
- GroupBy：需要 shuffle，可能很慢
- Distinct：需要收集所有唯一值
- 需要完整資料具體化的操作

### 優化技巧

**1. 使用 FoldBy 代替 GroupBy**
```python
# 較佳：使用 foldby 進行聚合
result = bag.foldby(key='category', binop=add, initial=0, combine=sum)

# 較差：GroupBy 然後 reduce
result = bag.groupby('category').map(lambda x: (x[0], sum(x[1])))
```

**2. 儘早轉換為 DataFrame**
```python
# 對於結構化操作，轉換為 DataFrame
bag = db.read_text('data/*.json').map(json.loads)
bag = bag.filter(lambda x: x['status'] == 'valid')
ddf = bag.to_dataframe()  # 現在使用高效的 DataFrame 操作
```

**3. 控制分區大小**
```python
# 在過多和過少分區之間取得平衡
bag = db.read_text('data/*.txt', blocksize='64MB')  # 合理的分區大小
```

**4. 使用延遲求值**
```python
# 在計算前串連操作
result = (bag
    .map(process1)
    .filter(condition)
    .map(process2)
    .compute()  # 最後單次 compute
)
```

## 除錯技巧

### 檢查分區
```python
# 取得分區數量
print(bag.npartitions)

# 取樣本
sample = bag.take(10)
print(sample)
```

### 在小資料上驗證
```python
# 在小子集上測試邏輯
small_bag = db.from_sequence(sample_data, partition_size=10)
result = process_pipeline(small_bag).compute()
# 驗證結果，然後擴展
```

### 檢查中間結果
```python
# 計算中間步驟以除錯
step1 = bag.map(parse).take(5)
print("After parsing:", step1)

step2 = bag.map(parse).filter(validate).take(5)
print("After filtering:", step2)
```

## 記憶體管理

Bags 設計用於記憶體高效的處理：

```python
# 串流處理 - 不會將所有資料載入記憶體
bag = db.read_text('huge_file.txt')  # 延遲
processed = bag.map(process_line)     # 仍然延遲
result = processed.compute()          # 分塊處理
```

對於非常大的結果，避免計算到記憶體：

```python
# 不要將巨大的結果計算到記憶體
# result = bag.compute()  # 可能會溢出記憶體

# 相反，轉換並儲存到磁碟
ddf = bag.to_dataframe()
ddf.to_parquet('output/')
```
