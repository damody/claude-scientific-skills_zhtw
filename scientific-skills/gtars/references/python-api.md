# Python API 參考

gtars Python 綁定的完整參考。

## 安裝

```bash
# 安裝 gtars Python 套件
uv pip install gtars

# 或使用 pip
pip install gtars
```

## 核心類別

### RegionSet

管理基因體區間集合：

```python
import gtars

# 從 BED 檔案建立
regions = gtars.RegionSet.from_bed("regions.bed")

# 從座標建立
regions = gtars.RegionSet([
    ("chr1", 1000, 2000),
    ("chr1", 3000, 4000),
    ("chr2", 5000, 6000)
])

# 存取區域
for region in regions:
    print(f"{region.chromosome}:{region.start}-{region.end}")

# 取得區域數量
num_regions = len(regions)

# 取得總涵蓋度
total_coverage = regions.total_coverage()
```

### 區域操作

對區域集執行操作：

```python
# 排序區域
sorted_regions = regions.sort()

# 合併重疊區域
merged = regions.merge()

# 依大小篩選
large_regions = regions.filter_by_size(min_size=1000)

# 依染色體篩選
chr1_regions = regions.filter_by_chromosome("chr1")
```

### 集合操作

對基因體區域執行集合操作：

```python
# 載入兩個區域集
set_a = gtars.RegionSet.from_bed("set_a.bed")
set_b = gtars.RegionSet.from_bed("set_b.bed")

# 聯集
union = set_a.union(set_b)

# 交集
intersection = set_a.intersect(set_b)

# 差集
difference = set_a.subtract(set_b)

# 對稱差集
sym_diff = set_a.symmetric_difference(set_b)
```

## 資料匯出

### 寫入 BED 檔案

將區域匯出為 BED 格式：

```python
# 寫入 BED 檔案
regions.to_bed("output.bed")

# 帶分數寫入
regions.to_bed("output.bed", scores=score_array)

# 帶名稱寫入
regions.to_bed("output.bed", names=name_list)
```

### 格式轉換

在格式之間轉換：

```python
# BED 轉 JSON
regions = gtars.RegionSet.from_bed("input.bed")
regions.to_json("output.json")

# JSON 轉 BED
regions = gtars.RegionSet.from_json("input.json")
regions.to_bed("output.bed")
```

## NumPy 整合

與 NumPy 陣列無縫整合：

```python
import numpy as np

# 匯出為 NumPy 陣列
starts = regions.starts_array()  # 起始位置的 NumPy 陣列
ends = regions.ends_array()      # 結束位置的 NumPy 陣列
sizes = regions.sizes_array()    # 區域大小的 NumPy 陣列

# 從 NumPy 陣列建立
chromosomes = ["chr1"] * len(starts)
regions = gtars.RegionSet.from_arrays(chromosomes, starts, ends)
```

## 平行處理

利用平行處理處理大型資料集：

```python
# 啟用平行處理
regions = gtars.RegionSet.from_bed("large_file.bed", parallel=True)

# 平行操作
result = regions.parallel_apply(custom_function)
```

## 記憶體管理

大型資料集的高效記憶體使用：

```python
# 串流處理大型 BED 檔案
for chunk in gtars.RegionSet.stream_bed("large_file.bed", chunk_size=10000):
    process_chunk(chunk)

# 記憶體對映模式
regions = gtars.RegionSet.from_bed("large_file.bed", mmap=True)
```

## 錯誤處理

處理常見錯誤：

```python
try:
    regions = gtars.RegionSet.from_bed("file.bed")
except gtars.FileNotFoundError:
    print("找不到檔案")
except gtars.InvalidFormatError as e:
    print(f"無效的 BED 格式：{e}")
except gtars.ParseError as e:
    print(f"第 {e.line} 行解析錯誤：{e.message}")
```

## 配置

配置 gtars 行為：

```python
# 設定全域選項
gtars.set_option("parallel.threads", 4)
gtars.set_option("memory.limit", "4GB")
gtars.set_option("warnings.strict", True)

# 暫時選項的上下文管理器
with gtars.option_context("parallel.threads", 8):
    # 此區塊使用 8 個執行緒
    regions = gtars.RegionSet.from_bed("large_file.bed", parallel=True)
```

## 日誌

啟用日誌進行除錯：

```python
import logging

# 啟用 gtars 日誌
gtars.set_log_level("DEBUG")

# 或使用 Python logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("gtars")
```

## 效能技巧

- 對大型資料集使用平行處理
- 對非常大的檔案啟用記憶體對映模式
- 盡可能使用串流處理以減少記憶體使用
- 在適用時預先排序區域再進行操作
- 使用 NumPy 陣列進行數值計算
- 快取經常存取的資料
