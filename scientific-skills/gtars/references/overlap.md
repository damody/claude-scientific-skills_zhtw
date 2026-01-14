# 重疊偵測與 IGD

overlaprs 模組使用整合基因體資料庫（Integrated Genome Database，IGD）資料結構提供基因體區間之間的高效重疊偵測。

## IGD 索引

IGD（整合基因體資料庫）是用於快速基因體區間查詢和重疊偵測的專門資料結構。

### 建立 IGD 索引

從基因體區域檔案建立索引：

```python
import gtars

# 從 BED 檔案建立 IGD 索引
igd = gtars.igd.build_index("regions.bed")

# 儲存索引以供重複使用
igd.save("regions.igd")

# 載入現有索引
igd = gtars.igd.load_index("regions.igd")
```

### 查詢重疊

高效尋找重疊區域：

```python
# 查詢單一區域
overlaps = igd.query("chr1", 1000, 2000)

# 查詢多個區域
results = []
for chrom, start, end in query_regions:
    overlaps = igd.query(chrom, start, end)
    results.append(overlaps)

# 僅取得重疊計數
count = igd.count_overlaps("chr1", 1000, 2000)
```

## CLI 使用

overlaprs 命令列工具提供重疊偵測：

```bash
# 尋找兩個 BED 檔案之間的重疊
gtars overlaprs query --index regions.bed --query query_regions.bed

# 計數重疊
gtars overlaprs count --index regions.bed --query query_regions.bed

# 輸出重疊區域
gtars overlaprs overlap --index regions.bed --query query_regions.bed --output overlaps.bed
```

### IGD CLI 命令

建立和查詢 IGD 索引：

```bash
# 建立 IGD 索引
gtars igd build --input regions.bed --output regions.igd

# 查詢 IGD 索引
gtars igd query --index regions.igd --region "chr1:1000-2000"

# 從檔案批次查詢
gtars igd query --index regions.igd --query-file queries.bed --output results.bed
```

## Python API

### 重疊偵測

計算區域集之間的重疊：

```python
import gtars

# 載入兩個區域集
set_a = gtars.RegionSet.from_bed("regions_a.bed")
set_b = gtars.RegionSet.from_bed("regions_b.bed")

# 尋找重疊
overlaps = set_a.overlap(set_b)

# 取得 A 中與 B 重疊的區域
overlapping_a = set_a.filter_overlapping(set_b)

# 取得 A 中不與 B 重疊的區域
non_overlapping_a = set_a.filter_non_overlapping(set_b)
```

### 重疊統計

計算重疊指標：

```python
# 計數重疊
overlap_count = set_a.count_overlaps(set_b)

# 計算重疊比例
overlap_fraction = set_a.overlap_fraction(set_b)

# 取得重疊涵蓋度
coverage = set_a.overlap_coverage(set_b)
```

## 效能特性

IGD 提供高效查詢：
- **索引建構**：O(n log n)，其中 n 是區域數量
- **查詢時間**：O(k + log n)，其中 k 是重疊數量
- **記憶體高效**：基因體區間的緊湊表示

## 使用案例

### 調控元件分析

識別基因體特徵之間的重疊：

```python
# 尋找與啟動子重疊的轉錄因子結合位點
tfbs = gtars.RegionSet.from_bed("chip_seq_peaks.bed")
promoters = gtars.RegionSet.from_bed("promoters.bed")

overlapping_tfbs = tfbs.filter_overlapping(promoters)
print(f"在啟動子中找到 {len(overlapping_tfbs)} 個 TFBS")
```

### 變異註釋

使用重疊特徵註釋變異：

```python
# 檢查哪些變異與編碼區域重疊
variants = gtars.RegionSet.from_bed("variants.bed")
cds = gtars.RegionSet.from_bed("coding_sequences.bed")

coding_variants = variants.filter_overlapping(cds)
```

### 染色質狀態分析

比較樣本之間的染色質狀態：

```python
# 尋找具有一致染色質狀態的區域
sample1 = gtars.RegionSet.from_bed("sample1_peaks.bed")
sample2 = gtars.RegionSet.from_bed("sample2_peaks.bed")

consistent_regions = sample1.overlap(sample2)
```
