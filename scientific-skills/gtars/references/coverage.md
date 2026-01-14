# 使用 Uniwig 進行涵蓋度分析

uniwig 模組從定序資料產生涵蓋度軌跡，提供將基因體區間高效轉換為涵蓋度圖譜的功能。

## 涵蓋度軌跡產生

從 BED 檔案建立涵蓋度軌跡：

```python
import gtars

# 從 BED 檔案產生涵蓋度
coverage = gtars.uniwig.coverage_from_bed("fragments.bed")

# 使用特定解析度產生涵蓋度
coverage = gtars.uniwig.coverage_from_bed("fragments.bed", resolution=10)

# 產生鏈特異性涵蓋度
fwd_coverage = gtars.uniwig.coverage_from_bed("fragments.bed", strand="+")
rev_coverage = gtars.uniwig.coverage_from_bed("fragments.bed", strand="-")
```

## CLI 使用

從命令列產生涵蓋度軌跡：

```bash
# 產生涵蓋度軌跡
gtars uniwig generate --input fragments.bed --output coverage.wig

# 指定解析度
gtars uniwig generate --input fragments.bed --output coverage.wig --resolution 10

# 產生 BigWig 格式
gtars uniwig generate --input fragments.bed --output coverage.bw --format bigwig

# 鏈特異性涵蓋度
gtars uniwig generate --input fragments.bed --output forward.wig --strand +
gtars uniwig generate --input fragments.bed --output reverse.wig --strand -
```

## 使用涵蓋度資料

### 存取涵蓋度值

查詢特定位置的涵蓋度：

```python
# 取得位置的涵蓋度
cov = coverage.get_coverage("chr1", 1000)

# 取得範圍的涵蓋度
cov_array = coverage.get_coverage_range("chr1", 1000, 2000)

# 取得涵蓋度統計
mean_cov = coverage.mean_coverage("chr1", 1000, 2000)
max_cov = coverage.max_coverage("chr1", 1000, 2000)
```

### 涵蓋度操作

對涵蓋度軌跡執行操作：

```python
# 正規化涵蓋度
normalized = coverage.normalize()

# 平滑涵蓋度
smoothed = coverage.smooth(window_size=10)

# 合併涵蓋度軌跡
combined = coverage1.add(coverage2)

# 計算涵蓋度差異
diff = coverage1.subtract(coverage2)
```

## 輸出格式

Uniwig 支援多種輸出格式：

### WIG 格式

標準 wiggle 格式：
```
fixedStep chrom=chr1 start=1000 step=1
12
15
18
22
...
```

### BigWig 格式

用於高效儲存和存取的二進位格式：
```bash
# 產生 BigWig
gtars uniwig generate --input fragments.bed --output coverage.bw --format bigwig
```

### BedGraph 格式

可變涵蓋度的彈性格式：
```
chr1    1000    1001    12
chr1    1001    1002    15
chr1    1002    1003    18
```

## 使用案例

### ATAC-seq 分析

產生染色質可及性圖譜：

```python
# 產生 ATAC-seq 涵蓋度
atac_fragments = gtars.RegionSet.from_bed("atac_fragments.bed")
coverage = gtars.uniwig.coverage_from_bed("atac_fragments.bed", resolution=1)

# 識別可及區域
peaks = coverage.call_peaks(threshold=10)
```

### ChIP-seq 峰視覺化

為 ChIP-seq 資料建立涵蓋度軌跡：

```bash
# 產生用於視覺化的涵蓋度
gtars uniwig generate --input chip_seq_fragments.bed \
                      --output chip_coverage.bw \
                      --format bigwig
```

### RNA-seq 涵蓋度

計算 RNA-seq 的讀取涵蓋度：

```python
# 產生鏈特異性 RNA-seq 涵蓋度
fwd = gtars.uniwig.coverage_from_bed("rnaseq.bed", strand="+")
rev = gtars.uniwig.coverage_from_bed("rnaseq.bed", strand="-")

# 匯出供 IGV 使用
fwd.to_bigwig("rnaseq_fwd.bw")
rev.to_bigwig("rnaseq_rev.bw")
```

### 差異涵蓋度分析

比較樣本之間的涵蓋度：

```python
# 為兩個樣本產生涵蓋度
control = gtars.uniwig.coverage_from_bed("control.bed")
treatment = gtars.uniwig.coverage_from_bed("treatment.bed")

# 計算倍數變化
fold_change = treatment.divide(control)

# 尋找差異區域
diff_regions = fold_change.find_regions(threshold=2.0)
```

## 效能最佳化

- 根據資料規模使用適當的解析度
- 建議大型資料集使用 BigWig 格式
- 多染色體可使用平行處理
- 大型檔案可使用記憶體高效的串流處理
