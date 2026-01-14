# 參考序列管理

refget 模組處理參考序列擷取和摘要計算，遵循用於序列識別的 refget 協定。

## RefgetStore

RefgetStore 管理參考序列及其摘要：

```python
import gtars

# 建立 RefgetStore
store = gtars.RefgetStore()

# 新增序列
store.add_sequence("chr1", sequence_data)

# 擷取序列
seq = store.get_sequence("chr1")

# 取得序列摘要
digest = store.get_digest("chr1")
```

## 序列摘要

計算和驗證序列摘要：

```python
# 計算序列的摘要
from gtars.refget import compute_digest

digest = compute_digest(sequence_data)

# 驗證摘要是否匹配
is_valid = store.verify_digest("chr1", expected_digest)
```

## 與參考基因體整合

使用標準參考基因體：

```python
# 載入參考基因體
store = gtars.RefgetStore.from_fasta("hg38.fa")

# 取得染色體序列
chr1 = store.get_sequence("chr1")
chr2 = store.get_sequence("chr2")

# 取得子序列
region_seq = store.get_subsequence("chr1", 1000, 2000)
```

## CLI 使用

從命令列管理參考序列：

```bash
# 計算 FASTA 檔案的摘要
gtars refget digest --input genome.fa --output digests.txt

# 驗證序列摘要
gtars refget verify --sequence sequence.fa --digest expected_digest
```

## Refget 協定相容性

refget 模組遵循 GA4GH refget 協定：

### 摘要計算

摘要使用 SHA-512 計算並截斷為 48 位元組：

```python
# 計算符合 refget 的摘要
digest = gtars.refget.compute_digest(sequence)
# 回傳："SQ.abc123..."
```

### 序列擷取

依摘要擷取序列：

```python
# 依 refget 摘要取得序列
seq = store.get_sequence_by_digest("SQ.abc123...")
```

## 使用案例

### 參考驗證

驗證參考基因體完整性：

```python
# 計算參考的摘要
store = gtars.RefgetStore.from_fasta("reference.fa")
digests = {chrom: store.get_digest(chrom) for chrom in store.chromosomes}

# 與預期摘要比較
for chrom, expected in expected_digests.items():
    actual = digests[chrom]
    if actual != expected:
        print(f"{chrom} 不匹配：{actual} != {expected}")
```

### 序列提取

提取特定基因體區域：

```python
# 提取感興趣的區域
store = gtars.RefgetStore.from_fasta("hg38.fa")

regions = [
    ("chr1", 1000, 2000),
    ("chr2", 5000, 6000),
    ("chr3", 10000, 11000)
]

sequences = [store.get_subsequence(c, s, e) for c, s, e in regions]
```

### 交叉參考比較

比較不同參考版本之間的序列：

```python
# 載入兩個參考版本
hg19 = gtars.RefgetStore.from_fasta("hg19.fa")
hg38 = gtars.RefgetStore.from_fasta("hg38.fa")

# 比較摘要
for chrom in hg19.chromosomes:
    digest_19 = hg19.get_digest(chrom)
    digest_38 = hg38.get_digest(chrom)
    if digest_19 != digest_38:
        print(f"{chrom} 在 hg19 和 hg38 之間不同")
```

## 效能說明

- 序列按需載入
- 摘要在計算後快取
- 高效的子序列提取
- 支援大型基因體的記憶體對映檔案
