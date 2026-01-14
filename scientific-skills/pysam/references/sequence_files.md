# 處理序列檔案（FASTA/FASTQ）

## FASTA 檔案

### 概述

Pysam 提供 `FastaFile` 類別用於索引化、隨機存取 FASTA 參考序列。使用前必須先用 `samtools faidx` 索引 FASTA 檔案。

### 開啟 FASTA 檔案

```python
import pysam

# 開啟已索引的 FASTA 檔案
fasta = pysam.FastaFile("reference.fasta")

# 自動尋找 reference.fasta.fai 索引
```

### 建立 FASTA 索引

```python
# 使用 pysam 建立索引
pysam.faidx("reference.fasta")

# 或使用 samtools 指令
pysam.samtools.faidx("reference.fasta")
```

這會建立隨機存取所需的 `.fai` 索引檔案。

### FastaFile 屬性

```python
fasta = pysam.FastaFile("reference.fasta")

# 參考序列列表
references = fasta.references
print(f"References: {references}")

# 取得長度
lengths = fasta.lengths
print(f"Lengths: {lengths}")

# 取得特定序列長度
chr1_length = fasta.get_reference_length("chr1")
```

### 取得序列

#### 依區域取得

使用 **0-based、半開區間** 座標：

```python
# 取得特定區域
sequence = fasta.fetch("chr1", 1000, 2000)
print(f"Sequence: {sequence}")  # 回傳 1000 個鹼基

# 取得整個染色體
chr1_seq = fasta.fetch("chr1")

# 使用區域字串取得（1-based）
sequence = fasta.fetch(region="chr1:1001-2000")
```

**重要：** 數值引數使用 0-based 座標，區域字串使用 1-based 座標（samtools 慣例）。

#### 常見用例

```python
# 取得變異位置的序列上下文
def get_variant_context(fasta, chrom, pos, window=10):
    """取得變異位置周圍的序列上下文（1-based）。"""
    start = max(0, pos - window - 1)  # 轉換為 0-based
    end = pos + window
    return fasta.fetch(chrom, start, end)

# 取得基因座標的序列
def get_gene_sequence(fasta, chrom, start, end, strand):
    """取得具有股向意識的基因序列。"""
    seq = fasta.fetch(chrom, start, end)

    if strand == "-":
        # 反向互補
        complement = str.maketrans("ATGCatgc", "TACGtacg")
        seq = seq.translate(complement)[::-1]

    return seq

# 檢查參考等位基因
def check_ref_allele(fasta, chrom, pos, expected_ref):
    """驗證位置的參考等位基因（1-based pos）。"""
    actual = fasta.fetch(chrom, pos-1, pos)  # 轉換為 0-based
    return actual.upper() == expected_ref.upper()
```

### 提取多個區域

```python
# 高效提取多個區域
regions = [
    ("chr1", 1000, 2000),
    ("chr1", 5000, 6000),
    ("chr2", 10000, 11000)
]

sequences = {}
for chrom, start, end in regions:
    seq_id = f"{chrom}:{start}-{end}"
    sequences[seq_id] = fasta.fetch(chrom, start, end)
```

### 處理模糊鹼基

FASTA 檔案可能包含 IUPAC 模糊碼：

- N = 任何鹼基
- R = A 或 G（嘌呤）
- Y = C 或 T（嘧啶）
- S = G 或 C（強）
- W = A 或 T（弱）
- K = G 或 T（酮）
- M = A 或 C（氨基）
- B = C、G 或 T（非 A）
- D = A、G 或 T（非 C）
- H = A、C 或 T（非 G）
- V = A、C 或 G（非 T）

```python
# 處理模糊鹼基
def count_ambiguous(sequence):
    """計算非 ATGC 鹼基。"""
    return sum(1 for base in sequence.upper() if base not in "ATGC")

# 移除有太多 N 的區域
def has_quality_sequence(fasta, chrom, start, end, max_n_frac=0.1):
    """檢查區域是否有可接受的 N 含量。"""
    seq = fasta.fetch(chrom, start, end)
    n_count = seq.upper().count('N')
    return (n_count / len(seq)) <= max_n_frac
```

## FASTQ 檔案

### 概述

Pysam 提供 `FastxFile`（或 `FastqFile`）用於讀取包含帶品質分數的原始定序讀取的 FASTQ 檔案。FASTQ 檔案不支援隨機存取—只能循序讀取。

### 開啟 FASTQ 檔案

```python
import pysam

# 開啟 FASTQ 檔案
fastq = pysam.FastxFile("reads.fastq")

# 支援壓縮檔案
fastq_gz = pysam.FastxFile("reads.fastq.gz")
```

### 讀取 FASTQ 記錄

```python
fastq = pysam.FastxFile("reads.fastq")

for read in fastq:
    print(f"Name: {read.name}")
    print(f"Sequence: {read.sequence}")
    print(f"Quality: {read.quality}")
    print(f"Comment: {read.comment}")  # 可選的標頭註解
```

**FastqProxy 屬性：**
- `name` - 讀取識別碼（不含 @ 前綴）
- `sequence` - DNA/RNA 序列
- `quality` - ASCII 編碼的品質字串
- `comment` - 標頭行的可選註解
- `get_quality_array()` - 將品質字串轉換為數值陣列

### 品質分數轉換

```python
# 將品質字串轉換為數值
for read in fastq:
    qual_array = read.get_quality_array()
    mean_quality = sum(qual_array) / len(qual_array)
    print(f"{read.name}: mean Q = {mean_quality:.1f}")
```

品質分數是 Phred 尺度（通常是 Phred+33 編碼）：
- Q = -10 * log10(P_error)
- ASCII 33（'!'）= Q0
- ASCII 43（'+'）= Q10
- ASCII 63（'?'）= Q30

### 常見 FASTQ 處理工作流程

#### 品質過濾

```python
def filter_by_quality(input_fastq, output_fastq, min_mean_quality=20):
    """依平均品質分數過濾讀取。"""
    with pysam.FastxFile(input_fastq) as infile:
        with open(output_fastq, 'w') as outfile:
            for read in infile:
                qual_array = read.get_quality_array()
                mean_q = sum(qual_array) / len(qual_array)

                if mean_q >= min_mean_quality:
                    # 以 FASTQ 格式寫入
                    outfile.write(f"@{read.name}\n")
                    outfile.write(f"{read.sequence}\n")
                    outfile.write("+\n")
                    outfile.write(f"{read.quality}\n")
```

#### 長度過濾

```python
def filter_by_length(input_fastq, output_fastq, min_length=50):
    """依最小長度過濾讀取。"""
    with pysam.FastxFile(input_fastq) as infile:
        with open(output_fastq, 'w') as outfile:
            kept = 0
            for read in infile:
                if len(read.sequence) >= min_length:
                    outfile.write(f"@{read.name}\n")
                    outfile.write(f"{read.sequence}\n")
                    outfile.write("+\n")
                    outfile.write(f"{read.quality}\n")
                    kept += 1
    print(f"Kept {kept} reads")
```

#### 計算品質統計

```python
def calculate_fastq_stats(fastq_file):
    """計算 FASTQ 檔案的基本統計。"""
    total_reads = 0
    total_bases = 0
    quality_sum = 0

    with pysam.FastxFile(fastq_file) as fastq:
        for read in fastq:
            total_reads += 1
            read_length = len(read.sequence)
            total_bases += read_length

            qual_array = read.get_quality_array()
            quality_sum += sum(qual_array)

    return {
        "total_reads": total_reads,
        "total_bases": total_bases,
        "mean_read_length": total_bases / total_reads if total_reads > 0 else 0,
        "mean_quality": quality_sum / total_bases if total_bases > 0 else 0
    }
```

#### 依名稱提取讀取

```python
def extract_reads_by_name(fastq_file, read_names, output_file):
    """依名稱提取特定讀取。"""
    read_set = set(read_names)

    with pysam.FastxFile(fastq_file) as infile:
        with open(output_file, 'w') as outfile:
            for read in infile:
                if read.name in read_set:
                    outfile.write(f"@{read.name}\n")
                    outfile.write(f"{read.sequence}\n")
                    outfile.write("+\n")
                    outfile.write(f"{read.quality}\n")
```

#### FASTQ 轉 FASTA

```python
def fastq_to_fasta(fastq_file, fasta_file):
    """將 FASTQ 轉換為 FASTA（丟棄品質分數）。"""
    with pysam.FastxFile(fastq_file) as infile:
        with open(fasta_file, 'w') as outfile:
            for read in infile:
                outfile.write(f">{read.name}\n")
                outfile.write(f"{read.sequence}\n")
```

#### 子抽樣 FASTQ

```python
import random

def subsample_fastq(input_fastq, output_fastq, fraction=0.1, seed=42):
    """從 FASTQ 檔案隨機子抽樣讀取。"""
    random.seed(seed)

    with pysam.FastxFile(input_fastq) as infile:
        with open(output_fastq, 'w') as outfile:
            for read in infile:
                if random.random() < fraction:
                    outfile.write(f"@{read.name}\n")
                    outfile.write(f"{read.sequence}\n")
                    outfile.write("+\n")
                    outfile.write(f"{read.quality}\n")
```

## Tabix 索引檔案

### 概述

Pysam 提供 `TabixFile` 用於存取 tabix 索引的基因體資料檔案（BED、GFF、GTF、一般製表符分隔）。

### 開啟 Tabix 檔案

```python
import pysam

# 開啟 tabix 索引檔案
tabix = pysam.TabixFile("annotations.bed.gz")

# 檔案必須是 bgzip 壓縮並 tabix 索引
```

### 建立 Tabix 索引

```python
# 索引檔案
pysam.tabix_index("annotations.bed", preset="bed", force=True)
# 建立 annotations.bed.gz 和 annotations.bed.gz.tbi

# 可用預設：bed、gff、vcf
```

### 取得記錄

```python
tabix = pysam.TabixFile("annotations.bed.gz")

# 取得區域
for row in tabix.fetch("chr1", 1000000, 2000000):
    print(row)  # 回傳製表符分隔字串

# 使用特定解析器解析
for row in tabix.fetch("chr1", 1000000, 2000000, parser=pysam.asBed()):
    print(f"Interval: {row.contig}:{row.start}-{row.end}")

# 可用解析器：asBed()、asGTF()、asVCF()、asTuple()
```

### 處理 BED 檔案

```python
bed = pysam.TabixFile("regions.bed.gz")

# 依名稱存取 BED 欄位
for interval in bed.fetch("chr1", 1000000, 2000000, parser=pysam.asBed()):
    print(f"Region: {interval.contig}:{interval.start}-{interval.end}")
    print(f"Name: {interval.name}")
    print(f"Score: {interval.score}")
    print(f"Strand: {interval.strand}")
```

### 處理 GTF/GFF 檔案

```python
gtf = pysam.TabixFile("annotations.gtf.gz")

# 存取 GTF 欄位
for feature in gtf.fetch("chr1", 1000000, 2000000, parser=pysam.asGTF()):
    print(f"Feature: {feature.feature}")
    print(f"Gene: {feature.gene_id}")
    print(f"Transcript: {feature.transcript_id}")
    print(f"Coordinates: {feature.start}-{feature.end}")
```

## 效能提示

### FASTA
1. **總是使用已索引的 FASTA** 檔案（使用 samtools faidx 建立 .fai）
2. **批次取得操作** 當提取多個區域時
3. **快取經常存取的序列** 在記憶體中
4. **使用適當的視窗大小** 避免載入過多序列資料

### FASTQ
1. **串流處理** - FASTQ 檔案循序讀取，即時處理
2. **使用壓縮的 FASTQ.gz** 節省磁碟空間（pysam 透明處理）
3. **避免將整個檔案載入** 記憶體—逐讀取處理
4. **對於大檔案**，考慮使用檔案分割進行平行處理

### Tabix
1. **總是 bgzip 並 tabix 索引** 檔案後再進行區域查詢
2. **建立索引時使用適當的預設**
3. **指定解析器** 以進行具名欄位存取
4. **批次查詢** 同一檔案以避免重新開啟

## 常見陷阱

1. **FASTA 座標系統：** fetch() 使用 0-based 座標，區域字串使用 1-based
2. **缺少索引：** FASTA 隨機存取需要 .fai 索引檔案
3. **FASTQ 僅循序：** 無法對 FASTQ 進行隨機存取或基於區域的查詢
4. **品質編碼：** 除非另有說明，假設 Phred+33
5. **Tabix 壓縮：** 必須使用 bgzip，而非一般 gzip，才能進行 tabix 索引
6. **解析器需求：** TabixFile 需要明確的解析器才能進行具名欄位存取
7. **大小寫敏感：** FASTA 序列保留大小寫—使用 .upper() 或 .lower() 進行一致性比較
