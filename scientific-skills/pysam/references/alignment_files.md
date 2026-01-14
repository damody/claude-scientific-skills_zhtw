# 處理比對檔案（SAM/BAM/CRAM）

## 概述

Pysam 提供 `AlignmentFile` 類別用於讀取和寫入包含比對序列資料的 SAM/BAM/CRAM 格式檔案。BAM/CRAM 檔案透過索引支援壓縮和隨機存取。

## 開啟比對檔案

透過模式限定符指定格式：
- `"rb"` - 讀取 BAM（二進位）
- `"r"` - 讀取 SAM（文字）
- `"rc"` - 讀取 CRAM（壓縮）
- `"wb"` - 寫入 BAM
- `"w"` - 寫入 SAM
- `"wc"` - 寫入 CRAM

```python
import pysam

# 讀取
samfile = pysam.AlignmentFile("example.bam", "rb")

# 寫入（需要模板或標頭）
outfile = pysam.AlignmentFile("output.bam", "wb", template=samfile)
```

### 串流處理

使用 `"-"` 作為檔名進行 stdin/stdout 操作：

```python
# 從 stdin 讀取
infile = pysam.AlignmentFile('-', 'rb')

# 寫入 stdout
outfile = pysam.AlignmentFile('-', 'w', template=infile)
```

**重要：** Pysam 不支援從真正的 Python 檔案物件讀取/寫入—僅支援 stdin/stdout 串流。

## AlignmentFile 屬性

**標頭資訊：**
- `references` - 染色體/contig 名稱列表
- `lengths` - 每個參考序列的對應長度
- `header` - 完整標頭作為字典

```python
samfile = pysam.AlignmentFile("example.bam", "rb")
print(f"References: {samfile.references}")
print(f"Lengths: {samfile.lengths}")
```

## 讀取讀取（Reads）

### fetch() - 基於區域的擷取

使用 **0-based 座標** 擷取與指定基因體區域重疊的讀取。

```python
# 取得特定區域
for read in samfile.fetch("chr1", 1000, 2000):
    print(read.query_name, read.reference_start)

# 取得整個 contig
for read in samfile.fetch("chr1"):
    print(read.query_name)

# 無索引取得（循序讀取）
for read in samfile.fetch(until_eof=True):
    print(read.query_name)
```

**重要注意事項：**
- 隨機存取需要索引（.bai/.crai）
- 回傳**重疊**區域的讀取（可能延伸超出邊界）
- 對非索引檔案或循序讀取使用 `until_eof=True`
- 預設只回傳已比對的讀取
- 對於未比對的讀取，使用 `fetch("*")` 或 `until_eof=True`

### 多個迭代器

在同一檔案上使用多個迭代器時：

```python
samfile = pysam.AlignmentFile("example.bam", "rb", multiple_iterators=True)
iter1 = samfile.fetch("chr1", 1000, 2000)
iter2 = samfile.fetch("chr2", 5000, 6000)
```

沒有 `multiple_iterators=True`，新的 fetch() 呼叫會重新定位檔案指標並破壞現有迭代器。

### count() - 計算區域內的讀取數

```python
# 計算所有讀取
num_reads = samfile.count("chr1", 1000, 2000)

# 帶品質過濾的計數
num_quality_reads = samfile.count("chr1", 1000, 2000, quality=20)
```

### count_coverage() - 逐鹼基覆蓋度

回傳四個陣列（A、C、G、T）包含逐鹼基覆蓋度：

```python
coverage = samfile.count_coverage("chr1", 1000, 2000)
a_counts, c_counts, g_counts, t_counts = coverage
```

## AlignedSegment 物件

每個讀取表示為具有以下關鍵屬性的 `AlignedSegment` 物件：

### 讀取資訊
- `query_name` - 讀取名稱/ID
- `query_sequence` - 讀取序列（鹼基）
- `query_qualities` - 鹼基品質分數（ASCII 編碼）
- `query_length` - 讀取的長度

### 比對資訊
- `reference_name` - 染色體/contig 名稱
- `reference_start` - 起始位置（0-based，包含）
- `reference_end` - 結束位置（0-based，排除）
- `mapping_quality` - MAPQ 分數
- `cigarstring` - CIGAR 字串（例如 "100M"）
- `cigartuples` - CIGAR 作為（操作，長度）元組列表

**重要：** `cigartuples` 格式與 SAM 規範不同。操作是整數：
- 0 = M（比對/錯配）
- 1 = I（插入）
- 2 = D（刪除）
- 3 = N（跳過參考）
- 4 = S（軟裁剪）
- 5 = H（硬裁剪）
- 6 = P（填充）
- 7 = =（序列比對）
- 8 = X（序列錯配）

### 旗標和狀態
- `flag` - SAM 旗標作為整數
- `is_paired` - 讀取是否配對？
- `is_proper_pair` - 讀取是否在正確配對中？
- `is_unmapped` - 讀取是否未比對？
- `mate_is_unmapped` - 配對是否未比對？
- `is_reverse` - 讀取是否在反向股？
- `mate_is_reverse` - 配對是否在反向股？
- `is_read1` - 這是 read1 嗎？
- `is_read2` - 這是 read2 嗎？
- `is_secondary` - 是次要比對嗎？
- `is_qcfail` - 讀取是否未通過 QC？
- `is_duplicate` - 讀取是否為重複？
- `is_supplementary` - 是補充比對嗎？

### 標籤和可選欄位
- `get_tag(tag)` - 取得可選欄位的值
- `set_tag(tag, value)` - 設定可選欄位
- `has_tag(tag)` - 檢查標籤是否存在
- `get_tags()` - 取得所有標籤作為元組列表

```python
for read in samfile.fetch("chr1", 1000, 2000):
    if read.has_tag("NM"):
        edit_distance = read.get_tag("NM")
        print(f"{read.query_name}: NM={edit_distance}")
```

## 寫入比對檔案

### 建立標頭

```python
header = {
    'HD': {'VN': '1.0'},
    'SQ': [
        {'LN': 1575, 'SN': 'chr1'},
        {'LN': 1584, 'SN': 'chr2'}
    ]
}

outfile = pysam.AlignmentFile("output.bam", "wb", header=header)
```

### 建立 AlignedSegment 物件

```python
# 建立新讀取
a = pysam.AlignedSegment()
a.query_name = "read001"
a.query_sequence = "AGCTTAGCTAGCTACCTATATCTTGGTCTTGGCCG"
a.flag = 0
a.reference_id = 0  # header['SQ'] 的索引
a.reference_start = 100
a.mapping_quality = 20
a.cigar = [(0, 35)]  # 35M
a.query_qualities = pysam.qualitystring_to_array("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

# 寫入檔案
outfile.write(a)
```

### 格式之間的轉換

```python
# BAM 轉 SAM
infile = pysam.AlignmentFile("input.bam", "rb")
outfile = pysam.AlignmentFile("output.sam", "w", template=infile)
for read in infile:
    outfile.write(read)
infile.close()
outfile.close()
```

## 堆疊分析

`pileup()` 方法提供跨區域的**列式**（逐位置）分析：

```python
for pileupcolumn in samfile.pileup("chr1", 1000, 2000):
    print(f"Position {pileupcolumn.pos}: coverage = {pileupcolumn.nsegments}")

    for pileupread in pileupcolumn.pileups:
        if not pileupread.is_del and not pileupread.is_refskip:
            # 查詢位置是讀取中的位置
            base = pileupread.alignment.query_sequence[pileupread.query_position]
            print(f"  {pileupread.alignment.query_name}: {base}")
```

**關鍵屬性：**
- `pileupcolumn.pos` - 0-based 參考位置
- `pileupcolumn.nsegments` - 覆蓋該位置的讀取數
- `pileupread.alignment` - AlignedSegment 物件
- `pileupread.query_position` - 讀取中的位置（刪除時為 None）
- `pileupread.is_del` - 這是刪除嗎？
- `pileupread.is_refskip` - 這是參考跳過（CIGAR 中的 N）嗎？

**重要：** 保持迭代器參考活躍。當迭代器過早離開作用域時會發生「PileupProxy accessed after iterator finished」錯誤。

## 座標系統

**關鍵：** Pysam 使用 **0-based、半開區間** 座標（Python 慣例）：
- `reference_start` 是 0-based（第一個鹼基是 0）
- `reference_end` 是排除的（不包含在範圍內）
- 區域 1000-2000 包含鹼基 1000-1999

**例外：** `fetch()` 和 `pileup()` 中的區域字串遵循 samtools 慣例（1-based）：
```python
# 這些是等價的：
samfile.fetch("chr1", 999, 2000)  # Python 風格：0-based
samfile.fetch("chr1:1000-2000")   # samtools 風格：1-based
```

## 索引

建立 BAM 索引：
```python
pysam.index("example.bam")
```

或使用命令列介面：
```python
pysam.samtools.index("example.bam")
```

## 效能提示

1. **使用索引存取** 當重複查詢特定區域時
2. **使用 `pileup()` 進行列式分析** 而非重複的 fetch 操作
3. **使用 `fetch(until_eof=True)` 循序讀取** 非索引檔案
4. **避免多個迭代器** 除非必要（有效能成本）
5. **使用 `count()` 進行簡單計數** 而非迭代並手動計數

## 常見陷阱

1. **部分重疊：** `fetch()` 回傳與區域邊界重疊的讀取—如需精確邊界則實作明確過濾
2. **品質分數編輯：** 修改 `query_sequence` 後無法就地編輯 `query_qualities`。先建立副本：`quals = read.query_qualities`
3. **缺少索引：** 沒有 `until_eof=True` 的 `fetch()` 需要索引檔案
4. **執行緒安全：** 雖然 pysam 在 I/O 期間釋放 GIL，但全面的執行緒安全性尚未完全驗證
5. **迭代器作用域：** 保持堆疊迭代器參考活躍以避免「PileupProxy accessed after iterator finished」錯誤
