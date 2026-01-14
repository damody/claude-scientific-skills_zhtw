# 處理變異檔案（VCF/BCF）

## 概述

Pysam 提供 `VariantFile` 類別用於讀取和寫入 VCF（Variant Call Format，變異呼叫格式）和 BCF（binary VCF，二進位 VCF）檔案。這些檔案包含遺傳變異的資訊，包括 SNP、indel 和結構變異。

## 開啟變異檔案

```python
import pysam

# 讀取 VCF
vcf = pysam.VariantFile("example.vcf")

# 讀取 BCF（二進位，壓縮）
bcf = pysam.VariantFile("example.bcf")

# 讀取壓縮的 VCF
vcf_gz = pysam.VariantFile("example.vcf.gz")

# 寫入
outvcf = pysam.VariantFile("output.vcf", "w", header=vcf.header)
```

## VariantFile 屬性

**標頭資訊：**
- `header` - 包含元資料的完整 VCF 標頭
- `header.contigs` - contig/染色體字典
- `header.samples` - 樣本名稱列表
- `header.filters` - FILTER 定義字典
- `header.info` - INFO 欄位定義字典
- `header.formats` - FORMAT 欄位定義字典

```python
vcf = pysam.VariantFile("example.vcf")

# 列出樣本
print(f"Samples: {list(vcf.header.samples)}")

# 列出 contig
for contig in vcf.header.contigs:
    print(f"{contig}: length={vcf.header.contigs[contig].length}")

# 列出 INFO 欄位
for info in vcf.header.info:
    print(f"{info}: {vcf.header.info[info].description}")
```

## 讀取變異記錄

### 迭代所有變異

```python
for variant in vcf:
    print(f"{variant.chrom}:{variant.pos} {variant.ref}>{variant.alts}")
```

### 取得特定區域

VCF.gz 需要 tabix 索引（.tbi）或 BCF 需要索引：

```python
# 取得區域內的變異（區域字串使用 1-based 座標）
for variant in vcf.fetch("chr1", 1000000, 2000000):
    print(f"{variant.chrom}:{variant.pos} {variant.id}")

# 使用區域字串（1-based）
for variant in vcf.fetch("chr1:1000000-2000000"):
    print(variant.pos)
```

**注意：** `fetch()` 呼叫使用 **1-based 座標** 以匹配 VCF 規範。

## VariantRecord 物件

每個變異表示為 `VariantRecord` 物件：

### 位置資訊
- `chrom` - 染色體/contig 名稱
- `pos` - 位置（1-based）
- `start` - 起始位置（0-based）
- `stop` - 結束位置（0-based，排除）
- `id` - 變異 ID（例如 rsID）

### 等位基因資訊
- `ref` - 參考等位基因
- `alts` - 替代等位基因元組
- `alleles` - 所有等位基因元組（ref + alts）

### 品質和過濾
- `qual` - 品質分數（QUAL 欄位）
- `filter` - 過濾狀態

### INFO 欄位

以字典方式存取 INFO 欄位：

```python
for variant in vcf:
    # 檢查欄位是否存在
    if "DP" in variant.info:
        depth = variant.info["DP"]
        print(f"Depth: {depth}")

    # 取得所有 INFO 鍵
    print(f"INFO fields: {variant.info.keys()}")

    # 存取特定欄位
    if "AF" in variant.info:
        allele_freq = variant.info["AF"]
        print(f"Allele frequency: {allele_freq}")
```

### 樣本基因型資料

透過 `samples` 字典存取樣本資料：

```python
for variant in vcf:
    for sample_name in variant.samples:
        sample = variant.samples[sample_name]

        # 基因型（GT 欄位）
        gt = sample["GT"]
        print(f"{sample_name} genotype: {gt}")

        # 其他 FORMAT 欄位
        if "DP" in sample:
            print(f"{sample_name} depth: {sample['DP']}")
        if "GQ" in sample:
            print(f"{sample_name} quality: {sample['GQ']}")

        # 此基因型的等位基因
        alleles = sample.alleles
        print(f"{sample_name} alleles: {alleles}")

        # 定相
        if sample.phased:
            print(f"{sample_name} is phased")
```

**基因型表示：**
- `(0, 0)` - 同型合子參考
- `(0, 1)` - 異型合子
- `(1, 1)` - 同型合子替代
- `(None, None)` - 缺失基因型
- 已定相：`(0|1)` vs 未定相：`(0/1)`

## 寫入變異檔案

### 建立標頭

```python
header = pysam.VariantHeader()

# 添加 contig
header.contigs.add("chr1", length=248956422)
header.contigs.add("chr2", length=242193529)

# 添加 INFO 欄位
header.add_line('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">')
header.add_line('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">')

# 添加 FORMAT 欄位
header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">')

# 添加樣本
header.add_sample("sample1")
header.add_sample("sample2")

# 建立輸出檔案
outvcf = pysam.VariantFile("output.vcf", "w", header=header)
```

### 建立變異記錄

```python
# 建立新變異
record = outvcf.new_record()
record.chrom = "chr1"
record.pos = 100000
record.id = "rs123456"
record.ref = "A"
record.alts = ("G",)
record.qual = 30
record.filter.add("PASS")

# 設定 INFO 欄位
record.info["DP"] = 100
record.info["AF"] = (0.25,)

# 設定基因型資料
record.samples["sample1"]["GT"] = (0, 1)
record.samples["sample1"]["DP"] = 50
record.samples["sample2"]["GT"] = (0, 0)
record.samples["sample2"]["DP"] = 50

# 寫入檔案
outvcf.write(record)
```

## 過濾變異

### 基本過濾

```python
# 依品質過濾
for variant in vcf:
    if variant.qual >= 30:
        print(f"High quality variant: {variant.chrom}:{variant.pos}")

# 依深度過濾
for variant in vcf:
    if "DP" in variant.info and variant.info["DP"] >= 20:
        print(f"High depth variant: {variant.chrom}:{variant.pos}")

# 依等位基因頻率過濾
for variant in vcf:
    if "AF" in variant.info:
        for af in variant.info["AF"]:
            if af >= 0.01:
                print(f"Common variant: {variant.chrom}:{variant.pos}")
```

### 依基因型過濾

```python
# 找出樣本具有替代等位基因的變異
for variant in vcf:
    sample = variant.samples["sample1"]
    gt = sample["GT"]

    # 檢查是否有替代等位基因
    if gt and any(allele and allele > 0 for allele in gt):
        print(f"Sample has alt allele: {variant.chrom}:{variant.pos}")

    # 檢查是否為同型合子替代
    if gt == (1, 1):
        print(f"Homozygous alt: {variant.chrom}:{variant.pos}")
```

### 過濾欄位

```python
# 檢查 FILTER 狀態
for variant in vcf:
    if "PASS" in variant.filter or len(variant.filter) == 0:
        print(f"Passed filters: {variant.chrom}:{variant.pos}")
    else:
        print(f"Failed: {variant.filter.keys()}")
```

## 索引 VCF 檔案

為壓縮的 VCF 建立 tabix 索引：

```python
# 壓縮並索引
pysam.tabix_index("example.vcf", preset="vcf", force=True)
# 建立 example.vcf.gz 和 example.vcf.gz.tbi
```

或對 BCF 使用 bcftools：

```python
pysam.bcftools.index("example.bcf")
```

## 常見工作流程

### 提取特定樣本的變異

```python
invcf = pysam.VariantFile("input.vcf")
samples_to_keep = ["sample1", "sample3"]

# 建立具有樣本子集的新標頭
new_header = invcf.header.copy()
new_header.samples.clear()
for sample in samples_to_keep:
    new_header.samples.add(sample)

outvcf = pysam.VariantFile("output.vcf", "w", header=new_header)

for variant in invcf:
    # 建立新記錄
    new_record = outvcf.new_record(
        contig=variant.chrom,
        start=variant.start,
        stop=variant.stop,
        alleles=variant.alleles,
        id=variant.id,
        qual=variant.qual,
        filter=variant.filter,
        info=variant.info
    )

    # 複製選定樣本的基因型資料
    for sample in samples_to_keep:
        new_record.samples[sample].update(variant.samples[sample])

    outvcf.write(new_record)
```

### 計算等位基因頻率

```python
vcf = pysam.VariantFile("example.vcf")

for variant in vcf:
    total_alleles = 0
    alt_alleles = 0

    for sample_name in variant.samples:
        gt = variant.samples[sample_name]["GT"]
        if gt and None not in gt:
            total_alleles += 2
            alt_alleles += sum(1 for allele in gt if allele > 0)

    if total_alleles > 0:
        af = alt_alleles / total_alleles
        print(f"{variant.chrom}:{variant.pos} AF={af:.4f}")
```

### VCF 轉摘要表

```python
import csv

vcf = pysam.VariantFile("example.vcf")

with open("variants.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "DP"])

    for variant in vcf:
        writer.writerow([
            variant.chrom,
            variant.pos,
            variant.id or ".",
            variant.ref,
            ",".join(variant.alts) if variant.alts else ".",
            variant.qual or ".",
            variant.info.get("DP", ".")
        ])
```

## 效能提示

1. **使用 BCF 格式** 比 VCF 有更好的壓縮和更快的存取
2. **索引檔案** 使用 tabix 進行高效的區域查詢
3. **提早過濾** 減少處理不相關變異
4. **高效使用 INFO 欄位** - 存取前檢查是否存在
5. **批次寫入操作** 建立 VCF 檔案時

## 常見陷阱

1. **座標系統：** VCF 使用 1-based 座標，但 VariantRecord.start 是 0-based
2. **缺失資料：** 存取前總是檢查 INFO/FORMAT 欄位是否存在
3. **基因型元組：** 基因型是元組，不是列表—處理缺失資料的 None 值
4. **等位基因索引：** 在基因型 (0, 1) 中，0=REF、1=第一個 ALT、2=第二個 ALT 等
5. **索引需求：** 基於區域的 `fetch()` 對 VCF.gz 需要 tabix 索引
6. **標頭修改：** 子集化樣本時，正確更新標頭並複製 FORMAT 欄位
