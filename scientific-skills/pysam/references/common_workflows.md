# 使用 Pysam 的常見生物資訊工作流程

## 概述

本文件提供使用 pysam 的常見生物資訊工作流程實用範例，展示如何結合不同的檔案類型和操作。

## 品質控制工作流程

### 計算 BAM 統計

```python
import pysam

def calculate_bam_stats(bam_file):
    """計算 BAM 檔案的基本統計。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")

    stats = {
        "total_reads": 0,
        "mapped_reads": 0,
        "unmapped_reads": 0,
        "paired_reads": 0,
        "proper_pairs": 0,
        "duplicates": 0,
        "total_bases": 0,
        "mapped_bases": 0
    }

    for read in samfile.fetch(until_eof=True):
        stats["total_reads"] += 1

        if read.is_unmapped:
            stats["unmapped_reads"] += 1
        else:
            stats["mapped_reads"] += 1
            stats["mapped_bases"] += read.query_alignment_length

        if read.is_paired:
            stats["paired_reads"] += 1
            if read.is_proper_pair:
                stats["proper_pairs"] += 1

        if read.is_duplicate:
            stats["duplicates"] += 1

        stats["total_bases"] += read.query_length

    samfile.close()

    # 計算衍生統計
    stats["mapping_rate"] = stats["mapped_reads"] / stats["total_reads"] if stats["total_reads"] > 0 else 0
    stats["duplication_rate"] = stats["duplicates"] / stats["total_reads"] if stats["total_reads"] > 0 else 0

    return stats
```

### 檢查參考一致性

```python
def check_bam_reference_consistency(bam_file, fasta_file):
    """驗證 BAM 讀取與參考基因體的匹配。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")
    fasta = pysam.FastaFile(fasta_file)

    mismatches = 0
    total_checked = 0

    for read in samfile.fetch():
        if read.is_unmapped:
            continue

        # 取得比對區域的參考序列
        ref_seq = fasta.fetch(
            read.reference_name,
            read.reference_start,
            read.reference_end
        )

        # 取得比對到參考的讀取序列
        aligned_pairs = read.get_aligned_pairs(with_seq=True)

        for query_pos, ref_pos, ref_base in aligned_pairs:
            if query_pos is not None and ref_pos is not None and ref_base is not None:
                read_base = read.query_sequence[query_pos]
                if read_base.upper() != ref_base.upper():
                    mismatches += 1
                total_checked += 1

        if total_checked >= 10000:  # 抽樣前 10k 個位置
            break

    samfile.close()
    fasta.close()

    error_rate = mismatches / total_checked if total_checked > 0 else 0
    return {
        "positions_checked": total_checked,
        "mismatches": mismatches,
        "error_rate": error_rate
    }
```

## 覆蓋度分析

### 計算逐鹼基覆蓋度

```python
def calculate_coverage(bam_file, chrom, start, end):
    """計算區域中每個位置的覆蓋度。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")

    # 初始化覆蓋度陣列
    length = end - start
    coverage = [0] * length

    # 計算每個位置的覆蓋度
    for pileupcolumn in samfile.pileup(chrom, start, end):
        if start <= pileupcolumn.pos < end:
            coverage[pileupcolumn.pos - start] = pileupcolumn.nsegments

    samfile.close()

    return coverage
```

### 識別低覆蓋度區域

```python
def find_low_coverage_regions(bam_file, chrom, start, end, min_coverage=10):
    """找出覆蓋度低於閾值的區域。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")

    low_coverage_regions = []
    in_low_region = False
    region_start = None

    for pileupcolumn in samfile.pileup(chrom, start, end):
        pos = pileupcolumn.pos
        if pos < start or pos >= end:
            continue

        coverage = pileupcolumn.nsegments

        if coverage < min_coverage:
            if not in_low_region:
                region_start = pos
                in_low_region = True
        else:
            if in_low_region:
                low_coverage_regions.append((region_start, pos))
                in_low_region = False

    # 如果仍開啟則關閉最後區域
    if in_low_region:
        low_coverage_regions.append((region_start, end))

    samfile.close()

    return low_coverage_regions
```

### 計算覆蓋度統計

```python
def coverage_statistics(bam_file, chrom, start, end):
    """計算區域的覆蓋度統計。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")

    coverages = []

    for pileupcolumn in samfile.pileup(chrom, start, end):
        if start <= pileupcolumn.pos < end:
            coverages.append(pileupcolumn.nsegments)

    samfile.close()

    if not coverages:
        return None

    coverages.sort()
    n = len(coverages)

    return {
        "mean": sum(coverages) / n,
        "median": coverages[n // 2],
        "min": coverages[0],
        "max": coverages[-1],
        "positions": n
    }
```

## 變異分析

### 提取區域內的變異

```python
def extract_variants_in_genes(vcf_file, bed_file):
    """提取與基因區域重疊的變異。"""
    vcf = pysam.VariantFile(vcf_file)
    bed = pysam.TabixFile(bed_file)

    variants_by_gene = {}

    for gene in bed.fetch(parser=pysam.asBed()):
        gene_name = gene.name
        variants_by_gene[gene_name] = []

        # 找出基因區域內的變異
        for variant in vcf.fetch(gene.contig, gene.start, gene.end):
            variant_info = {
                "chrom": variant.chrom,
                "pos": variant.pos,
                "ref": variant.ref,
                "alt": variant.alts,
                "qual": variant.qual
            }
            variants_by_gene[gene_name].append(variant_info)

    vcf.close()
    bed.close()

    return variants_by_gene
```

### 使用覆蓋度註解變異

```python
def annotate_variants_with_coverage(vcf_file, bam_file, output_file):
    """為變異添加覆蓋度資訊。"""
    vcf = pysam.VariantFile(vcf_file)
    samfile = pysam.AlignmentFile(bam_file, "rb")

    # 如果標頭中沒有 DP 則添加
    if "DP" not in vcf.header.info:
        vcf.header.info.add("DP", "1", "Integer", "Total Depth from BAM")

    outvcf = pysam.VariantFile(output_file, "w", header=vcf.header)

    for variant in vcf:
        # 取得變異位置的覆蓋度
        coverage = samfile.count(
            variant.chrom,
            variant.pos - 1,  # 轉換為 0-based
            variant.pos
        )

        # 添加到 INFO 欄位
        variant.info["DP"] = coverage

        outvcf.write(variant)

    vcf.close()
    samfile.close()
    outvcf.close()
```

### 依讀取支持過濾變異

```python
def filter_variants_by_support(vcf_file, bam_file, output_file, min_alt_reads=3):
    """過濾需要最少替代等位基因支持的變異。"""
    vcf = pysam.VariantFile(vcf_file)
    samfile = pysam.AlignmentFile(bam_file, "rb")
    outvcf = pysam.VariantFile(output_file, "w", header=vcf.header)

    for variant in vcf:
        # 計算支持每個等位基因的讀取
        allele_counts = {variant.ref: 0}
        for alt in variant.alts:
            allele_counts[alt] = 0

        # 在變異位置進行堆疊
        for pileupcolumn in samfile.pileup(
            variant.chrom,
            variant.pos - 1,
            variant.pos
        ):
            if pileupcolumn.pos == variant.pos - 1:  # 0-based
                for pileupread in pileupcolumn.pileups:
                    if not pileupread.is_del and not pileupread.is_refskip:
                        base = pileupread.alignment.query_sequence[
                            pileupread.query_position
                        ]
                        if base in allele_counts:
                            allele_counts[base] += 1

        # 檢查是否有任何 alt 等位基因有足夠支持
        has_support = any(
            allele_counts.get(alt, 0) >= min_alt_reads
            for alt in variant.alts
        )

        if has_support:
            outvcf.write(variant)

    vcf.close()
    samfile.close()
    outvcf.close()
```

## 序列提取

### 提取變異周圍的序列

```python
def extract_variant_contexts(vcf_file, fasta_file, output_file, window=50):
    """提取變異周圍的參考序列。"""
    vcf = pysam.VariantFile(vcf_file)
    fasta = pysam.FastaFile(fasta_file)

    with open(output_file, 'w') as out:
        for variant in vcf:
            # 取得序列上下文
            start = max(0, variant.pos - window - 1)  # 轉換為 0-based
            end = variant.pos + window

            context = fasta.fetch(variant.chrom, start, end)

            # 標記變異位置
            var_pos_in_context = variant.pos - 1 - start

            out.write(f">{variant.chrom}:{variant.pos} {variant.ref}>{variant.alts}\n")
            out.write(context[:var_pos_in_context].lower())
            out.write(context[var_pos_in_context:var_pos_in_context+len(variant.ref)].upper())
            out.write(context[var_pos_in_context+len(variant.ref):].lower())
            out.write("\n")

    vcf.close()
    fasta.close()
```

### 提取基因序列

```python
def extract_gene_sequences(bed_file, fasta_file, output_fasta):
    """從 BED 檔案提取基因序列。"""
    bed = pysam.TabixFile(bed_file)
    fasta = pysam.FastaFile(fasta_file)

    with open(output_fasta, 'w') as out:
        for gene in bed.fetch(parser=pysam.asBed()):
            sequence = fasta.fetch(gene.contig, gene.start, gene.end)

            # 處理股向
            if hasattr(gene, 'strand') and gene.strand == '-':
                # 反向互補
                complement = str.maketrans("ATGCatgcNn", "TACGtacgNn")
                sequence = sequence.translate(complement)[::-1]

            out.write(f">{gene.name} {gene.contig}:{gene.start}-{gene.end}\n")

            # 以 60 字元為一行寫入序列
            for i in range(0, len(sequence), 60):
                out.write(sequence[i:i+60] + "\n")

    bed.close()
    fasta.close()
```

## 讀取過濾和子集化

### 依區域和品質過濾 BAM

```python
def filter_bam(input_bam, output_bam, chrom, start, end, min_mapq=20):
    """依區域和比對品質過濾 BAM 檔案。"""
    infile = pysam.AlignmentFile(input_bam, "rb")
    outfile = pysam.AlignmentFile(output_bam, "wb", template=infile)

    for read in infile.fetch(chrom, start, end):
        if read.mapping_quality >= min_mapq and not read.is_duplicate:
            outfile.write(read)

    infile.close()
    outfile.close()

    # 建立索引
    pysam.index(output_bam)
```

### 提取特定變異的讀取

```python
def extract_reads_at_variants(bam_file, vcf_file, output_bam, window=100):
    """提取與變異位置重疊的讀取。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")
    vcf = pysam.VariantFile(vcf_file)
    outfile = pysam.AlignmentFile(output_bam, "wb", template=samfile)

    # 收集所有讀取（使用 set 避免重複）
    reads_to_keep = set()

    for variant in vcf:
        start = max(0, variant.pos - window - 1)
        end = variant.pos + window

        for read in samfile.fetch(variant.chrom, start, end):
            reads_to_keep.add(read.query_name)

    # 寫入所有讀取
    samfile.close()
    samfile = pysam.AlignmentFile(bam_file, "rb")

    for read in samfile.fetch(until_eof=True):
        if read.query_name in reads_to_keep:
            outfile.write(read)

    samfile.close()
    vcf.close()
    outfile.close()

    pysam.index(output_bam)
```

## 整合工作流程

### 從 BAM 建立覆蓋度軌道

```python
def create_coverage_bedgraph(bam_file, output_file, chrom=None):
    """從 BAM 建立 bedGraph 覆蓋度軌道。"""
    samfile = pysam.AlignmentFile(bam_file, "rb")

    chroms = [chrom] if chrom else samfile.references

    with open(output_file, 'w') as out:
        out.write("track type=bedGraph name=\"Coverage\"\n")

        for chrom in chroms:
            current_cov = None
            region_start = None

            for pileupcolumn in samfile.pileup(chrom):
                pos = pileupcolumn.pos
                cov = pileupcolumn.nsegments

                if cov != current_cov:
                    # 寫入前一個區域
                    if current_cov is not None:
                        out.write(f"{chrom}\t{region_start}\t{pos}\t{current_cov}\n")

                    # 開始新區域
                    current_cov = cov
                    region_start = pos

            # 寫入最後區域
            if current_cov is not None:
                out.write(f"{chrom}\t{region_start}\t{pos+1}\t{current_cov}\n")

    samfile.close()
```

### 合併多個 VCF 檔案

```python
def merge_vcf_samples(vcf_files, output_file):
    """合併多個單樣本 VCF。"""
    # 開啟所有輸入檔案
    vcf_readers = [pysam.VariantFile(f) for f in vcf_files]

    # 建立合併標頭
    merged_header = vcf_readers[0].header.copy()
    for vcf in vcf_readers[1:]:
        for sample in vcf.header.samples:
            merged_header.samples.add(sample)

    outvcf = pysam.VariantFile(output_file, "w", header=merged_header)

    # 取得所有變異位置
    all_variants = {}
    for vcf in vcf_readers:
        for variant in vcf:
            key = (variant.chrom, variant.pos, variant.ref, variant.alts)
            if key not in all_variants:
                all_variants[key] = []
            all_variants[key].append(variant)

    # 寫入合併的變異
    for key, variants in sorted(all_variants.items()):
        # 從第一個變異建立合併記錄
        merged = outvcf.new_record(
            contig=variants[0].chrom,
            start=variants[0].start,
            stop=variants[0].stop,
            alleles=variants[0].alleles
        )

        # 添加所有樣本的基因型
        for variant in variants:
            for sample in variant.samples:
                merged.samples[sample].update(variant.samples[sample])

        outvcf.write(merged)

    # 關閉所有檔案
    for vcf in vcf_readers:
        vcf.close()
    outvcf.close()
```

## 工作流程效能提示

1. **使用索引檔案** 進行所有隨機存取操作
2. **分析多個獨立區域時平行處理區域**
3. **盡可能串流資料** - 避免將整個檔案載入記憶體
4. **明確關閉檔案** 以釋放資源
5. **使用 `until_eof=True`** 循序處理整個檔案
6. **批次操作** 同一檔案以最小化 I/O
7. **注意記憶體使用** 高覆蓋度區域的堆疊操作
8. **當只需要計數時使用 count() 而非 pileup()**

## 常見整合模式

1. **BAM + 參考**：驗證比對、提取比對序列
2. **BAM + VCF**：驗證變異、計算等位基因頻率
3. **VCF + BED**：使用基因/區域資訊註解變異
4. **BAM + BED**：計算特定區域的覆蓋度統計
5. **FASTA + VCF**：提取變異上下文序列
6. **多個 BAM**：跨樣本比較覆蓋度或變異
7. **BAM + FASTQ**：提取未比對讀取以重新比對
