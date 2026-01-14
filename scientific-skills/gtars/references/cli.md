# 命令列介面

Gtars 提供全面的 CLI，可直接從終端機進行基因體區間分析。

## 安裝

```bash
# 安裝所有功能
cargo install gtars-cli --features "uniwig overlaprs igd bbcache scoring fragsplit"

# 只安裝特定功能
cargo install gtars-cli --features "uniwig overlaprs"
```

## 全域選項

```bash
# 顯示說明
gtars --help

# 顯示版本
gtars --version

# 詳細輸出
gtars --verbose <command>

# 安靜模式
gtars --quiet <command>
```

## IGD 命令

建立和查詢 IGD 索引以進行重疊偵測：

```bash
# 建立 IGD 索引
gtars igd build --input regions.bed --output regions.igd

# 查詢單一區域
gtars igd query --index regions.igd --region "chr1:1000-2000"

# 從檔案查詢
gtars igd query --index regions.igd --query-file queries.bed --output results.bed

# 計數重疊
gtars igd count --index regions.igd --query-file queries.bed
```

## 重疊命令

計算基因體區域集之間的重疊：

```bash
# 尋找重疊區域
gtars overlaprs overlap --set-a regions_a.bed --set-b regions_b.bed --output overlaps.bed

# 計數重疊
gtars overlaprs count --set-a regions_a.bed --set-b regions_b.bed

# 依重疊篩選區域
gtars overlaprs filter --input regions.bed --filter overlapping.bed --output filtered.bed

# 區域差集
gtars overlaprs subtract --set-a regions_a.bed --set-b regions_b.bed --output difference.bed
```

## Uniwig 命令

從基因體區間產生涵蓋度軌跡：

```bash
# 產生涵蓋度軌跡
gtars uniwig generate --input fragments.bed --output coverage.wig

# 指定解析度
gtars uniwig generate --input fragments.bed --output coverage.wig --resolution 10

# 產生 BigWig
gtars uniwig generate --input fragments.bed --output coverage.bw --format bigwig

# 鏈特異性涵蓋度
gtars uniwig generate --input fragments.bed --output forward.wig --strand +
```

## BBCache 命令

從 BEDbase.org 快取和管理 BED 檔案：

```bash
# 從 bedbase 快取 BED 檔案
gtars bbcache fetch --id <bedbase_id> --output cached.bed

# 列出已快取的檔案
gtars bbcache list

# 清除快取
gtars bbcache clear

# 更新快取
gtars bbcache update
```

## 評分命令

對參考資料集評分片段重疊：

```bash
# 評分片段
gtars scoring score --fragments fragments.bed --reference reference.bed --output scores.txt

# 批次評分
gtars scoring batch --fragments-dir ./fragments/ --reference reference.bed --output-dir ./scores/

# 帶權重評分
gtars scoring score --fragments fragments.bed --reference reference.bed --weights weights.txt --output scores.txt
```

## FragSplit 命令

依細胞條碼或群集分割片段檔案：

```bash
# 依條碼分割
gtars fragsplit split --input fragments.tsv --barcodes barcodes.txt --output-dir ./split/

# 依群集分割
gtars fragsplit cluster-split --input fragments.tsv --clusters clusters.txt --output-dir ./clustered/

# 篩選片段
gtars fragsplit filter --input fragments.tsv --min-fragments 100 --output filtered.tsv
```

## 常見工作流程

### 工作流程 1：重疊分析流程

```bash
# 步驟 1：為參考建立 IGD 索引
gtars igd build --input reference_regions.bed --output reference.igd

# 步驟 2：用實驗資料查詢
gtars igd query --index reference.igd --query-file experimental.bed --output overlaps.bed

# 步驟 3：產生統計
gtars overlaprs count --set-a experimental.bed --set-b reference_regions.bed
```

### 工作流程 2：涵蓋度軌跡產生

```bash
# 步驟 1：產生涵蓋度
gtars uniwig generate --input fragments.bed --output coverage.wig --resolution 10

# 步驟 2：轉換為 BigWig
gtars uniwig generate --input fragments.bed --output coverage.bw --format bigwig
```

### 工作流程 3：片段處理

```bash
# 步驟 1：篩選片段
gtars fragsplit filter --input raw_fragments.tsv --min-fragments 100 --output filtered.tsv

# 步驟 2：依群集分割
gtars fragsplit cluster-split --input filtered.tsv --clusters clusters.txt --output-dir ./by_cluster/

# 步驟 3：對參考評分
gtars scoring batch --fragments-dir ./by_cluster/ --reference reference.bed --output-dir ./scores/
```

## 輸入/輸出格式

### BED 格式
標準 3 欄或擴展 BED 格式：
```
chr1    1000    2000
chr1    3000    4000
chr2    5000    6000
```

### 片段格式（TSV）
用於單細胞片段的製表符分隔格式：
```
chr1    1000    2000    BARCODE1
chr1    3000    4000    BARCODE2
chr2    5000    6000    BARCODE1
```

### WIG 格式
用於涵蓋度軌跡的 Wiggle 格式：
```
fixedStep chrom=chr1 start=1000 step=10
12
15
18
```

## 效能選項

```bash
# 設定執行緒數
gtars --threads 8 <command>

# 記憶體限制
gtars --memory-limit 4G <command>

# 緩衝區大小
gtars --buffer-size 10000 <command>
```

## 錯誤處理

```bash
# 錯誤時繼續
gtars --continue-on-error <command>

# 嚴格模式（警告時失敗）
gtars --strict <command>

# 記錄到檔案
gtars --log-file output.log <command>
```
