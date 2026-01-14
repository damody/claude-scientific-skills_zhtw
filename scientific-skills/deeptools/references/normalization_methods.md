# deepTools 正規化方法

本文件說明 deepTools 中可用的各種正規化方法以及何時使用每種方法。

## 為何要正規化？

正規化對於以下情況至關重要：
1. **比較具有不同定序深度的樣本**
2. **考慮文庫大小差異**
3. **使覆蓋度值在不同實驗間可解釋**
4. **實現條件間的公平比較**

如果沒有正規化，擁有 1 億讀段的樣本會顯示比擁有 5 千萬讀段的樣本更高的覆蓋度，即使真實的生物訊號相同。

---

## 可用的正規化方法

### 1. RPKM（每千鹼基每百萬定位讀段的讀段數）

**公式：** `（讀段數）/（區域長度（kb）× 總定位讀段（百萬）`

**何時使用：**
- 比較同一樣本內的不同基因組區域
- 調整定序深度和區域長度
- RNA-seq 基因表現分析

**可用於：** `bamCoverage`

**範例：**
```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing RPKM
```

**解釋：** RPKM 為 10 表示每千鹼基特徵每百萬定位讀段有 10 個讀段。

**優點：**
- 同時考慮區域長度和文庫大小
- 在基因組學中廣泛使用和理解

**缺點：**
- 如果總 RNA 含量不同，不適合樣本間比較
- 當比較組成非常不同的樣本時可能產生誤導

---

### 2. CPM（每百萬定位讀段的計數）

**公式：** `（讀段數）/（總定位讀段（百萬））`

**又稱：** RPM（每百萬讀段數）

**何時使用：**
- 在不同樣本間比較相同基因組區域
- 當區域長度固定或不相關時
- ChIP-seq、ATAC-seq、DNase-seq 分析

**可用於：** `bamCoverage`、`bamCompare`

**範例：**
```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing CPM
```

**解釋：** CPM 為 5 表示該區間每百萬定位讀段有 5 個讀段。

**優點：**
- 簡單直觀
- 適合比較具有不同定序深度的樣本
- 適用於比較固定大小的區間

**缺點：**
- 不考慮區域長度
- 受高豐度區域影響（例如 RNA-seq 中的 rRNA）

---

### 3. BPM（每百萬定位讀段的區間數）

**公式：** `（區間中的讀段數）/（所有區間中讀段總和（百萬））`

**與 CPM 的關鍵差異：** 只考慮落在分析區間內的讀段，而非所有定位讀段。

**何時使用：**
- 類似於 CPM，但當您想排除分析區域外的讀段時
- 在忽略背景的情況下比較特定基因組區域

**可用於：** `bamCoverage`、`bamCompare`

**範例：**
```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing BPM
```

**解釋：** BPM 只考慮區間化區域中的讀段。

**優點：**
- 將正規化集中在分析區域
- 較少受未分析區域讀段的影響

**缺點：**
- 較不常用，可能較難與發表的資料比較

---

### 4. RPGC（每基因組內容的讀段數）

**公式：** `（讀段數 × 縮放因子）/ 有效基因組大小`

**縮放因子：** 計算以達到 1× 基因組覆蓋度（每鹼基 1 個讀段）

**何時使用：**
- 想要樣本間可比較的覆蓋度值
- 需要可解釋的絕對覆蓋度值
- 比較總讀段計數非常不同的樣本
- ChIP-seq 與加入正規化情境

**可用於：** `bamCoverage`、`bamCompare`

**需要：** `--effectiveGenomeSize` 參數

**範例：**
```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing RPGC \
    --effectiveGenomeSize 2913022398
```

**解釋：** 訊號值近似覆蓋深度（例如值為 2 ≈ 2× 覆蓋度）。

**優點：**
- 產生 1× 正規化覆蓋度
- 以基因組覆蓋度來解釋
- 適合比較具有不同定序深度的樣本

**缺點：**
- 需要知道有效基因組大小
- 假設均勻覆蓋（對於有峰值的 ChIP-seq 不成立）

---

### 5. None（無正規化）

**公式：** 原始讀段計數

**何時使用：**
- 初步分析
- 當樣本具有相同的文庫大小時（罕見）
- 當下游工具會進行正規化時
- 除錯或品質控制

**可用於：** 所有工具（通常是預設）

**範例：**
```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing None
```

**解釋：** 每個區間的原始讀段計數。

**優點：**
- 不做任何假設
- 有助於查看原始資料
- 最快的計算

**缺點：**
- 無法公平比較具有不同定序深度的樣本
- 不適合出版品質的圖形

---

### 6. SES（選擇性富集統計）

**方法：** 訊號提取縮放 - 比較 ChIP 與對照的更複雜方法

**何時使用：**
- 使用 bamCompare 的 ChIP-seq 分析
- 想要複雜的背景校正
- 作為簡單讀段計數縮放的替代方案

**可用於：** 僅限 `bamCompare`

**範例：**
```bash
bamCompare -b1 chip.bam -b2 input.bam -o output.bw \
    --scaleFactorsMethod SES
```

**注意：** SES 專為 ChIP-seq 資料設計，對於雜訊資料可能比簡單的讀段計數縮放效果更好。

---

### 7. readCount（讀段計數縮放）

**方法：** 按樣本間總讀段計數的比例縮放

**何時使用：**
- `bamCompare` 的預設
- 補償比較中的定序深度差異
- 當您相信總讀段計數反映文庫大小時

**可用於：** `bamCompare`

**範例：**
```bash
bamCompare -b1 treatment.bam -b2 control.bam -o output.bw \
    --scaleFactorsMethod readCount
```

**工作原理：** 如果樣本 1 有 1 億讀段，樣本 2 有 5 千萬讀段，則樣本 2 在比較前會被縮放 2×。

---

## 正規化方法選擇指南

### ChIP-seq 覆蓋度軌跡

**建議：** RPGC 或 CPM

```bash
bamCoverage --bam chip.bam --outFileName chip.bw \
    --normalizeUsing RPGC \
    --effectiveGenomeSize 2913022398 \
    --extendReads 200 \
    --ignoreDuplicates
```

**理由：** 考慮定序深度差異；RPGC 提供可解釋的覆蓋度值。

---

### ChIP-seq 比較（處理 vs 對照）

**建議：** 使用 readCount 或 SES 縮放的 log2 比值

```bash
bamCompare -b1 chip.bam -b2 input.bam -o ratio.bw \
    --operation log2 \
    --scaleFactorsMethod readCount \
    --extendReads 200 \
    --ignoreDuplicates
```

**理由：** Log2 比值顯示富集（正值）和耗竭（負值）；readCount 調整深度。

---

### RNA-seq 覆蓋度軌跡

**建議：** CPM 或 RPKM

```bash
# 正股特異性
bamCoverage --bam rnaseq.bam --outFileName forward.bw \
    --normalizeUsing CPM \
    --filterRNAstrand forward

# 對於基因層級：RPKM 考慮基因長度
bamCoverage --bam rnaseq.bam --outFileName output.bw \
    --normalizeUsing RPKM
```

**理由：** CPM 用於比較固定寬度區間；RPKM 用於基因（考慮長度）。

---

### ATAC-seq

**建議：** RPGC 或 CPM

```bash
bamCoverage --bam atac_shifted.bam --outFileName atac.bw \
    --normalizeUsing RPGC \
    --effectiveGenomeSize 2913022398
```

**理由：** 類似於 ChIP-seq；想要跨樣本可比較的覆蓋度。

---

### 樣本相關性分析

**建議：** CPM 或 RPGC

```bash
multiBamSummary bins \
    --bamfiles sample1.bam sample2.bam sample3.bam \
    -o readCounts.npz

plotCorrelation -in readCounts.npz \
    --corMethod pearson \
    --whatToShow heatmap \
    -o correlation.png
```

**注意：** `multiBamSummary` 不會明確正規化，但相關性分析對縮放具有穩健性。對於文庫大小差異很大的情況，考慮先正規化 BAM 檔案或使用 `multiBigwigSummary` 與 CPM 正規化的 bigWig 檔案。

---

## 進階正規化考量

### Spike-in 正規化

對於具有 spike-in 對照的實驗（例如 ChIP-seq 的*果蠅*染色質 spike-in）：

1. 從 spike-in 讀段計算縮放因子
2. 使用 `--scaleFactor` 參數應用自訂縮放因子

```bash
# 計算 spike-in 因子（範例：0.8）
SCALE_FACTOR=0.8

bamCoverage --bam chip.bam --outFileName chip_spikenorm.bw \
    --scaleFactor ${SCALE_FACTOR} \
    --extendReads 200
```

---

### 手動縮放因子

您可以應用自訂縮放因子：

```bash
# 應用 2× 縮放
bamCoverage --bam input.bam --outFileName output.bw \
    --scaleFactor 2.0
```

---

### 染色體排除

從正規化計算中排除特定染色體：

```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing RPGC \
    --effectiveGenomeSize 2913022398 \
    --ignoreForNormalization chrX chrY chrM
```

**何時使用：** 混合性別樣本中的性染色體、粒線體 DNA 或具有異常覆蓋度的染色體。

---

## 常見陷阱

### 1. 對區間資料使用 RPKM
**問題：** RPKM 考慮區域長度，但所有區間大小相同
**解決方案：** 改用 CPM 或 RPGC

### 2. 比較未正規化的樣本
**問題：** 定序深度 2× 的樣本顯示 2× 的訊號
**解決方案：** 比較樣本時始終正規化

### 3. 錯誤的有效基因組大小
**問題：** 對 hg38 資料使用 hg19 基因組大小
**解決方案：** 再次檢查基因組組裝版本並使用正確的大小

### 4. GC 校正後忽略重複
**問題：** 可能引入偏差
**解決方案：** 永遠不要在 `correctGCBias` 後使用 `--ignoreDuplicates`

### 5. 不帶有效基因組大小使用 RPGC
**問題：** 命令失敗
**解決方案：** 使用 RPGC 時始終指定 `--effectiveGenomeSize`

---

## 不同比較的正規化

### 樣本內比較（不同區域）
**使用：** RPKM（考慮區域長度）

### 樣本間比較（相同區域）
**使用：** CPM、RPGC 或 BPM（考慮文庫大小）

### 處理 vs 對照
**使用：** 帶有 log2 比值和 readCount/SES 縮放的 bamCompare

### 多樣本相關性
**使用：** CPM 或 RPGC 正規化的 bigWig 檔案，然後使用 multiBigwigSummary

---

## 快速參考表

| 方法 | 考慮深度 | 考慮長度 | 最適用於 | 命令 |
|------|----------|----------|----------|------|
| RPKM | ✓ | ✓ | RNA-seq 基因 | `--normalizeUsing RPKM` |
| CPM | ✓ | ✗ | 固定大小區間 | `--normalizeUsing CPM` |
| BPM | ✓ | ✗ | 特定區域 | `--normalizeUsing BPM` |
| RPGC | ✓ | ✗ | 可解釋的覆蓋度 | `--normalizeUsing RPGC --effectiveGenomeSize X` |
| None | ✗ | ✗ | 原始資料 | `--normalizeUsing None` |
| SES | ✓ | ✗ | ChIP 比較 | `bamCompare --scaleFactorsMethod SES` |
| readCount | ✓ | ✗ | ChIP 比較 | `bamCompare --scaleFactorsMethod readCount` |

---

## 延伸閱讀

有關正規化理論和最佳實踐的更多詳細資訊：
- deepTools 文件：https://deeptools.readthedocs.io/
- ENCODE ChIP-seq 分析指南
- RNA-seq 正規化論文（DESeq2、TMM 方法）
