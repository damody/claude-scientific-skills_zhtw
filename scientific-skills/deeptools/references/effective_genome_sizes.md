# 有效基因組大小

## 定義

有效基因組大小指的是「可定位」基因組的長度 - 可被定序讀段唯一定位的區域。此指標對於許多 deepTools 命令中的正確正規化至關重要。

## 為何重要

- RPGC 正規化所需（`--normalizeUsing RPGC`）
- 影響覆蓋度計算的準確性
- 必須與您的資料處理方法相匹配（過濾 vs 未過濾讀段）

## 計算方法

1. **非 N 鹼基**：基因組序列中非 N 核苷酸的計數
2. **唯一可定位性**：可被唯一定位的特定大小區域（可能考慮編輯距離）

## 常見生物體數值

### 使用非 N 鹼基方法

| 生物體 | 組裝版本 | 有效大小 | 完整命令 |
|--------|----------|----------|----------|
| 人類 | GRCh38/hg38 | 2,913,022,398 | `--effectiveGenomeSize 2913022398` |
| 人類 | GRCh37/hg19 | 2,864,785,220 | `--effectiveGenomeSize 2864785220` |
| 小鼠 | GRCm39/mm39 | 2,654,621,837 | `--effectiveGenomeSize 2654621837` |
| 小鼠 | GRCm38/mm10 | 2,652,783,500 | `--effectiveGenomeSize 2652783500` |
| 斑馬魚 | GRCz11 | 1,368,780,147 | `--effectiveGenomeSize 1368780147` |
| *果蠅* | dm6 | 142,573,017 | `--effectiveGenomeSize 142573017` |
| *秀麗隱桿線蟲* | WBcel235/ce11 | 100,286,401 | `--effectiveGenomeSize 100286401` |
| *秀麗隱桿線蟲* | ce10 | 100,258,171 | `--effectiveGenomeSize 100258171` |

### 人類（GRCh38）按讀段長度

對於品質過濾的讀段，數值依讀段長度而異：

| 讀段長度 | 有效大小 |
|----------|----------|
| 50bp | ~27 億 |
| 75bp | ~28 億 |
| 100bp | ~28 億 |
| 150bp | ~29 億 |
| 250bp | ~29 億 |

### 小鼠（GRCm38）按讀段長度

| 讀段長度 | 有效大小 |
|----------|----------|
| 50bp | ~23 億 |
| 75bp | ~25 億 |
| 100bp | ~26 億 |

## deepTools 中的使用

有效基因組大小最常用於：

### 使用 RPGC 正規化的 bamCoverage
```bash
bamCoverage --bam input.bam --outFileName output.bw \
    --normalizeUsing RPGC \
    --effectiveGenomeSize 2913022398
```

### 使用 RPGC 正規化的 bamCompare
```bash
bamCompare -b1 treatment.bam -b2 control.bam \
    --outFileName comparison.bw \
    --scaleFactorsMethod RPGC \
    --effectiveGenomeSize 2913022398
```

### computeGCBias / correctGCBias
```bash
computeGCBias --bamfile input.bam \
    --effectiveGenomeSize 2913022398 \
    --genome genome.2bit \
    --fragmentLength 200 \
    --biasPlot bias.png
```

## 選擇正確的數值

**對於大多數分析：** 對您的參考基因組使用非 N 鹼基方法的數值

**對於過濾資料：** 如果您應用嚴格的品質過濾或移除多重定位讀段，請考慮使用讀段長度特定的數值

**不確定時：** 使用保守的非 N 鹼基數值 - 它更廣泛適用

## 常見縮寫

deepTools 在某些情況下也接受這些縮寫值：

- `hs` 或 `GRCh38`：2913022398
- `mm` 或 `GRCm38`：2652783500
- `dm` 或 `dm6`：142573017
- `ce` 或 `ce10`：100286401

請檢查您特定 deepTools 版本的文件以確認支援的縮寫。

## 計算自訂數值

對於自訂基因組或組裝版本，計算非 N 鹼基計數：

```bash
# 使用 faCount（UCSC 工具）
faCount genome.fa | grep "total" | awk '{print $2-$7}'

# 使用 seqtk
seqtk comp genome.fa | awk '{x+=$2}END{print x}'
```

## 參考資料

關於最新的有效基因組大小和詳細計算方法，請參閱：
- deepTools 文件：https://deeptools.readthedocs.io/en/latest/content/feature/effectiveGenomeSize.html
- ENCODE 文件以獲取參考基因組詳細資訊
