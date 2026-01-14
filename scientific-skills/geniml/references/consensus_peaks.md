# 共識峰：Universe 建構

## 概述

Geniml 提供從 BED 檔案集合建構基因體「universes」的工具——來自 BED 檔案集合的標準化參考共識峰集。這些 universes 代表分析資料集顯示顯著覆蓋重疊的基因體區域，作為符記化和分析的參考詞彙表。

## 使用時機

在以下情況使用共識峰建立：
- 從多個實驗建構參考峰集
- 為 Region2Vec 或 BEDspace 符記化建立 universe 檔案
- 跨資料集集合標準化基因體區域
- 以統計顯著性定義感興趣的區域

## 工作流程

### 步驟 1：合併 BED 檔案

將所有 BED 檔案合併為單一檔案：

```bash
cat /path/to/bed/files/*.bed > combined_files.bed
```

### 步驟 2：生成覆蓋軌跡

使用 uniwig 以平滑視窗建立 bigWig 覆蓋軌跡：

```bash
uniwig -m 25 combined_files.bed chrom.sizes coverage/
```

**參數：**
- `-m 25`：平滑視窗大小（25bp 典型用於染色質可及性）
- `chrom.sizes`：您的基因體的染色體大小檔案
- `coverage/`：bigWig 檔案的輸出目錄

平滑視窗有助於減少噪聲並建立更穩健的峰邊界。

### 步驟 3：建構 Universe

使用四種方法之一建構共識峰：

## Universe 建構方法

### 1. 覆蓋截止（CC）

使用固定覆蓋閾值的最簡單方法：

```bash
geniml universe build cc \
  --coverage-folder coverage/ \
  --output-file universe_cc.bed \
  --cutoff 5 \
  --merge 100 \
  --filter-size 50
```

**參數：**
- `--cutoff`：覆蓋閾值（1 = 聯集；檔案數 = 交集）
- `--merge`：合併相鄰峰的距離（bp）
- `--filter-size`：納入的最小峰大小（bp）

**使用時機：** 簡單的閾值基礎選擇即足夠時

### 2. 彈性覆蓋截止（CCF）

為邊界和區域核心的可能性截止建立信賴區間：

```bash
geniml universe build ccf \
  --coverage-folder coverage/ \
  --output-file universe_ccf.bed \
  --cutoff 5 \
  --confidence 0.95 \
  --merge 100 \
  --filter-size 50
```

**額外參數：**
- `--confidence`：彈性邊界的信賴水準（0-1）

**使用時機：** 應捕捉峰邊界的不確定性時

### 3. 最大概似（ML）

建構考慮區域起始/結束位置的機率模型：

```bash
geniml universe build ml \
  --coverage-folder coverage/ \
  --output-file universe_ml.bed \
  --merge 100 \
  --filter-size 50 \
  --model-type gaussian
```

**參數：**
- `--model-type`：概似估計的分布（gaussian、poisson）

**使用時機：** 峰位置的統計建模很重要時

### 4. 隱馬可夫模型（HMM）

將基因體區域建模為隱藏狀態，覆蓋作為發射：

```bash
geniml universe build hmm \
  --coverage-folder coverage/ \
  --output-file universe_hmm.bed \
  --states 3 \
  --merge 100 \
  --filter-size 50
```

**參數：**
- `--states`：HMM 隱藏狀態數（通常 2-5）

**使用時機：** 應捕捉複雜的基因體狀態模式時

## Python API

```python
from geniml.universe import build_universe

# 使用覆蓋截止方法建構
universe = build_universe(
    coverage_folder='coverage/',
    method='cc',
    cutoff=5,
    merge_distance=100,
    min_size=50,
    output_file='universe.bed'
)
```

## 方法比較

| 方法 | 複雜度 | 彈性 | 計算成本 | 最適用於 |
|------|--------|------|----------|----------|
| CC | 低 | 低 | 低 | 快速參考集 |
| CCF | 中 | 中 | 中 | 邊界不確定性 |
| ML | 高 | 高 | 高 | 統計嚴謹性 |
| HMM | 高 | 高 | 非常高 | 複雜模式 |

## 最佳實踐

### 選擇方法

1. **從 CC 開始**：快速且可解釋，適合初步探索
2. **使用 CCF**：當峰邊界不確定或有噪聲時
3. **應用 ML**：用於發表品質的統計分析
4. **部署 HMM**：用於建模複雜的染色質狀態

### 參數選擇

**覆蓋截止：**
- `cutoff = 1`：所有峰的聯集（最寬鬆）
- `cutoff = n_files`：交集（最嚴格）
- `cutoff = 0.5 * n_files`：中等共識（典型選擇）

**合併距離：**
- ATAC-seq：100-200bp
- ChIP-seq（窄峰）：50-100bp
- ChIP-seq（寬峰）：500-1000bp

**過濾大小：**
- 最小 30bp 以避免假象
- 50-100bp 適用於大多數分析
- 較大用於寬組蛋白標記

### 品質控制

建構後評估 universe 品質：

```python
from geniml.evaluation import assess_universe

metrics = assess_universe(
    universe_file='universe.bed',
    coverage_folder='coverage/',
    bed_files='bed_files/'
)

print(f"區域數量：{metrics['n_regions']}")
print(f"平均區域大小：{metrics['mean_size']:.1f}bp")
print(f"輸入峰覆蓋率：{metrics['coverage']:.1%}")
```

**關鍵指標：**
- **區域數量**：應捕捉主要特徵而不過度片段化
- **大小分布**：應符合預期的生物學（例如：ATAC-seq 約 500bp）
- **輸入覆蓋率**：代表的原始峰比例（通常 >80%）

## 輸出格式

共識峰儲存為具有三個必要欄位的 BED 檔案：

```
chr1    1000    1500
chr1    2000    2800
chr2    500     1000
```

根據方法，額外欄位可能包含信賴分數或狀態註解。

## 常見工作流程

### 用於 Region2Vec

1. 使用偏好的方法建構 universe
2. 使用 universe 作為符記化參考
3. 符記化 BED 檔案
4. 訓練 Region2Vec 模型

### 用於 BEDspace

1. 從所有資料集建構 universe
2. 在預處理步驟中使用 universe
3. 使用元資料訓練 BEDspace
4. 跨區域和標籤查詢

### 用於 scEmbed

1. 從批量或聚合的 scATAC-seq 建立 universe
2. 用於細胞符記化
3. 訓練 scEmbed 模型
4. 生成細胞嵌入

## 疑難排解

**區域太少：** 降低截止閾值或減少過濾大小

**區域太多：** 提高截止閾值、增加合併距離，或增加過濾大小

**邊界有噪聲：** 使用 CCF 或 ML 方法代替 CC

**計算時間長：** 從 CC 方法開始以獲得快速結果，然後在需要時用 ML/HMM 改進
