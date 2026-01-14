# 基因體標記器

標記器將基因體區域轉換為用於機器學習應用的離散標記（token），特別適用於訓練基因體深度學習模型。

## Python API

### 建立標記器

從各種來源載入標記器配置：

```python
import gtars

# 從 BED 檔案
tokenizer = gtars.tokenizers.TreeTokenizer.from_bed_file("regions.bed")

# 從配置檔案
tokenizer = gtars.tokenizers.TreeTokenizer.from_config("tokenizer_config.yaml")

# 從區域字串
tokenizer = gtars.tokenizers.TreeTokenizer.from_region_string("chr1:1000-2000")
```

### 標記化基因體區域

將基因體座標轉換為標記：

```python
# 標記化單一區域
token = tokenizer.tokenize("chr1", 1000, 2000)

# 標記化多個區域
tokens = []
for chrom, start, end in regions:
    token = tokenizer.tokenize(chrom, start, end)
    tokens.append(token)
```

### 標記屬性

存取標記資訊：

```python
# 取得標記 ID
token_id = token.id

# 取得基因體座標
chrom = token.chromosome
start = token.start
end = token.end

# 取得標記中繼資料
metadata = token.metadata
```

## 使用案例

### 機器學習預處理

標記器對於準備用於 ML 模型的基因體資料至關重要：

1. **序列建模**：將基因體區間轉換為用於 transformer 模型的離散標記
2. **位置編碼**：在資料集之間建立一致的位置編碼
3. **資料增強**：為訓練產生替代標記化

### 與 geniml 整合

標記器模組與 geniml 函式庫無縫整合用於基因體 ML：

```python
# 為 geniml 標記化區域
from gtars.tokenizers import TreeTokenizer
import geniml

tokenizer = TreeTokenizer.from_bed_file("training_regions.bed")
tokens = [tokenizer.tokenize(r.chrom, r.start, r.end) for r in regions]

# 在 geniml 模型中使用標記
model = geniml.Model(vocab_size=tokenizer.vocab_size)
```

## 配置格式

標記器配置檔案支援 YAML 格式：

```yaml
# tokenizer_config.yaml
type: tree
resolution: 1000  # 標記解析度（鹼基對）
chromosomes:
  - chr1
  - chr2
  - chr3
options:
  overlap_handling: merge
  gap_threshold: 100
```

## 效能考量

- TreeTokenizer 使用高效資料結構進行快速標記化
- 建議對大型資料集進行批次標記化
- 預載入標記器可減少重複操作的開銷
