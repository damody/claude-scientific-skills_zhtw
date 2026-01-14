# 使用 Arboreto 進行基本 GRN 推斷

## 輸入資料需求

Arboreto 需要以下兩種格式之一的基因表達資料：

### Pandas DataFrame（推薦）
- **列**：觀測值（細胞、樣本、條件）
- **欄**：基因（以基因名稱作為欄位標題）
- **格式**：數值表達值

範例：
```python
import pandas as pd

# 載入基因為欄位的表達矩陣
expression_matrix = pd.read_csv('expression_data.tsv', sep='\t')
# 欄位：['gene1', 'gene2', 'gene3', ...]
# 列：觀測資料
```

### NumPy 陣列
- **形狀**：(觀測值, 基因)
- **需求**：需另外提供與欄位順序相符的基因名稱清單

範例：
```python
import numpy as np

expression_matrix = np.genfromtxt('expression_data.tsv', delimiter='\t', skip_header=1)
with open('expression_data.tsv') as f:
    gene_names = [gene.strip() for gene in f.readline().split('\t')]

assert expression_matrix.shape[1] == len(gene_names)
```

## 轉錄因子（TF）

可選擇提供轉錄因子名稱清單以限制調控推斷：

```python
from arboreto.utils import load_tf_names

# 從檔案載入（每行一個 TF）
tf_names = load_tf_names('transcription_factors.txt')

# 或直接定義
tf_names = ['TF1', 'TF2', 'TF3']
```

如果未提供，所有基因都會被視為潛在調控者。

## 基本推斷工作流程

### 使用 Pandas DataFrame

```python
import pandas as pd
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2

if __name__ == '__main__':
    # 載入表達資料
    expression_matrix = pd.read_csv('expression_data.tsv', sep='\t')

    # 載入轉錄因子（可選）
    tf_names = load_tf_names('tf_list.txt')

    # 執行 GRN 推斷
    network = grnboost2(
        expression_data=expression_matrix,
        tf_names=tf_names  # 可選
    )

    # 儲存結果
    network.to_csv('network_output.tsv', sep='\t', index=False, header=False)
```

**重要**：`if __name__ == '__main__':` 保護是必要的，因為 Dask 會在內部產生新的行程。

### 使用 NumPy 陣列

```python
import numpy as np
from arboreto.algo import grnboost2

if __name__ == '__main__':
    # 載入表達矩陣
    expression_matrix = np.genfromtxt('expression_data.tsv', delimiter='\t', skip_header=1)

    # 從標題擷取基因名稱
    with open('expression_data.tsv') as f:
        gene_names = [gene.strip() for gene in f.readline().split('\t')]

    # 驗證維度相符
    assert expression_matrix.shape[1] == len(gene_names)

    # 使用明確的基因名稱執行推斷
    network = grnboost2(
        expression_data=expression_matrix,
        gene_names=gene_names,
        tf_names=tf_names
    )

    network.to_csv('network_output.tsv', sep='\t', index=False, header=False)
```

## 輸出格式

Arboreto 回傳包含三個欄位的 Pandas DataFrame：

| 欄位 | 說明 |
|--------|-------------|
| `TF` | 轉錄因子（調控者）基因名稱 |
| `target` | 標靶基因名稱 |
| `importance` | 調控重要性分數（越高 = 調控越強） |

輸出範例：
```
TF1    gene5    0.856
TF2    gene12   0.743
TF1    gene8    0.621
```

## 設定隨機種子

為獲得可重現的結果，請提供 seed 參數：

```python
network = grnboost2(
    expression_data=expression_matrix,
    tf_names=tf_names,
    seed=777
)
```

## 演算法選擇

大多數情況使用 `grnboost2()`（較快，可處理大型資料集）：
```python
from arboreto.algo import grnboost2
network = grnboost2(expression_data=expression_matrix)
```

用於比較或特定需求時使用 `genie3()`：
```python
from arboreto.algo import genie3
network = genie3(expression_data=expression_matrix)
```

詳細的演算法比較請參閱 `references/algorithms.md`。
