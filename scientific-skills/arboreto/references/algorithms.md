# GRN 推斷演算法

Arboreto 提供兩種基因調控網路（GRN）推斷演算法，兩者都基於多重迴歸方法。

## 演算法概述

兩種演算法遵循相同的推斷策略：
1. 對資料集中的每個標靶基因，訓練一個迴歸模型
2. 從模型中識別最重要的特徵（潛在調控者）
3. 輸出這些特徵作為候選調控者及其重要性分數

關鍵差異在於**計算效率**和底層迴歸方法。

## GRNBoost2（推薦）

**用途**：使用梯度提升進行大規模資料集的快速 GRN 推斷。

### 使用時機
- **大型資料集**：數萬個觀測值（例如單細胞 RNA-seq）
- **時間受限分析**：需要比 GENIE3 更快的結果
- **預設選擇**：GRNBoost2 是旗艦演算法，推薦用於大多數使用案例

### 技術細節
- **方法**：具有早停正則化的隨機梯度提升
- **效能**：在大型資料集上顯著快於 GENIE3
- **輸出**：與 GENIE3 相同的格式（TF-標靶-重要性三元組）

### 使用方法
```python
from arboreto.algo import grnboost2

network = grnboost2(
    expression_data=expression_matrix,
    tf_names=tf_names,
    seed=42  # 用於可重現性
)
```

### 參數
```python
grnboost2(
    expression_data,           # 必要：pandas DataFrame 或 numpy 陣列
    gene_names=None,           # numpy 陣列時必要
    tf_names='all',            # TF 名稱清單或 'all'
    verbose=False,             # 印出進度訊息
    client_or_address='local', # Dask 客戶端或排程器位址
    seed=None                  # 用於可重現性的隨機種子
)
```

## GENIE3

**用途**：經典的隨機森林 GRN 推斷，作為概念藍圖。

### 使用時機
- **較小資料集**：當資料集大小允許較長的運算時間
- **比較研究**：與已發表的 GENIE3 結果進行比較
- **驗證**：驗證 GRNBoost2 結果

### 技術細節
- **方法**：隨機森林或極端樹迴歸
- **基礎**：原始的多重迴歸 GRN 推斷策略
- **權衡**：計算成本較高但方法成熟

### 使用方法
```python
from arboreto.algo import genie3

network = genie3(
    expression_data=expression_matrix,
    tf_names=tf_names,
    seed=42
)
```

### 參數
```python
genie3(
    expression_data,           # 必要：pandas DataFrame 或 numpy 陣列
    gene_names=None,           # numpy 陣列時必要
    tf_names='all',            # TF 名稱清單或 'all'
    verbose=False,             # 印出進度訊息
    client_or_address='local', # Dask 客戶端或排程器位址
    seed=None                  # 用於可重現性的隨機種子
)
```

## 演算法比較

| 特性 | GRNBoost2 | GENIE3 |
|---------|-----------|--------|
| **速度** | 快速（針對大型資料優化） | 較慢 |
| **方法** | 梯度提升 | 隨機森林 |
| **最適合** | 大規模資料（10k+ 觀測值） | 中小型資料集 |
| **輸出格式** | 相同 | 相同 |
| **推斷策略** | 多重迴歸 | 多重迴歸 |
| **推薦** | 是（預設選擇） | 用於比較/驗證 |

## 進階：自訂迴歸器參數

進階使用者可傳入自訂的 scikit-learn 迴歸器參數：

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# 自訂 GRNBoost2 參數
custom_grnboost2 = grnboost2(
    expression_data=expression_matrix,
    regressor_type='GBM',
    regressor_kwargs={
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    }
)

# 自訂 GENIE3 參數
custom_genie3 = genie3(
    expression_data=expression_matrix,
    regressor_type='RF',
    regressor_kwargs={
        'n_estimators': 1000,
        'max_features': 'sqrt'
    }
)
```

## 選擇正確的演算法

**決策指南**：

1. **從 GRNBoost2 開始** - 它更快且能更好地處理大型資料集
2. **使用 GENIE3 如果**：
   - 與現有 GENIE3 出版物進行比較
   - 資料集為中小型
   - 驗證 GRNBoost2 結果

兩種演算法產生具有相同輸出格式的可比較調控網路，使它們在大多數分析中可互換使用。
