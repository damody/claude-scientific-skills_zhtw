# 距離度量

Aeon 提供專門的距離函數用於測量時間序列之間的相似性，與 aeon 和 scikit-learn 估計器相容。

## 距離類別

### 彈性距離

允許序列之間靈活的時間對齊：

**動態時間扭曲系列：**
- `dtw` - 經典動態時間扭曲
- `ddtw` - 導數 DTW（比較導數）
- `wdtw` - 加權 DTW（按位置懲罰扭曲）
- `wddtw` - 加權導數 DTW
- `shape_dtw` - 基於形狀的 DTW

**基於編輯：**
- `erp` - 帶實數懲罰的編輯距離
- `edr` - 實數序列上的編輯距離
- `lcss` - 最長公共子序列
- `twe` - 時間扭曲編輯距離

**專門的：**
- `msm` - 移動-分割-合併距離
- `adtw` - 罰款 DTW
- `sbd` - 基於形狀的距離

**使用時機**：時間序列可能具有時間偏移、速度變化或相位差異。

### 同步距離

逐點比較時間序列，不進行對齊：

- `euclidean` - 歐幾里德距離（L2 範數）
- `manhattan` - 曼哈頓距離（L1 範數）
- `minkowski` - 廣義 Minkowski 距離（Lp 範數）
- `squared` - 平方歐幾里德距離

**使用時機**：序列已對齊，需要計算速度，或不預期時間扭曲。

## 使用模式

### 計算單一距離

```python
from aeon.distances import dtw_distance

# 兩個時間序列之間的距離
distance = dtw_distance(x, y)

# 帶視窗約束（Sakoe-Chiba 帶）
distance = dtw_distance(x, y, window=0.1)
```

### 成對距離矩陣

```python
from aeon.distances import dtw_pairwise_distance

# 集合中的所有成對距離
X = [series1, series2, series3, series4]
distance_matrix = dtw_pairwise_distance(X)

# 跨集合距離
distance_matrix = dtw_pairwise_distance(X_train, X_test)
```

### 成本矩陣和對齊路徑

```python
from aeon.distances import dtw_cost_matrix, dtw_alignment_path

# 取得完整成本矩陣
cost_matrix = dtw_cost_matrix(x, y)

# 取得最佳對齊路徑
path = dtw_alignment_path(x, y)
# 回傳索引：[(0,0), (1,1), (2,1), (2,2), ...]
```

### 搭配估計器使用

```python
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

# 在分類器中使用 DTW 距離
clf = KNeighborsTimeSeriesClassifier(
    n_neighbors=5,
    distance="dtw",
    distance_params={"window": 0.2}
)
clf.fit(X_train, y_train)
```

## 距離參數

### 視窗約束

限制扭曲路徑偏差（提高速度並防止病態扭曲）：

```python
# Sakoe-Chiba 帶：視窗作為序列長度的比例
dtw_distance(x, y, window=0.1)  # 允許 10% 偏差

# Itakura 平行四邊形：斜率約束路徑
dtw_distance(x, y, itakura_max_slope=2.0)
```

### 標準化

控制在計算距離前是否進行 z 標準化：

```python
# 大多數彈性距離支援標準化
distance = dtw_distance(x, y, normalize=True)
```

### 距離特定參數

```python
# ERP：間隙懲罰
distance = erp_distance(x, y, g=0.5)

# TWE：剛性和懲罰參數
distance = twe_distance(x, y, nu=0.001, lmbda=1.0)

# LCSS：匹配的 epsilon 閾值
distance = lcss_distance(x, y, epsilon=0.5)
```

## 演算法選擇

### 按使用案例：

**時間錯位**：DTW、DDTW、WDTW
**速度變化**：帶視窗約束的 DTW
**形狀相似性**：Shape DTW、SBD
**編輯操作**：ERP、EDR、LCSS
**導數匹配**：DDTW
**計算速度**：Euclidean、Manhattan
**離群值穩健性**：Manhattan、LCSS

### 按計算成本：

**最快**：Euclidean（O(n)）
**快速**：受約束的 DTW（O(nw)，其中 w 是視窗）
**中等**：完整 DTW（O(n²)）
**較慢**：複雜彈性距離（ERP、TWE、MSM）

## 快速參考表

| 距離 | 對齊 | 速度 | 穩健性 | 可解釋性 |
|----------|-----------|-------|------------|------------------|
| Euclidean | 同步 | 非常快 | 低 | 高 |
| DTW | 彈性 | 中等 | 中等 | 中等 |
| DDTW | 彈性 | 中等 | 高 | 中等 |
| WDTW | 彈性 | 中等 | 中等 | 中等 |
| ERP | 基於編輯 | 慢 | 高 | 低 |
| LCSS | 基於編輯 | 慢 | 非常高 | 低 |
| Shape DTW | 彈性 | 中等 | 中等 | 高 |

## 最佳實務

### 1. 標準化

大多數距離對尺度敏感；適當時進行標準化：

```python
from aeon.transformations.collection import Normalizer

normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)
```

### 2. 視窗約束

對於 DTW 變體，使用視窗約束以提高速度和更好的泛化：

```python
# 從 10-20% 視窗開始
distance = dtw_distance(x, y, window=0.1)
```

### 3. 序列長度

- 需要等長：大多數同步距離
- 支援不等長：彈性距離（DTW、ERP 等）

### 4. 多變量序列

大多數距離支援多變量時間序列：

```python
# x.shape = (n_channels, n_timepoints)
distance = dtw_distance(x_multivariate, y_multivariate)
```

### 5. 效能優化

- 使用 numba 編譯的實作（aeon 中的預設）
- 如果不需要對齊，考慮同步距離
- 使用視窗 DTW 而非完整 DTW
- 預先計算距離矩陣以供重複使用

### 6. 選擇正確的距離

```python
# 快速決策樹：
if series_aligned:
    use_distance = "euclidean"
elif need_speed:
    use_distance = "dtw"  # 帶視窗約束
elif temporal_shifts_expected:
    use_distance = "dtw" or "shape_dtw"
elif outliers_present:
    use_distance = "lcss" or "manhattan"
elif derivatives_matter:
    use_distance = "ddtw" or "wddtw"
```

## 與 scikit-learn 整合

Aeon 距離可與 sklearn 估計器配合使用：

```python
from sklearn.neighbors import KNeighborsClassifier
from aeon.distances import dtw_pairwise_distance

# 預先計算距離矩陣
X_train_distances = dtw_pairwise_distance(X_train)

# 搭配 sklearn 使用
clf = KNeighborsClassifier(metric='precomputed')
clf.fit(X_train_distances, y_train)
```

## 可用的距離函數

取得所有可用距離的列表：

```python
from aeon.distances import get_distance_function_names

print(get_distance_function_names())
# ['dtw', 'ddtw', 'wdtw', 'euclidean', 'erp', 'edr', ...]
```

擷取特定距離函數：

```python
from aeon.distances import get_distance_function

distance_func = get_distance_function("dtw")
result = distance_func(x, y, window=0.1)
```
