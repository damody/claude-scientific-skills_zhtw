# 相似性搜尋

Aeon 提供工具用於在時間序列內部或跨時間序列尋找相似模式，包括子序列搜尋、motif 發現和近似最近鄰。

## 子序列最近鄰（SNN）

在時間序列中尋找最相似的子序列。

### MASS 演算法
- `MassSNN` - Mueen 的相似性搜尋演算法
  - 快速標準化互相關用於相似性
  - 高效計算距離輪廓
  - **使用時機**：需要精確的最近鄰距離，大型序列

### 基於 STOMP 的 Motif 發現
- `StompMotif` - 發現重複模式（motifs）
  - 尋找前 k 個最相似的子序列對
  - 基於矩陣輪廓計算
  - **使用時機**：想發現重複模式

### 暴力基準
- `DummySNN` - 窮舉距離計算
  - 計算所有成對距離
  - **使用時機**：小型序列，需要精確基準

## 集合層級搜尋

在集合中尋找相似的時間序列。

### 近似最近鄰（ANN）
- `RandomProjectionIndexANN` - 局部敏感雜湊
  - 使用帶有餘弦相似度的隨機投影
  - 建立索引用於快速近似搜尋
  - **使用時機**：大型集合，速度比精確度更重要

## 快速開始：Motif 發現

```python
from aeon.similarity_search import StompMotif
import numpy as np

# 建立具有重複模式的時間序列
pattern = np.sin(np.linspace(0, 2*np.pi, 50))
y = np.concatenate([
    pattern + np.random.normal(0, 0.1, 50),
    np.random.normal(0, 1, 100),
    pattern + np.random.normal(0, 0.1, 50),
    np.random.normal(0, 1, 100)
])

# 尋找前 3 個 motifs
motif_finder = StompMotif(window_size=50, k=3)
motifs = motif_finder.fit_predict(y)

# motifs 包含 motif 出現的索引
for i, (idx1, idx2) in enumerate(motifs):
    print(f"Motif {i+1} 在位置 {idx1} 和 {idx2}")
```

## 快速開始：子序列搜尋

```python
from aeon.similarity_search import MassSNN
import numpy as np

# 要搜尋的時間序列
y = np.sin(np.linspace(0, 20, 500))

# 查詢子序列
query = np.sin(np.linspace(0, 2, 50))

# 尋找最近的子序列
searcher = MassSNN()
distances = searcher.fit_transform(y, query)

# 尋找最佳匹配
best_match_idx = np.argmin(distances)
print(f"最佳匹配在索引 {best_match_idx}")
```

## 快速開始：集合上的近似 NN

```python
from aeon.similarity_search import RandomProjectionIndexANN
from aeon.datasets import load_classification

# 載入時間序列集合
X_train, _ = load_classification("GunPoint", split="train")

# 建立索引
ann = RandomProjectionIndexANN(n_projections=8, n_bits=4)
ann.fit(X_train)

# 尋找近似最近鄰
query = X_train[0]
neighbors, distances = ann.kneighbors(query, k=5)
```

## 矩陣輪廓

矩陣輪廓是許多相似性搜尋任務的基本資料結構：

- **距離輪廓**：從查詢到所有子序列的距離
- **矩陣輪廓**：每個子序列到任何其他子序列的最小距離
- **Motif**：具有最小距離的子序列對
- **Discord**：具有最大最小距離的子序列（異常）

```python
from aeon.similarity_search import StompMotif

# 計算矩陣輪廓並尋找 motifs/discords
mp = StompMotif(window_size=50)
mp.fit(y)

# 存取矩陣輪廓
profile = mp.matrix_profile_
profile_indices = mp.matrix_profile_index_

# 尋找 discords（異常）
discord_idx = np.argmax(profile)
```

## 演算法選擇

- **精確子序列搜尋**：MassSNN
- **Motif 發現**：StompMotif
- **異常檢測**：矩陣輪廓（請參閱 anomaly_detection.md）
- **快速近似搜尋**：RandomProjectionIndexANN
- **小型資料**：DummySNN 用於精確結果

## 使用案例

### 模式匹配
找到長序列中模式出現的位置：

```python
# 在 ECG 資料中尋找心跳模式
searcher = MassSNN()
distances = searcher.fit_transform(ecg_data, heartbeat_pattern)
occurrences = np.where(distances < threshold)[0]
```

### Motif 發現
識別重複模式：

```python
# 尋找重複的行為模式
motif_finder = StompMotif(window_size=100, k=5)
motifs = motif_finder.fit_predict(activity_data)
```

### 時間序列檢索
在資料庫中尋找相似的時間序列：

```python
# 建立可搜尋的索引
ann = RandomProjectionIndexANN()
ann.fit(time_series_database)

# 查詢相似序列
neighbors = ann.kneighbors(query_series, k=10)
```

## 最佳實務

1. **視窗大小**：子序列方法的關鍵參數
   - 太小：捕捉雜訊
   - 太大：錯過細粒度模式
   - 經驗法則：序列長度的 10-20%

2. **標準化**：大多數方法假設 z 標準化的子序列
   - 處理振幅變化
   - 專注於形狀相似性

3. **距離度量**：不同需求使用不同度量
   - Euclidean：快速，基於形狀
   - DTW：處理時間扭曲
   - Cosine：尺度不變

4. **排除區**：對於 motif 發現，排除平凡匹配
   - 通常設為 0.5-1.0 × 視窗大小
   - 防止找到重疊的出現

5. **效能**：
   - MASS 是 O(n log n) vs O(n²) 暴力
   - ANN 以準確度換取速度
   - 某些方法有 GPU 加速
