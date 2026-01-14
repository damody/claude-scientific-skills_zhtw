# 異常檢測

Aeon 提供異常檢測方法，用於在序列和集合層級識別時間序列中的不尋常模式。

## 集合異常檢測器

檢測集合中的異常時間序列：

- `ClassificationAdapter` - 將分類器適配用於異常檢測
  - 在正常資料上訓練，在預測時標記離群值
  - **使用時機**：有標記的正常資料，需要基於分類的方法

- `OutlierDetectionAdapter` - 包裝 sklearn 離群值檢測器
  - 可與 IsolationForest、LOF、OneClassSVM 配合使用
  - **使用時機**：想在集合上使用 sklearn 異常檢測器

## 序列異常檢測器

檢測單一時間序列中的異常點或子序列。

### 基於距離的方法

使用相似性度量來識別異常：

- `CBLOF` - 基於聚類的局部離群因子
  - 對資料進行聚類，基於聚類屬性識別離群值
  - **使用時機**：異常形成稀疏聚類

- `KMeansAD` - 基於 K-means 的異常檢測
  - 到最近聚類中心的距離表示異常
  - **使用時機**：正常模式能良好聚類

- `LeftSTAMPi` - 左側 STAMP 增量方法
  - 用於線上異常檢測的矩陣輪廓
  - **使用時機**：串流資料，需要線上檢測

- `STOMP` - 可擴展時間序列有序搜尋矩陣輪廓
  - 計算子序列異常的矩陣輪廓
  - **使用時機**：Discord 發現、motif 檢測

- `MERLIN` - 基於矩陣輪廓的方法
  - 高效的矩陣輪廓計算
  - **使用時機**：大型時間序列，需要可擴展性

- `LOF` - 適配於時間序列的局部離群因子
  - 基於密度的離群值檢測
  - **使用時機**：低密度區域中的異常

- `ROCKAD` - 基於 ROCKET 的半監督檢測
  - 使用 ROCKET 特徵進行異常識別
  - **使用時機**：有部分標記資料，需要基於特徵的方法

### 基於分布的方法

分析統計分布：

- `COPOD` - 基於 Copula 的離群值檢測
  - 建模邊際和聯合分布
  - **使用時機**：多維時間序列，複雜的依賴關係

- `DWT_MLEAD` - 離散小波變換多層異常檢測
  - 將序列分解為頻率帶
  - **使用時機**：特定頻率的異常

### 基於隔離的方法

使用隔離原理：

- `IsolationForest` - 基於隨機森林的隔離
  - 異常比正常點更容易被隔離
  - **使用時機**：高維資料，對分布無假設

- `OneClassSVM` - 用於新奇檢測的支援向量機
  - 學習正常資料周圍的邊界
  - **使用時機**：定義明確的正常區域，需要穩健的邊界

- `STRAY` - 串流穩健異常檢測
  - 對資料分布變化穩健
  - **使用時機**：串流資料，分布漂移

### 外部函式庫整合

- `PyODAdapter` - 將 PyOD 函式庫橋接到 aeon
  - 存取 40+ PyOD 異常檢測器
  - **使用時機**：需要特定的 PyOD 演算法

## 快速開始

```python
from aeon.anomaly_detection import STOMP
import numpy as np

# 建立含有異常的時間序列
y = np.concatenate([
    np.sin(np.linspace(0, 10, 100)),
    [5.0],  # 異常尖峰
    np.sin(np.linspace(10, 20, 100))
])

# 檢測異常
detector = STOMP(window_size=10)
anomaly_scores = detector.fit_predict(y)

# 較高的分數表示更異常的點
threshold = np.percentile(anomaly_scores, 95)
anomalies = anomaly_scores > threshold
```

## 點異常 vs 子序列異常

- **點異常**：單一不尋常的值
  - 使用：COPOD、DWT_MLEAD、IsolationForest

- **子序列異常**（discords）：不尋常的模式
  - 使用：STOMP、LeftSTAMPi、MERLIN

- **集體異常**：形成不尋常模式的點群
  - 使用：矩陣輪廓方法、基於聚類的方法

## 評估指標

異常檢測的專門指標：

```python
from aeon.benchmarking.metrics.anomaly_detection import (
    range_precision,
    range_recall,
    range_f_score,
    roc_auc_score
)

# 基於範圍的指標考慮視窗檢測
precision = range_precision(y_true, y_pred, alpha=0.5)
recall = range_recall(y_true, y_pred, alpha=0.5)
f1 = range_f_score(y_true, y_pred, alpha=0.5)
```

## 演算法選擇

- **速度優先**：KMeansAD、IsolationForest
- **準確度優先**：STOMP、COPOD
- **串流資料**：LeftSTAMPi、STRAY
- **Discord 發現**：STOMP、MERLIN
- **多維**：COPOD、PyODAdapter
- **半監督**：ROCKAD、OneClassSVM
- **無訓練資料**：IsolationForest、STOMP

## 最佳實務

1. **標準化資料**：許多方法對尺度敏感
2. **選擇視窗大小**：對矩陣輪廓方法，視窗大小很關鍵
3. **設定閾值**：使用百分位數或領域特定的閾值
4. **驗證結果**：視覺化檢測以驗證有意義性
5. **處理季節性**：檢測前進行去趨勢/去季節化
