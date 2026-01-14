# 時間序列聚類

Aeon 提供適配於時間資料的聚類演算法，具有專門的距離度量和平均方法。

## 分割演算法

適配於時間序列的標準 k-means/k-medoids：

- `TimeSeriesKMeans` - 搭配時間距離度量的 K-means（DTW、Euclidean 等）
- `TimeSeriesKMedoids` - 使用實際時間序列作為聚類中心
- `TimeSeriesKShape` - 基於形狀的聚類演算法
- `TimeSeriesKernelKMeans` - 用於非線性模式的核心變體

**使用時機**：已知聚類數量，預期為球形聚類形狀。

## 大型資料集方法

用於大型集合的高效聚類：

- `TimeSeriesCLARA` - 帶抽樣的大型應用聚類
- `TimeSeriesCLARANS` - CLARA 的隨機搜尋變體

**使用時機**：資料集太大而無法使用標準 k-medoids，需要可擴展性。

## 彈性距離聚類

專門用於基於對齊相似性的聚類：

- `KASBA` - 具有平移不變彈性平均的 K-means
- `ElasticSOM` - 使用彈性距離的自組織映射

**使用時機**：時間序列具有時間偏移或扭曲。

## 頻譜方法

基於圖的聚類：

- `KSpectralCentroid` - 帶有質心計算的頻譜聚類

**使用時機**：非凸聚類形狀，需要基於圖的方法。

## 深度學習聚類

使用自動編碼器的神經網路聚類：

- `AEFCNClusterer` - 全卷積自動編碼器
- `AEResNetClusterer` - 殘差網路自動編碼器
- `AEDCNNClusterer` - 膨脹 CNN 自動編碼器
- `AEDRNNClusterer` - 膨脹 RNN 自動編碼器
- `AEBiGRUClusterer` - 雙向 GRU 自動編碼器
- `AEAttentionBiGRUClusterer` - 注意力增強的 BiGRU 自動編碼器

**使用時機**：大型資料集，需要學習表示，或複雜模式。

## 基於特徵的聚類

在聚類前轉換到特徵空間：

- `Catch22Clusterer` - 在 22 個典型特徵上聚類
- `SummaryClusterer` - 使用摘要統計
- `TSFreshClusterer` - 自動化 tsfresh 特徵

**使用時機**：原始時間序列資訊不足，需要可解釋特徵。

## 組合

建立自訂聚類流程：

- `ClustererPipeline` - 將轉換器與聚類器串聯

## 平均方法

計算時間序列的聚類中心：

- `mean_average` - 算術平均
- `ba_average` - 搭配 DTW 的重心平均
- `kasba_average` - 平移不變平均
- `shift_invariant_average` - 通用平移不變方法

**使用時機**：需要代表性聚類中心用於視覺化或初始化。

## 快速開始

```python
from aeon.clustering import TimeSeriesKMeans
from aeon.datasets import load_classification

# 載入資料（使用分類資料進行聚類）
X_train, _ = load_classification("GunPoint", split="train")

# 聚類時間序列
clusterer = TimeSeriesKMeans(
    n_clusters=3,
    distance="dtw",  # 使用 DTW 距離
    averaging_method="ba"  # 重心平均
)
labels = clusterer.fit_predict(X_train)
centers = clusterer.cluster_centers_
```

## 演算法選擇

- **速度優先**：TimeSeriesKMeans 搭配 Euclidean 距離
- **時間對齊**：KASBA、TimeSeriesKMeans 搭配 DTW
- **大型資料集**：TimeSeriesCLARA、TimeSeriesCLARANS
- **複雜模式**：深度學習聚類器
- **可解釋性**：Catch22Clusterer、SummaryClusterer
- **非凸聚類**：KSpectralCentroid

## 距離度量

相容的距離度量包括：
- Euclidean、Manhattan、Minkowski（同步）
- DTW、DDTW、WDTW（具有對齊的彈性）
- ERP、EDR、LCSS（基於編輯）
- MSM、TWE（專門的彈性）

## 評估

使用 sklearn 或 aeon 基準測試的聚類指標：
- 輪廓係數
- Davies-Bouldin 指數
- Calinski-Harabasz 指數
