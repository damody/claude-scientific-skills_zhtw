# 時間序列分類

Aeon 提供 13 類時間序列分類器，具有與 scikit-learn 相容的 API。

## 基於卷積的分類器

應用隨機卷積轉換進行高效特徵擷取：

- `Arsenal` - ROCKET 分類器的集成，使用不同的核心
- `HydraClassifier` - 帶有膨脹的多解析度卷積
- `RocketClassifier` - 隨機卷積核心搭配嶺迴歸
- `MiniRocketClassifier` - 簡化的 ROCKET 變體，追求速度
- `MultiRocketClassifier` - 結合多個 ROCKET 變體

**使用時機**：需要快速、可擴展的分類，在各種資料集上具有強大效能。

## 深度學習分類器

針對時間序列優化的神經網路架構：

- `FCNClassifier` - 全卷積網路
- `ResNetClassifier` - 具有跳躍連接的殘差網路
- `InceptionTimeClassifier` - 多尺度 Inception 模組
- `TimeCNNClassifier` - 標準 CNN 用於時間序列
- `MLPClassifier` - 多層感知器基準
- `EncoderClassifier` - 通用編碼器包裝器
- `DisjointCNNClassifier` - 聚焦 Shapelet 的架構

**使用時機**：有大型資料集可用，需要端對端學習，或複雜的時間模式。

## 基於字典的分類器

將時間序列轉換為符號表示：

- `BOSSEnsemble` - SFA 符號袋搭配集成投票
- `TemporalDictionaryEnsemble` - 結合多種字典方法
- `WEASEL` - 時間序列分類的詞擷取
- `MrSEQLClassifier` - 多重符號序列學習

**使用時機**：需要可解釋模型、稀疏模式或符號推理。

## 基於距離的分類器

利用專門的時間序列距離度量：

- `KNeighborsTimeSeriesClassifier` - 使用時間距離的 k-NN（DTW、LCSS、ERP 等）
- `ElasticEnsemble` - 結合多種彈性距離測量
- `ProximityForest` - 使用基於距離分割的樹集成

**使用時機**：小型資料集，需要基於相似性的分類，或可解釋的決策。

## 基於特徵的分類器

在分類前擷取統計和簽名特徵：

- `Catch22Classifier` - 22 個典型的時間序列特徵
- `TSFreshClassifier` - 透過 tsfresh 自動擷取特徵
- `SignatureClassifier` - 路徑簽名轉換
- `SummaryClassifier` - 摘要統計擷取
- `FreshPRINCEClassifier` - 結合多個特徵擷取器

**使用時機**：需要可解釋特徵、有領域專業知識，或特徵工程方法。

## 基於區間的分類器

從隨機或監督區間擷取特徵：

- `CanonicalIntervalForestClassifier` - 搭配決策樹的隨機區間特徵
- `DrCIFClassifier` - 多樣表示 CIF 搭配 catch22 特徵
- `TimeSeriesForestClassifier` - 搭配摘要統計的隨機區間
- `RandomIntervalClassifier` - 簡單的基於區間方法
- `RandomIntervalSpectralEnsembleClassifier` - 來自區間的頻譜特徵
- `SupervisedTimeSeriesForest` - 監督區間選擇

**使用時機**：判別模式出現在特定時間視窗中。

## 基於 Shapelet 的分類器

識別判別性子序列（shapelets）：

- `ShapeletTransformClassifier` - 發現並使用判別性 shapelets
- `LearningShapeletClassifier` - 透過梯度下降學習 shapelets
- `SASTClassifier` - 可擴展的近似 shapelet 轉換
- `RDSTClassifier` - 隨機膨脹 shapelet 轉換

**使用時機**：需要可解釋的判別模式或相位不變特徵。

## 混合分類器

結合多種分類範式：

- `HIVECOTEV1` - 基於轉換的階層投票集成（第 1 版）
- `HIVECOTEV2` - 增強版，具有更新的元件

**使用時機**：需要最高準確度，有可用的計算資源。

## 早期分類

在觀察完整時間序列之前做出預測：

- `TEASER` - 兩層早期且準確的序列分類器
- `ProbabilityThresholdEarlyClassifier` - 當信心超過閾值時進行預測

**使用時機**：需要即時決策，或觀測有成本。

## 序數分類

處理有序類別標籤：

- `OrdinalTDE` - 用於序數輸出的時間字典集成

**使用時機**：類別具有自然順序（例如，嚴重程度等級）。

## 組合工具

建立自訂流程和集成：

- `ClassifierPipeline` - 將轉換器與分類器串聯
- `WeightedEnsembleClassifier` - 分類器的加權組合
- `SklearnClassifierWrapper` - 將 sklearn 分類器適配用於時間序列

## 快速開始

```python
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_classification

# 載入資料
X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

# 訓練並預測
clf = RocketClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

## 演算法選擇

- **速度優先**：MiniRocketClassifier、Arsenal
- **準確度優先**：HIVECOTEV2、InceptionTimeClassifier
- **可解釋性**：ShapeletTransformClassifier、Catch22Classifier
- **小型資料**：KNeighborsTimeSeriesClassifier、基於距離的方法
- **大型資料**：深度學習分類器、ROCKET 變體
