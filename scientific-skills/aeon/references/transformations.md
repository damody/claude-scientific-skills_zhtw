# 轉換

Aeon 提供廣泛的轉換功能，用於時間序列資料的預處理、特徵擷取和表示學習。

## 轉換類型

Aeon 區分：
- **CollectionTransformers**：轉換多個時間序列（集合）
- **SeriesTransformers**：轉換單個時間序列

## 集合轉換器

### 基於卷積的特徵擷取

使用隨機核心的快速、可擴展特徵生成：

- `RocketTransformer` - 隨機卷積核心
- `MiniRocketTransformer` - 簡化的 ROCKET，追求速度
- `MultiRocketTransformer` - 增強的 ROCKET 變體
- `HydraTransformer` - 多解析度膨脹卷積
- `MultiRocketHydraTransformer` - 結合 ROCKET 和 Hydra
- `ROCKETGPU` - GPU 加速變體

**使用時機**：需要快速、可擴展的特徵用於任何 ML 演算法，強大的基準效能。

### 統計特徵擷取

基於時間序列特徵的領域無關特徵：

- `Catch22` - 22 個典型的時間序列特徵
- `TSFresh` - 全面的自動特徵擷取（100+ 特徵）
- `TSFreshRelevant` - 帶有相關性過濾的特徵擷取
- `SevenNumberSummary` - 描述性統計（平均值、標準差、分位數）

**使用時機**：需要可解釋特徵、領域無關方法，或輸入傳統 ML。

### 基於字典的表示

用於離散表示的符號近似：

- `SAX` - 符號聚合近似
- `PAA` - 分段聚合近似
- `SFA` - 符號傅立葉近似
- `SFAFast` - 優化的 SFA
- `SFAWhole` - 整個序列上的 SFA（無視窗）
- `BORF` - 感受野袋

**使用時機**：需要離散/符號表示、降維、可解釋性。

### 基於 Shapelet 的特徵

判別性子序列擷取：

- `RandomShapeletTransform` - 隨機判別性 shapelets
- `RandomDilatedShapeletTransform` - 用於多尺度的膨脹 shapelets
- `SAST` - 可擴展且準確的子序列轉換
- `RSAST` - 隨機化 SAST

**使用時機**：需要可解釋的判別模式、相位不變特徵。

### 基於區間的特徵

來自時間區間的統計摘要：

- `RandomIntervals` - 來自隨機區間的特徵
- `SupervisedIntervals` - 監督區間選擇
- `QUANTTransformer` - 基於分位數的區間特徵

**使用時機**：預測模式局限於特定視窗。

### 預處理轉換

資料準備和標準化：

- `MinMaxScaler` - 縮放到 [0, 1] 範圍
- `Normalizer` - Z 標準化（零平均值、單位方差）
- `Centerer` - 中心化為零平均值
- `SimpleImputer` - 填充遺失值
- `DownsampleTransformer` - 降低時間解析度
- `Tabularizer` - 將時間序列轉換為表格格式

**使用時機**：需要標準化、遺失值處理、格式轉換。

### 專門轉換

進階分析方法：

- `MatrixProfile` - 計算用於模式發現的距離輪廓
- `DWTTransformer` - 離散小波變換
- `AutocorrelationFunctionTransformer` - ACF 計算
- `Dobin` - 使用鄰居的基於距離的離群值基礎
- `SignatureTransformer` - 路徑簽名方法
- `PLATransformer` - 分段線性近似

### 類別不平衡處理

- `ADASYN` - 自適應合成抽樣
- `SMOTE` - 合成少數過抽樣
- `OHIT` - 高度不平衡時間序列的過抽樣

**使用時機**：類別不平衡的分類。

### 流程組合

- `CollectionTransformerPipeline` - 串聯多個轉換器

## 序列轉換器

轉換單個時間序列（例如，用於預測中的預處理）。

### 統計分析

- `AutoCorrelationSeriesTransformer` - 自相關
- `StatsModelsACF` - 使用 statsmodels 的 ACF
- `StatsModelsPACF` - 偏自相關

### 平滑和過濾

- `ExponentialSmoothing` - 指數加權移動平均
- `MovingAverage` - 簡單或加權移動平均
- `SavitzkyGolayFilter` - 多項式平滑
- `GaussianFilter` - 高斯核平滑
- `BKFilter` - Baxter-King 帶通濾波器
- `DiscreteFourierApproximation` - 基於傅立葉的過濾

**使用時機**：需要降噪、趨勢擷取或頻率過濾。

### 降維

- `PCASeriesTransformer` - 主成分分析
- `PlASeriesTransformer` - 分段線性近似

### 變換

- `BoxCoxTransformer` - 方差穩定化
- `LogTransformer` - 對數縮放
- `ClaSPTransformer` - 分類分數輪廓

### 流程組合

- `SeriesTransformerPipeline` - 串聯序列轉換器

## 快速開始：特徵擷取

```python
from aeon.transformations.collection.convolution_based import RocketTransformer
from aeon.classification.sklearn import RotationForest
from aeon.datasets import load_classification

# 載入資料
X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

# 擷取 ROCKET 特徵
rocket = RocketTransformer()
X_train_features = rocket.fit_transform(X_train)
X_test_features = rocket.transform(X_test)

# 搭配任何 sklearn 分類器使用
clf = RotationForest()
clf.fit(X_train_features, y_train)
accuracy = clf.score(X_test_features, y_test)
```

## 快速開始：預處理流程

```python
from aeon.transformations.collection import (
    MinMaxScaler,
    SimpleImputer,
    CollectionTransformerPipeline
)

# 建立預處理流程
pipeline = CollectionTransformerPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

X_transformed = pipeline.fit_transform(X_train)
```

## 快速開始：序列平滑

```python
from aeon.transformations.series import MovingAverage

# 平滑單個時間序列
smoother = MovingAverage(window_size=5)
y_smoothed = smoother.fit_transform(y)
```

## 演算法選擇

### 特徵擷取：
- **速度 + 效能**：MiniRocketTransformer
- **可解釋性**：Catch22、TSFresh
- **降維**：PAA、SAX、PCA
- **判別模式**：Shapelet 轉換
- **全面特徵**：TSFresh（執行時間較長）

### 預處理：
- **標準化**：Normalizer、MinMaxScaler
- **平滑**：MovingAverage、SavitzkyGolayFilter
- **遺失值**：SimpleImputer
- **頻率分析**：DWTTransformer、傅立葉方法

### 符號表示：
- **快速近似**：PAA
- **基於字母**：SAX
- **基於頻率**：SFA、SFAFast

## 最佳實務

1. **僅在訓練資料上擬合**：避免資料洩漏
   ```python
   transformer.fit(X_train)
   X_train_tf = transformer.transform(X_train)
   X_test_tf = transformer.transform(X_test)
   ```

2. **流程組合**：串聯轉換器用於複雜工作流程
   ```python
   pipeline = CollectionTransformerPipeline([
       ('imputer', SimpleImputer()),
       ('scaler', Normalizer()),
       ('features', RocketTransformer())
   ])
   ```

3. **特徵選擇**：TSFresh 可生成許多特徵；考慮選擇
   ```python
   from sklearn.feature_selection import SelectKBest
   selector = SelectKBest(k=100)
   X_selected = selector.fit_transform(X_features, y)
   ```

4. **記憶體考量**：某些轉換器在大型資料集上記憶體密集
   - 使用 MiniRocket 而非 ROCKET 以提高速度
   - 考慮對非常長的序列進行降抽樣
   - 使用 ROCKETGPU 進行 GPU 加速

5. **領域知識**：選擇與領域匹配的轉換：
   - 週期性資料：基於傅立葉的方法
   - 雜訊資料：平滑濾波器
   - 尖峰檢測：小波變換
