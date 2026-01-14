# 時間序列迴歸

Aeon 提供 9 類時間序列迴歸器，用於從時間序列預測連續值。

## 基於卷積的迴歸器

應用卷積核心進行特徵擷取：

- `HydraRegressor` - 多解析度膨脹卷積
- `RocketRegressor` - 隨機卷積核心
- `MiniRocketRegressor` - 簡化的 ROCKET，追求速度
- `MultiRocketRegressor` - 結合的 ROCKET 變體
- `MultiRocketHydraRegressor` - 合併 ROCKET 和 Hydra 方法

**使用時機**：需要快速迴歸與強大的基準效能。

## 深度學習迴歸器

用於端對端時間迴歸的神經架構：

- `FCNRegressor` - 全卷積網路
- `ResNetRegressor` - 具有跳躍連接的殘差區塊
- `InceptionTimeRegressor` - 多尺度 Inception 模組
- `TimeCNNRegressor` - 標準 CNN 架構
- `RecurrentRegressor` - RNN/LSTM/GRU 變體
- `MLPRegressor` - 多層感知器
- `EncoderRegressor` - 通用編碼器包裝器
- `LITERegressor` - 輕量 Inception Time 集成
- `DisjointCNNRegressor` - 專門的 CNN 架構

**使用時機**：大型資料集，複雜模式，或需要特徵學習。

## 基於距離的迴歸器

搭配時間距離度量的 k 最近鄰：

- `KNeighborsTimeSeriesRegressor` - 搭配 DTW、LCSS、ERP 或其他距離的 k-NN

**使用時機**：小型資料集，局部相似模式，或可解釋的預測。

## 基於特徵的迴歸器

在迴歸前擷取統計特徵：

- `Catch22Regressor` - 22 個典型的時間序列特徵
- `FreshPRINCERegressor` - 結合多個特徵擷取器的流程
- `SummaryRegressor` - 摘要統計特徵
- `TSFreshRegressor` - 自動化 tsfresh 特徵擷取

**使用時機**：需要可解釋特徵或領域特定的特徵工程。

## 混合迴歸器

結合多種方法：

- `RISTRegressor` - 隨機化區間-Shapelet 轉換

**使用時機**：受益於結合區間和 shapelet 方法。

## 基於區間的迴歸器

從時間區間擷取特徵：

- `CanonicalIntervalForestRegressor` - 搭配決策樹的隨機區間
- `DrCIFRegressor` - 多樣表示 CIF
- `TimeSeriesForestRegressor` - 隨機區間集成
- `RandomIntervalRegressor` - 簡單的基於區間方法
- `RandomIntervalSpectralEnsembleRegressor` - 頻譜區間特徵
- `QUANTRegressor` - 基於分位數的區間特徵

**使用時機**：預測模式出現在特定時間視窗中。

## 基於 Shapelet 的迴歸器

使用判別性子序列進行預測：

- `RDSTRegressor` - 隨機膨脹 Shapelet 轉換

**使用時機**：需要相位不變的判別模式。

## 組合工具

建立自訂迴歸流程：

- `RegressorPipeline` - 將轉換器與迴歸器串聯
- `RegressorEnsemble` - 帶有可學習權重的加權集成
- `SklearnRegressorWrapper` - 將 sklearn 迴歸器適配用於時間序列

## 工具

- `DummyRegressor` - 基準策略（平均值、中位數）
- `BaseRegressor` - 自訂迴歸器的抽象基類
- `BaseDeepRegressor` - 深度學習迴歸器的基類

## 快速開始

```python
from aeon.regression.convolution_based import RocketRegressor
from aeon.datasets import load_regression

# 載入資料
X_train, y_train = load_regression("Covid3Month", split="train")
X_test, y_test = load_regression("Covid3Month", split="test")

# 訓練並預測
reg = RocketRegressor()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

## 演算法選擇

- **速度優先**：MiniRocketRegressor
- **準確度優先**：InceptionTimeRegressor、MultiRocketHydraRegressor
- **可解釋性**：Catch22Regressor、SummaryRegressor
- **小型資料**：KNeighborsTimeSeriesRegressor
- **大型資料**：深度學習迴歸器、ROCKET 變體
- **區間模式**：DrCIFRegressor、CanonicalIntervalForestRegressor
