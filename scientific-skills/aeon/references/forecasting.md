# 時間序列預測

Aeon 提供預測演算法用於預測未來的時間序列值。

## 樸素和基準方法

用於比較的簡單預測策略：

- `NaiveForecaster` - 多種策略：最後值、平均值、季節樸素
  - 參數：`strategy`（"last"、"mean"、"seasonal"）、`sp`（季節週期）
  - **使用時機**：建立基準或簡單模式

## 統計模型

經典時間序列預測方法：

### ARIMA
- `ARIMA` - 自迴歸整合移動平均
  - 參數：`p`（AR 階數）、`d`（差分）、`q`（MA 階數）
  - **使用時機**：線性模式、平穩或差分平穩序列

### 指數平滑
- `ETS` - 誤差-趨勢-季節分解
  - 參數：`error`、`trend`、`seasonal` 類型
  - **使用時機**：存在趨勢和季節模式

### 門檻自迴歸
- `TAR` - 用於體制轉換的門檻自迴歸模型
- `AutoTAR` - 自動門檻發現
  - **使用時機**：序列在不同體制下表現不同行為

### Theta 方法
- `Theta` - 經典 Theta 預測
  - 參數：`theta`、用於分解的 `weights`
  - **使用時機**：需要簡單但有效的基準

### 時變參數
- `TVP` - 使用 Kalman 濾波的時變參數模型
  - **使用時機**：參數隨時間變化

## 深度學習預測器

用於複雜時間模式的神經網路：

- `TCNForecaster` - 時間卷積網路
  - 膨脹卷積用於大感受野
  - **使用時機**：長序列，需要非遞迴架構

- `DeepARNetwork` - 使用 RNN 的機率預測
  - 提供預測區間
  - **使用時機**：需要機率預測、不確定性量化

## 基於迴歸的預測

對延遲特徵應用迴歸：

- `RegressionForecaster` - 將迴歸器包裝用於預測
  - 參數：`window_length`、`horizon`
  - **使用時機**：想使用任何迴歸器作為預測器

## 快速開始

```python
from aeon.forecasting.naive import NaiveForecaster
from aeon.forecasting.arima import ARIMA
import numpy as np

# 建立時間序列
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 樸素基準
naive = NaiveForecaster(strategy="last")
naive.fit(y)
forecast_naive = naive.predict(fh=[1, 2, 3])

# ARIMA 模型
arima = ARIMA(order=(1, 1, 1))
arima.fit(y)
forecast_arima = arima.predict(fh=[1, 2, 3])
```

## 預測時間範圍

預測時間範圍（`fh`）指定要預測的未來時間點：

```python
# 相對時間範圍（接下來的 3 步）
fh = [1, 2, 3]

# 絕對時間範圍（特定時間索引）
from aeon.forecasting.base import ForecastingHorizon
fh = ForecastingHorizon([11, 12, 13], is_relative=False)
```

## 模型選擇

- **基準**：NaiveForecaster 搭配季節策略
- **線性模式**：ARIMA
- **趨勢 + 季節性**：ETS
- **體制變化**：TAR、AutoTAR
- **複雜模式**：TCNForecaster
- **機率性**：DeepARNetwork
- **長序列**：TCNForecaster
- **短序列**：ARIMA、ETS

## 評估指標

使用標準預測指標：

```python
from aeon.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# 計算誤差
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
```

## 外生變數

許多預測器支援外生特徵：

```python
# 使用外生變數訓練
forecaster.fit(y, X=X_train)

# 預測需要未來的外生值
y_pred = forecaster.predict(fh=[1, 2, 3], X=X_test)
```

## 基類

- `BaseForecaster` - 所有預測器的抽象基類
- `BaseDeepForecaster` - 深度學習預測器的基類

擴展這些以實作自訂預測演算法。
