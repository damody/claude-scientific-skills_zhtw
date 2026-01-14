# 時間序列分析參考

本文件提供 statsmodels 中時間序列模型的完整指引，包括 ARIMA、狀態空間模型、VAR、指數平滑和預測方法。

## 概述

Statsmodels 提供廣泛的時間序列功能：
- **單變量模型**：AR、ARIMA、SARIMAX、指數平滑
- **多變量模型**：VAR、VARMAX、動態因子模型
- **狀態空間框架**：自訂模型、Kalman 濾波
- **診斷工具**：ACF、PACF、定態性檢定、殘差分析
- **預測**：點預測和預測區間

## 單變量時間序列模型

### AutoReg（AR 模型）

自迴歸模型：當前值取決於過去值。

**使用時機：**
- 單變量時間序列
- 過去值預測未來
- 定態序列

**模型**：yₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ + εₜ

```python
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd

# 配適 AR(p) 模型
model = AutoReg(y, lags=5)  # AR(5)
results = model.fit()

print(results.summary())
```

**帶外生回歸變數：**
```python
# 帶外生變數的 AR（ARX）
model = AutoReg(y, lags=5, exog=X_exog)
results = model.fit()
```

**季節性 AR：**
```python
# 季節性滯後（例如，具有年度季節性的月資料）
model = AutoReg(y, lags=12, seasonal=True)
results = model.fit()
```

### ARIMA（自迴歸整合移動平均）

結合 AR、差分（I）和 MA 成分。

**使用時機：**
- 非定態時間序列（需要差分）
- 過去值和誤差預測未來
- 許多時間序列的彈性模型

**模型**：ARIMA(p,d,q)
- p：AR 階數（滯後）
- d：差分階數（達到定態）
- q：MA 階數（滯後預測誤差）

```python
from statsmodels.tsa.arima.model import ARIMA

# 配適 ARIMA(p,d,q)
model = ARIMA(y, order=(1, 1, 1))  # ARIMA(1,1,1)
results = model.fit()

print(results.summary())
```

**選擇 p, d, q：**

1. **決定 d（差分階數）**：
```python
from statsmodels.tsa.stattools import adfuller

# 定態性的 ADF 檢定
def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("Series is stationary")
        return True
    else:
        print("Series is non-stationary, needs differencing")
        return False

# 檢定原始序列
if not check_stationarity(y):
    # 差分一次
    y_diff = y.diff().dropna()
    if not check_stationarity(y_diff):
        # 再差分
        y_diff2 = y_diff.diff().dropna()
        check_stationarity(y_diff2)
```

2. **決定 p 和 q（ACF/PACF）**：
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# 差分至定態後
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ACF：幫助決定 q（MA 階數）
plot_acf(y_stationary, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')

# PACF：幫助決定 p（AR 階數）
plot_pacf(y_stationary, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# 經驗法則：
# - PACF 在滯後 p 截斷 → AR(p)
# - ACF 在滯後 q 截斷 → MA(q)
# - 兩者都衰減 → ARMA(p,q)
```

3. **模型選擇（AIC/BIC）**：
```python
# 給定 d 對最佳 (p,q) 進行網格搜尋
import numpy as np

best_aic = np.inf
best_order = None

for p in range(5):
    for q in range(5):
        try:
            model = ARIMA(y, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except:
            continue

print(f"Best order: {best_order} with AIC: {best_aic:.2f}")
```

### SARIMAX（帶外生變數的季節性 ARIMA）

將 ARIMA 擴展為含季節性和外生回歸變數。

**使用時機：**
- 季節性模式（月度、季度資料）
- 外部變數影響序列
- 最彈性的單變量模型

**模型**：SARIMAX(p,d,q)(P,D,Q,s)
- (p,d,q)：非季節性 ARIMA
- (P,D,Q,s)：週期為 s 的季節性 ARIMA

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 月資料的季節性 ARIMA（s=12）
model = SARIMAX(y,
                order=(1, 1, 1),           # (p,d,q)
                seasonal_order=(1, 1, 1, 12))  # (P,D,Q,s)
results = model.fit()

print(results.summary())
```

**帶外生變數：**
```python
# 帶外部預測變數的 SARIMAX
model = SARIMAX(y,
                exog=X_exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
results = model.fit()
```

**範例：含趨勢和季節性的月銷售額**
```python
# 月資料的典型：(p,d,q)(P,D,Q,12)
# 從 (1,1,1)(1,1,1,12) 或 (0,1,1)(0,1,1,12) 開始

model = SARIMAX(monthly_sales,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()
```

### 指數平滑

以指數遞減權重對過去觀測值的加權平均。

**使用時機：**
- 簡單、可解釋的預測
- 存在趨勢和/或季節性
- 無需明確模型規格

**類型：**
- 簡單指數平滑：無趨勢、無季節性
- Holt 方法：有趨勢
- Holt-Winters：有趨勢和季節性

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 簡單指數平滑
model = ExponentialSmoothing(y, trend=None, seasonal=None)
results = model.fit()

# Holt 方法（有趨勢）
model = ExponentialSmoothing(y, trend='add', seasonal=None)
results = model.fit()

# Holt-Winters（趨勢 + 季節性）
model = ExponentialSmoothing(y,
                            trend='add',           # 'add' 或 'mul'
                            seasonal='add',        # 'add' 或 'mul'
                            seasonal_periods=12)   # 例如，月資料為 12
results = model.fit()

print(results.summary())
```

**加法 vs 乘法：**
```python
# 加法：恆定季節變動
# yₜ = Level + Trend + Seasonal + Error

# 乘法：比例季節變動
# yₜ = Level × Trend × Seasonal × Error

# 根據資料選擇：
# - 加法：季節變動隨時間恆定
# - 乘法：季節變動隨水準增加
```

**創新狀態空間（ETS）：**
```python
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# 更穩健的狀態空間公式
model = ETSModel(y,
                error='add',           # 'add' 或 'mul'
                trend='add',           # 'add'、'mul' 或 None
                seasonal='add',        # 'add'、'mul' 或 None
                seasonal_periods=12)
results = model.fit()
```

## 多變量時間序列

### VAR（向量自迴歸）

方程組，其中每個變數取決於所有變數的過去值。

**使用時機：**
- 多個相關時間序列
- 雙向關係
- Granger 因果檢定

**模型**：每個變數對所有變數進行 AR：
- y₁ₜ = c₁ + φ₁₁y₁ₜ₋₁ + φ₁₂y₂ₜ₋₁ + ... + ε₁ₜ
- y₂ₜ = c₂ + φ₂₁y₁ₜ₋₁ + φ₂₂y₂ₜ₋₁ + ... + ε₂ₜ

```python
from statsmodels.tsa.api import VAR
import pandas as pd

# 資料應為含多列的 DataFrame
# 每列是一個時間序列
df_multivariate = pd.DataFrame({'series1': y1, 'series2': y2, 'series3': y3})

# 配適 VAR
model = VAR(df_multivariate)

# 使用 AIC/BIC 選擇滯後階數
lag_order_results = model.select_order(maxlags=15)
print(lag_order_results.summary())

# 以最優滯後配適
results = model.fit(maxlags=5, ic='aic')
print(results.summary())
```

**Granger 因果檢定：**
```python
# 檢定 series1 是否 Granger 導致 series2
from statsmodels.tsa.stattools import grangercausalitytests

# 需要 2D 陣列 [series2, series1]
test_data = df_multivariate[['series2', 'series1']]

# 檢定直到 max_lag
max_lag = 5
results = grangercausalitytests(test_data, max_lag, verbose=True)

# 每個滯後的 P 值
for lag in range(1, max_lag + 1):
    p_value = results[lag][0]['ssr_ftest'][1]
    print(f"Lag {lag}: p-value = {p_value:.4f}")
```

**衝擊反應函數（IRF）：**
```python
# 追蹤衝擊通過系統的效果
irf = results.irf(10)  # 向前 10 期

# 繪製 IRF
irf.plot(orth=True)  # 正交化（Cholesky 分解）
plt.show()

# 累積效果
irf.plot_cum_effects(orth=True)
plt.show()
```

**預測誤差變異數分解：**
```python
# 每個變數對預測誤差變異數的貢獻
fevd = results.fevd(10)  # 向前 10 期
fevd.plot()
plt.show()
```

### VARMAX（帶移動平均和外生變數的 VAR）

以 MA 成分和外部回歸變數擴展 VAR。

**使用時機：**
- VAR 不足（需要 MA 成分）
- 外部變數影響系統
- 更彈性的多變量模型

```python
from statsmodels.tsa.statespace.varmax import VARMAX

# 帶外生變數的 VARMAX(p, q)
model = VARMAX(df_multivariate,
               order=(1, 1),        # (p, q)
               exog=X_exog)
results = model.fit()

print(results.summary())
```

## 狀態空間模型

用於自訂時間序列模型的彈性框架。

**使用時機：**
- 自訂模型規格
- 不可觀測成分
- Kalman 濾波/平滑
- 缺失資料

```python
from statsmodels.tsa.statespace.mlemodel import MLEModel

# 擴展 MLEModel 用於自訂狀態空間模型
# 範例：局部水準模型（隨機遊走 + 雜訊）
```

**動態因子模型：**
```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# 從多個時間序列萃取共同因子
model = DynamicFactor(df_multivariate,
                      k_factors=2,          # 因子數
                      factor_order=2)       # 因子的 AR 階數
results = model.fit()

# 估計因子
factors = results.factors.filtered
```

## 預測

### 點預測

```python
# ARIMA 預測
model = ARIMA(y, order=(1, 1, 1))
results = model.fit()

# 向前預測 h 步
h = 10
forecast = results.forecast(steps=h)

# 帶外生變數（SARIMAX）
model = SARIMAX(y, exog=X, order=(1, 1, 1))
results = model.fit()

# 需要未來外生值
forecast = results.forecast(steps=h, exog=X_future)
```

### 預測區間

```python
# 取得帶信賴區間的預測
forecast_obj = results.get_forecast(steps=h)
forecast_df = forecast_obj.summary_frame()

print(forecast_df)
# 包含：mean、mean_se、mean_ci_lower、mean_ci_upper

# 萃取成分
forecast_mean = forecast_df['mean']
forecast_ci_lower = forecast_df['mean_ci_lower']
forecast_ci_upper = forecast_df['mean_ci_upper']

# 繪圖
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label='Historical')
plt.plot(forecast_df.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_df.index,
                 forecast_ci_lower,
                 forecast_ci_upper,
                 alpha=0.3, color='red', label='95% CI')
plt.legend()
plt.title('Forecast with Prediction Intervals')
plt.show()
```

### 動態 vs 靜態預測

```python
# 靜態（一步向前，使用實際值）
static_forecast = results.get_prediction(start=split_point, end=len(y)-1)

# 動態（多步，使用預測值）
dynamic_forecast = results.get_prediction(start=split_point,
                                          end=len(y)-1,
                                          dynamic=True)

# 繪製比較
fig, ax = plt.subplots(figsize=(12, 6))
y.plot(ax=ax, label='Actual')
static_forecast.predicted_mean.plot(ax=ax, label='Static forecast')
dynamic_forecast.predicted_mean.plot(ax=ax, label='Dynamic forecast')
ax.legend()
plt.show()
```

## 診斷檢定

### 定態性檢定

```python
from statsmodels.tsa.stattools import adfuller, kpss

# 增廣 Dickey-Fuller（ADF）檢定
# H0：單位根（非定態）
adf_result = adfuller(y, autolag='AIC')
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] <= 0.05:
    print("Reject H0: Series is stationary")
else:
    print("Fail to reject H0: Series is non-stationary")

# KPSS 檢定
# H0：定態（與 ADF 相反）
kpss_result = kpss(y, regression='c', nlags='auto')
print(f"KPSS Statistic: {kpss_result[0]:.4f}")
print(f"p-value: {kpss_result[1]:.4f}")
if kpss_result[1] <= 0.05:
    print("Reject H0: Series is non-stationary")
else:
    print("Fail to reject H0: Series is stationary")
```

### 殘差診斷

```python
# 殘差自相關的 Ljung-Box 檢定
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(results.resid, lags=10, return_df=True)
print(lb_test)
# P 值 > 0.05 表示無顯著自相關（好）

# 繪製殘差診斷
results.plot_diagnostics(figsize=(12, 8))
plt.show()

# 成分：
# 1. 隨時間的標準化殘差
# 2. 殘差的直方圖 + KDE
# 3. 常態性的 Q-Q 圖
# 4. 相關圖（殘差的 ACF）
```

### 異質變異數檢定

```python
from statsmodels.stats.diagnostic import het_arch

# 異質變異數的 ARCH 檢定
arch_test = het_arch(results.resid, nlags=10)
print(f"ARCH test statistic: {arch_test[0]:.4f}")
print(f"p-value: {arch_test[1]:.4f}")

# 若顯著，考慮 GARCH 模型
```

## 季節性分解

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# 分解為趨勢、季節、殘差
decomposition = seasonal_decompose(y,
                                   model='additive',  # 或 'multiplicative'
                                   period=12)         # 季節週期

# 繪製成分
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# 存取成分
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# STL 分解（更穩健）
from statsmodels.tsa.seasonal import STL

stl = STL(y, seasonal=13)  # seasonal 必須為奇數
stl_result = stl.fit()

fig = stl_result.plot()
plt.show()
```

## 模型評估

### 樣本內指標

```python
# 從 results 物件
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")
print(f"Log-likelihood: {results.llf:.2f}")

# 訓練資料上的 MSE
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, results.fittedvalues)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, results.fittedvalues)
print(f"MAE: {mae:.4f}")
```

### 樣本外評估

```python
# 時間序列的訓練-測試分割（不洗牌！）
train_size = int(0.8 * len(y))
y_train = y[:train_size]
y_test = y[train_size:]

# 在訓練資料上配適
model = ARIMA(y_train, order=(1, 1, 1))
results = model.fit()

# 預測測試期間
forecast = results.forecast(steps=len(y_test))

# 指標
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_test, forecast))
mae = mean_absolute_error(y_test, forecast)
mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MAPE: {mape:.2f}%")
```

### 滾動預測

```python
# 更現實的評估：滾動一步向前預測
forecasts = []

for t in range(len(y_test)):
    # 以新觀測值重新配適或更新
    y_current = y[:train_size + t]
    model = ARIMA(y_current, order=(1, 1, 1))
    fit = model.fit()

    # 一步預測
    fc = fit.forecast(steps=1)[0]
    forecasts.append(fc)

forecasts = np.array(forecasts)

rmse = np.sqrt(mean_squared_error(y_test, forecasts))
print(f"Rolling forecast RMSE: {rmse:.4f}")
```

### 交叉驗證

```python
# 時間序列交叉驗證（擴展視窗）
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

for train_idx, test_idx in tscv.split(y):
    y_train_cv = y.iloc[train_idx]
    y_test_cv = y.iloc[test_idx]

    model = ARIMA(y_train_cv, order=(1, 1, 1))
    results = model.fit()

    forecast = results.forecast(steps=len(test_idx))
    rmse = np.sqrt(mean_squared_error(y_test_cv, forecast))
    rmse_scores.append(rmse)

print(f"CV RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
```

## 進階主題

### ARDL（自迴歸分布滯後）

連接單變量和多變量時間序列。

```python
from statsmodels.tsa.ardl import ARDL

# ARDL(p, q) 模型
# y 取決於自己的滯後和 X 的滯後
model = ARDL(y, lags=2, exog=X, exog_lags=2)
results = model.fit()
```

### 誤差修正模型

用於共整合序列。

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# 共整合檢定
johansen_test = coint_johansen(df_multivariate, det_order=0, k_ar_diff=1)

# 若共整合則配適 VECM
from statsmodels.tsa.vector_ar.vecm import VECM

model = VECM(df_multivariate, k_ar_diff=1, coint_rank=1)
results = model.fit()
```

### 體制轉換模型

用於結構斷點和體制變化。

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Markov 轉換模型
model = MarkovRegression(y, k_regimes=2, order=1)
results = model.fit()

# 體制的平滑機率
regime_probs = results.smoothed_marginal_probabilities
```

## 最佳實務

1. **檢查定態性**：必要時差分，以 ADF/KPSS 檢定驗證
2. **繪製資料**：建模前始終視覺化
3. **識別季節性**：使用適當的季節性模型（SARIMAX、Holt-Winters）
4. **模型選擇**：使用 AIC/BIC 和樣本外驗證
5. **殘差診斷**：檢查自相關、常態性、異質變異數
6. **預測評估**：使用滾動預測和適當的時間序列交叉驗證
7. **避免過度配適**：偏好較簡單模型，使用資訊準則
8. **記錄假設**：註記任何資料轉換（對數、差分）
9. **預測區間**：始終提供不確定性估計
10. **定期重新配適**：新資料到達時更新模型

## 常見陷阱

1. **未檢查定態性**：在非定態資料上配適 ARIMA
2. **資料洩漏**：在轉換中使用未來資料
3. **錯誤的季節週期**：季度 S=4，月度 S=12
4. **過度配適**：相對於資料的參數過多
5. **忽略殘差自相關**：模型不充分
6. **使用不當指標**：MAPE 在零或負值時失效
7. **未處理缺失資料**：影響模型估計
8. **外推外生變數**：SARIMAX 需要未來 X 值
9. **混淆靜態和動態預測**：動態對多步更現實
10. **未驗證預測**：始終檢查樣本外表現
