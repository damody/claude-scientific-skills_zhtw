# PyMC 抽樣與推斷方法

此參考涵蓋 PyMC 中可用於後驗推斷的抽樣演算法和推斷方法。

## MCMC 抽樣方法

### 主要抽樣函數

**`pm.sample(draws=1000, tune=1000, chains=4, **kwargs)`**

PyMC 中 MCMC 抽樣的主要介面。

**關鍵參數：**
- `draws`：每條鏈抽取的樣本數（預設：1000）
- `tune`：調整/暖機樣本數（預設：1000，丟棄）
- `chains`：平行鏈數（預設：4）
- `cores`：使用的 CPU 核心數（預設：所有可用）
- `target_accept`：步長調整的目標接受率（預設：0.8，困難後驗增加到 0.9-0.95）
- `random_seed`：用於可重現性的隨機種子
- `return_inferencedata`：回傳 ArviZ InferenceData 物件（預設：True）
- `idata_kwargs`：InferenceData 建立的額外參數（例如 `{"log_likelihood": True}` 用於模型比較）

**回傳：** 包含後驗樣本、抽樣統計和診斷的 InferenceData 物件

**範例：**
```python
with pm.Model() as model:
    # ... 定義模型 ...
    idata = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9)
```

### 抽樣演算法

PyMC 根據模型結構自動選擇適當的抽樣器，但您可以手動指定演算法。

#### NUTS（No-U-Turn Sampler）

連續參數的**預設演算法**。高效率的漢密爾頓蒙地卡羅變體。

- 自動調整步長和質量矩陣
- 自適應：在調整期間探索後驗幾何
- 最適合平滑、連續的後驗
- 高相關性或多模態時可能遇到困難

**手動指定：**
```python
with model:
    idata = pm.sample(step=pm.NUTS(target_accept=0.95))
```

**何時調整：**
- 如果看到發散，增加 `target_accept`（0.9-0.99）
- 使用 `init='adapt_diag'` 加快初始化（預設）
- 對困難的初始化使用 `init='jitter+adapt_diag'`

#### Metropolis

通用 Metropolis-Hastings 抽樣器。

- 適用於連續和離散變數
- 對平滑連續後驗不如 NUTS 高效
- 對離散參數或不可微分模型有用
- 需要手動調整

**範例：**
```python
with model:
    idata = pm.sample(step=pm.Metropolis())
```

#### 切片抽樣器

單變量分布的切片抽樣。

- 無需調整
- 適合困難的單變量後驗
- 高維度時可能緩慢

**範例：**
```python
with model:
    idata = pm.sample(step=pm.Slice())
```

#### CompoundStep

為不同參數組合不同的抽樣器。

**範例：**
```python
with model:
    # 連續參數用 NUTS，離散參數用 Metropolis
    step1 = pm.NUTS([continuous_var1, continuous_var2])
    step2 = pm.Metropolis([discrete_var])
    idata = pm.sample(step=[step1, step2])
```

### 抽樣診斷

PyMC 自動計算診斷。在信任結果前檢查這些：

#### 有效樣本大小（ESS）

測量相關樣本中的獨立資訊。

- **經驗法則**：每條鏈 ESS > 400（4 條鏈共 1600）
- ESS 低表示高自相關
- 存取方式：`az.ess(idata)`

#### R-hat（Gelman-Rubin 統計量）

測量跨鏈收斂性。

- **經驗法則**：所有參數 R-hat < 1.01
- R-hat > 1.01 表示未收斂
- 存取方式：`az.rhat(idata)`

#### 發散

指示 NUTS 遇到困難的區域。

- **經驗法則**：0 次發散（或非常少）
- 發散表示樣本可能有偏差
- **修復**：增加 `target_accept`、重新參數化或使用更強的先驗
- 存取方式：`idata.sample_stats.diverging.sum()`

#### 能量圖

視覺化漢密爾頓蒙地卡羅能量轉換。

```python
az.plot_energy(idata)
```

能量分布之間良好的分離表示健康的抽樣。

### 處理抽樣問題

#### 發散

```python
# 增加目標接受率
idata = pm.sample(target_accept=0.95)

# 或使用非中心化參數化重新參數化
# 不好（中心化）：
mu = pm.Normal('mu', 0, 1)
sigma = pm.HalfNormal('sigma', 1)
x = pm.Normal('x', mu, sigma, observed=data)

# 好（非中心化）：
mu = pm.Normal('mu', 0, 1)
sigma = pm.HalfNormal('sigma', 1)
x_offset = pm.Normal('x_offset', 0, 1, observed=(data - mu) / sigma)
```

#### 抽樣緩慢

```python
# 如果模型簡單，使用較少的調整步驟
idata = pm.sample(tune=500)

# 增加核心數進行平行化
idata = pm.sample(cores=8, chains=8)

# 使用變分推斷進行初始化
with model:
    approx = pm.fit()  # 執行 ADVI
    idata = pm.sample(start=approx.sample(return_inferencedata=False)[0])
```

#### 高自相關

```python
# 增加抽樣數
idata = pm.sample(draws=5000)

# 重新參數化以減少相關性
# 考慮對迴歸模型使用 QR 分解
```

## 變分推斷

用於大型模型或快速探索的較快近似推斷。

### ADVI（自動微分變分推斷）

**`pm.fit(n=10000, method='advi', **kwargs)`**

用較簡單的分布（通常是平均場高斯）近似後驗。

**關鍵參數：**
- `n`：迭代次數（預設：10000）
- `method`：VI 演算法（'advi'、'fullrank_advi'、'svgd'）
- `random_seed`：隨機種子

**回傳：** 用於抽樣和分析的近似物件

**範例：**
```python
with model:
    approx = pm.fit(n=50000)
    # 從近似抽取樣本
    idata = approx.sample(1000)
    # 或抽樣用於 MCMC 初始化
    start = approx.sample(return_inferencedata=False)[0]
```

**權衡：**
- **優點**：比 MCMC 快得多，可擴展到大資料
- **缺點**：近似，可能遺漏後驗結構，低估不確定性

### 全秩 ADVI

捕捉參數之間的相關性。

```python
with model:
    approx = pm.fit(method='fullrank_advi')
```

比平均場更準確但更慢。

### SVGD（Stein 變分梯度下降）

非參數變分推斷。

```python
with model:
    approx = pm.fit(method='svgd', n=20000)
```

更好地捕捉多模態，但計算成本更高。

## 先驗和後驗預測抽樣

### 先驗預測抽樣

從先驗分布抽樣（觀察資料前）。

**`pm.sample_prior_predictive(samples=500, **kwargs)`**

**目的：**
- 驗證先驗是否合理
- 擬合前檢查隱含預測
- 確保模型生成合理資料

**範例：**
```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000)

# 視覺化先驗預測
az.plot_ppc(prior_pred, group='prior')
```

### 後驗預測抽樣

從後驗預測分布抽樣（擬合後）。

**`pm.sample_posterior_predictive(trace, **kwargs)`**

**目的：**
- 透過後驗預測檢查驗證模型
- 為新資料生成預測
- 評估擬合優度

**範例：**
```python
with model:
    # 抽樣後
    idata = pm.sample()

    # 添加後驗預測樣本
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# 後驗預測檢查
az.plot_ppc(idata)
```

### 新資料預測

更新資料並抽取預測分布：

```python
with model:
    # 原始模型擬合
    idata = pm.sample()

    # 用新的預測變數值更新
    pm.set_data({'X': X_new})

    # 抽取預測
    post_pred_new = pm.sample_posterior_predictive(
        idata.posterior,
        var_names=['y_pred']
    )
```

## 最大後驗（MAP）估計

找到後驗眾數（點估計）。

**`pm.find_MAP(start=None, method='L-BFGS-B', **kwargs)`**

**何時使用：**
- 快速點估計
- MCMC 初始化
- 不需要完整後驗時

**範例：**
```python
with model:
    map_estimate = pm.find_MAP()
    print(map_estimate)
```

**限制：**
- 不量化不確定性
- 多模態後驗可能找到局部最優
- 對先驗設定敏感

## 推斷建議

### 標準工作流程

1. **從 ADVI 開始**進行快速探索：
   ```python
   approx = pm.fit(n=20000)
   ```

2. **執行 MCMC** 進行完整推斷：
   ```python
   idata = pm.sample(draws=2000, tune=1000)
   ```

3. **檢查診斷**：
   ```python
   az.summary(idata, var_names=['~mu_log__'])  # 排除轉換變數
   ```

4. **抽取後驗預測**：
   ```python
   pm.sample_posterior_predictive(idata, extend_inferencedata=True)
   ```

### 選擇推斷方法

| 情境 | 推薦方法 |
|----------|-------------------|
| 中小型模型，需要完整不確定性 | 使用 NUTS 的 MCMC |
| 大型模型，初步探索 | ADVI |
| 離散參數 | Metropolis 或邊際化 |
| 有發散的階層式模型 | 非中心化參數化 + NUTS |
| 非常大的資料 | Minibatch ADVI |
| 快速點估計 | MAP 或 ADVI |

### 重新參數化技巧

階層式模型的**非中心化參數化**：

```python
# 中心化（可能導致發散）：
mu = pm.Normal('mu', 0, 10)
sigma = pm.HalfNormal('sigma', 1)
theta = pm.Normal('theta', mu, sigma, shape=n_groups)

# 非中心化（更好的抽樣）：
mu = pm.Normal('mu', 0, 10)
sigma = pm.HalfNormal('sigma', 1)
theta_offset = pm.Normal('theta_offset', 0, 1, shape=n_groups)
theta = pm.Deterministic('theta', mu + sigma * theta_offset)
```

相關預測變數的 **QR 分解**：

```python
import numpy as np

# QR 分解
Q, R = np.linalg.qr(X)

with pm.Model():
    # 不相關係數
    beta_tilde = pm.Normal('beta_tilde', 0, 1, shape=p)

    # 轉換回原始尺度
    beta = pm.Deterministic('beta', pm.math.solve(R, beta_tilde))

    mu = pm.math.dot(Q, beta_tilde)
    sigma = pm.HalfNormal('sigma', 1)
    y = pm.Normal('y', mu, sigma, observed=y_obs)
```

## 進階抽樣

### 序列蒙地卡羅（SMC）

用於複雜後驗或模型證據估計：

```python
with model:
    idata = pm.sample_smc(draws=2000, chains=4)
```

適合多模態後驗或 NUTS 遇到困難時。

### 自訂初始化

提供起始值：

```python
start = {'mu': 0, 'sigma': 1}
with model:
    idata = pm.sample(start=start)
```

或使用 MAP 估計：

```python
with model:
    start = pm.find_MAP()
    idata = pm.sample(start=start)
```
