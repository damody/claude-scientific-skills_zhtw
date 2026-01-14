# PyMC 工作流程與常見模式

此參考提供在 PyMC 中建構、驗證和分析貝氏模型的標準工作流程和模式。

## 標準貝氏工作流程

### 完整工作流程範本

```python
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# 1. 準備資料
# ===============
X = ...  # 預測變數
y = ...  # 觀察結果

# 標準化預測變數以改善抽樣
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# 2. 建構模型
# ==============
with pm.Model() as model:
    # 定義具名維度的座標
    coords = {
        'predictors': ['var1', 'var2', 'var3'],
        'obs_id': np.arange(len(y))
    }

    # 先驗分布
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, dims='predictors')
    sigma = pm.HalfNormal('sigma', sigma=1)

    # 線性預測器
    mu = alpha + pm.math.dot(X_scaled, beta)

    # 概似函數
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y, dims='obs_id')

# 3. 先驗預測檢查
# ==========================
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)

# 視覺化先驗預測
az.plot_ppc(prior_pred, group='prior', num_pp_samples=100)
plt.title('先驗預測檢查')
plt.show()

# 4. 擬合模型
# ============
with model:
    # 快速 VI 探索（可選）
    approx = pm.fit(n=20000, random_seed=42)

    # 完整 MCMC 推斷
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        idata_kwargs={'log_likelihood': True}  # 用於模型比較
    )

# 5. 檢查診斷
# ====================
# 摘要統計
print(az.summary(idata, var_names=['alpha', 'beta', 'sigma']))

# R-hat 和 ESS
summary = az.summary(idata)
if (summary['r_hat'] > 1.01).any():
    print("警告：某些 R-hat 值 > 1.01，鏈可能未收斂")

if (summary['ess_bulk'] < 400).any():
    print("警告：某些 ESS 值 < 400，考慮更多樣本")

# 檢查發散
divergences = idata.sample_stats.diverging.sum().item()
print(f"發散次數：{divergences}")

# 軌跡圖
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'])
plt.tight_layout()
plt.show()

# 6. 後驗預測檢查
# ==============================
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

# 視覺化擬合
az.plot_ppc(idata, num_pp_samples=100)
plt.title('後驗預測檢查')
plt.show()

# 7. 分析結果
# ==================
# 後驗分布
az.plot_posterior(idata, var_names=['alpha', 'beta', 'sigma'])
plt.tight_layout()
plt.show()

# 係數森林圖
az.plot_forest(idata, var_names=['beta'], combined=True)
plt.title('係數估計')
plt.show()

# 8. 新資料預測
# ============================
X_new = ...  # 新的預測變數值
X_new_scaled = (X_new - X.mean(axis=0)) / X.std(axis=0)

with model:
    # 更新資料
    pm.set_data({'X': X_new_scaled})

    # 抽取預測
    post_pred = pm.sample_posterior_predictive(
        idata.posterior,
        var_names=['y_obs'],
        random_seed=42
    )

# 預測區間
y_pred_mean = post_pred.posterior_predictive['y_obs'].mean(dim=['chain', 'draw'])
y_pred_hdi = az.hdi(post_pred.posterior_predictive, var_names=['y_obs'])

# 9. 儲存結果
# ===============
idata.to_netcdf('model_results.nc')  # 儲存以供後續使用
```

## 模型建構模式

### 線性迴歸

```python
with pm.Model() as linear_model:
    # 先驗分布
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=n_predictors)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # 線性預測器
    mu = alpha + pm.math.dot(X, beta)

    # 概似函數
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs)
```

### 邏輯迴歸

```python
with pm.Model() as logistic_model:
    # 先驗分布
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=n_predictors)

    # 線性預測器
    logit_p = alpha + pm.math.dot(X, beta)

    # 概似函數
    y = pm.Bernoulli('y', logit_p=logit_p, observed=y_obs)
```

### 階層式/多層次模型

```python
with pm.Model(coords={'group': group_names, 'obs': np.arange(n_obs)}) as hierarchical_model:
    # 超先驗
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)

    mu_beta = pm.Normal('mu_beta', mu=0, sigma=10)
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)

    # 群組層級參數（非中心化）
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, dims='group')
    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset, dims='group')

    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, dims='group')
    beta = pm.Deterministic('beta', mu_beta + sigma_beta * beta_offset, dims='group')

    # 觀察層級模型
    mu = alpha[group_idx] + beta[group_idx] * X

    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs, dims='obs')
```

### 卜瓦松迴歸（計數資料）

```python
with pm.Model() as poisson_model:
    # 先驗分布
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=n_predictors)

    # 對數尺度上的線性預測器
    log_lambda = alpha + pm.math.dot(X, beta)

    # 概似函數
    y = pm.Poisson('y', mu=pm.math.exp(log_lambda), observed=y_obs)
```

### 時間序列（自迴歸）

```python
with pm.Model() as ar_model:
    # 創新標準差
    sigma = pm.HalfNormal('sigma', sigma=1)

    # AR 係數
    rho = pm.Normal('rho', mu=0, sigma=0.5, shape=ar_order)

    # 初始分布
    init_dist = pm.Normal.dist(mu=0, sigma=sigma)

    # AR 過程
    y = pm.AR('y', rho=rho, sigma=sigma, init_dist=init_dist, observed=y_obs)
```

### 混合模型

```python
with pm.Model() as mixture_model:
    # 成分權重
    w = pm.Dirichlet('w', a=np.ones(n_components))

    # 成分參數
    mu = pm.Normal('mu', mu=0, sigma=10, shape=n_components)
    sigma = pm.HalfNormal('sigma', sigma=1, shape=n_components)

    # 混合
    components = [pm.Normal.dist(mu=mu[i], sigma=sigma[i]) for i in range(n_components)]
    y = pm.Mixture('y', w=w, comp_dists=components, observed=y_obs)
```

## 資料準備最佳實務

### 標準化

標準化連續預測變數以改善抽樣：

```python
# 標準化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# 使用縮放資料的模型
with pm.Model() as model:
    beta_scaled = pm.Normal('beta_scaled', 0, 1)
    # ... 模型其餘部分 ...

# 轉換回原始尺度
beta_original = beta_scaled / X_std
alpha_original = alpha - (beta_scaled * X_mean / X_std).sum()
```

### 處理遺漏資料

將遺漏值視為參數：

```python
# 識別遺漏值
missing_idx = np.isnan(X)
X_observed = np.where(missing_idx, 0, X)  # 佔位符

with pm.Model() as model:
    # 遺漏值的先驗
    X_missing = pm.Normal('X_missing', mu=0, sigma=1, shape=missing_idx.sum())

    # 組合觀察值和插補值
    X_complete = pm.math.switch(missing_idx.flatten(), X_missing, X_observed.flatten())

    # ... 使用 X_complete 的模型其餘部分 ...
```

### 中心化和縮放

對於迴歸模型，中心化預測變數和結果變數：

```python
# 中心化
X_centered = X - X.mean(axis=0)
y_centered = y - y.mean()

with pm.Model() as model:
    # 截距的較簡單先驗
    alpha = pm.Normal('alpha', mu=0, sigma=1)  # 中心化時截距接近 0
    beta = pm.Normal('beta', mu=0, sigma=1, shape=n_predictors)

    mu = alpha + pm.math.dot(X_centered, beta)
    sigma = pm.HalfNormal('sigma', sigma=1)

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_centered)
```

## 先驗選擇指南

### 弱資訊先驗

先驗知識有限時使用：

```python
# 標準化預測變數
beta = pm.Normal('beta', mu=0, sigma=1)

# 尺度參數
sigma = pm.HalfNormal('sigma', sigma=1)

# 機率
p = pm.Beta('p', alpha=2, beta=2)  # 略微偏好中間值
```

### 資訊先驗

使用領域知識：

```python
# 來自文獻的效應大小：Cohen's d ≈ 0.3
beta = pm.Normal('beta', mu=0.3, sigma=0.1)

# 物理約束：機率介於 0.7-0.9
p = pm.Beta('p', alpha=8, beta=2)  # 用先驗預測檢查！
```

### 先驗預測檢查

始終驗證先驗：

```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000)

# 檢查預測是否合理
print(f"先驗預測範圍：{prior_pred.prior_predictive['y'].min():.2f} 到 {prior_pred.prior_predictive['y'].max():.2f}")
print(f"觀察範圍：{y_obs.min():.2f} 到 {y_obs.max():.2f}")

# 視覺化
az.plot_ppc(prior_pred, group='prior')
```

## 模型比較工作流程

### 比較多個模型

```python
import arviz as az

# 擬合多個模型
models = {}
idatas = {}

# 模型 1：簡單線性
with pm.Model() as models['linear']:
    # ... 定義模型 ...
    idatas['linear'] = pm.sample(idata_kwargs={'log_likelihood': True})

# 模型 2：帶交互作用
with pm.Model() as models['interaction']:
    # ... 定義模型 ...
    idatas['interaction'] = pm.sample(idata_kwargs={'log_likelihood': True})

# 模型 3：階層式
with pm.Model() as models['hierarchical']:
    # ... 定義模型 ...
    idatas['hierarchical'] = pm.sample(idata_kwargs={'log_likelihood': True})

# 使用 LOO 比較
comparison = az.compare(idatas, ic='loo')
print(comparison)

# 視覺化比較
az.plot_compare(comparison)
plt.show()

# 檢查 LOO 可靠性
for name, idata in idatas.items():
    loo = az.loo(idata, pointwise=True)
    high_pareto_k = (loo.pareto_k > 0.7).sum().item()
    if high_pareto_k > 0:
        print(f"警告：{name} 有 {high_pareto_k} 個觀察值具有高 Pareto-k")
```

### 模型權重

```python
# 取得模型權重（偽 BMA）
weights = comparison['weight'].values

print("模型機率：")
for name, weight in zip(comparison.index, weights):
    print(f"  {name}：{weight:.2%}")

# 模型平均（加權預測）
def weighted_predictions(idatas, weights):
    preds = []
    for (name, idata), weight in zip(idatas.items(), weights):
        pred = idata.posterior_predictive['y_obs'].mean(dim=['chain', 'draw'])
        preds.append(weight * pred)
    return sum(preds)

averaged_pred = weighted_predictions(idatas, weights)
```

## 診斷與疑難排解

### 診斷抽樣問題

```python
def diagnose_sampling(idata, var_names=None):
    """全面的抽樣診斷"""

    # 檢查收斂
    summary = az.summary(idata, var_names=var_names)

    print("=== 收斂診斷 ===")
    bad_rhat = summary[summary['r_hat'] > 1.01]
    if len(bad_rhat) > 0:
        print(f"⚠️  {len(bad_rhat)} 個變數的 R-hat > 1.01")
        print(bad_rhat[['r_hat']])
    else:
        print("✓ 所有 R-hat 值 < 1.01")

    # 檢查有效樣本大小
    print("\n=== 有效樣本大小 ===")
    low_ess = summary[summary['ess_bulk'] < 400]
    if len(low_ess) > 0:
        print(f"⚠️  {len(low_ess)} 個變數的 ESS < 400")
        print(low_ess[['ess_bulk', 'ess_tail']])
    else:
        print("✓ 所有 ESS 值 > 400")

    # 檢查發散
    print("\n=== 發散 ===")
    divergences = idata.sample_stats.diverging.sum().item()
    if divergences > 0:
        print(f"⚠️  {divergences} 次發散轉換")
        print("   考慮：增加 target_accept、重新參數化或更強的先驗")
    else:
        print("✓ 無發散")

    # 檢查樹深度
    print("\n=== NUTS 統計 ===")
    max_treedepth = idata.sample_stats.tree_depth.max().item()
    hits_max = (idata.sample_stats.tree_depth == max_treedepth).sum().item()
    if hits_max > 0:
        print(f"⚠️  達到最大樹深度 {hits_max} 次")
        print("   考慮：重新參數化或增加 max_treedepth")
    else:
        print(f"✓ 無最大樹深度問題（最大：{max_treedepth}）")

    return summary

# 使用方式
diagnose_sampling(idata, var_names=['alpha', 'beta', 'sigma'])
```

### 常見修復方法

| 問題 | 解決方案 |
|---------|----------|
| 發散 | 增加 `target_accept=0.95`，使用非中心化參數化 |
| ESS 低 | 抽取更多樣本，重新參數化以減少相關性 |
| R-hat 高 | 執行更長的鏈，檢查多模態，改善初始化 |
| 抽樣緩慢 | 使用 ADVI 初始化，重新參數化，降低模型複雜度 |
| 後驗有偏差 | 檢查先驗預測，確保概似函數正確 |

## 使用具名維度（dims）

### dims 的好處

- 更易讀的程式碼
- 更容易的子集選取和分析
- 更好的 xarray 整合

```python
# 定義座標
coords = {
    'predictors': ['age', 'income', 'education'],
    'groups': ['A', 'B', 'C'],
    'time': pd.date_range('2020-01-01', periods=100, freq='D')
}

with pm.Model(coords=coords) as model:
    # 使用 dims 而非 shape
    beta = pm.Normal('beta', mu=0, sigma=1, dims='predictors')
    alpha = pm.Normal('alpha', mu=0, sigma=1, dims='groups')
    y = pm.Normal('y', mu=0, sigma=1, dims=['groups', 'time'], observed=data)

# 抽樣後，維度被保留
idata = pm.sample()

# 容易的子集選取
beta_age = idata.posterior['beta'].sel(predictors='age')
group_A = idata.posterior['alpha'].sel(groups='A')
```

## 儲存和載入結果

```python
# 儲存 InferenceData
idata.to_netcdf('results.nc')

# 載入 InferenceData
loaded_idata = az.from_netcdf('results.nc')

# 儲存模型以供後續預測
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'idata': idata}, f)

# 載入模型
with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    idata = saved['idata']
```
