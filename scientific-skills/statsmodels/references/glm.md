# 廣義線性模型（GLM）參考

本文件提供 statsmodels 中廣義線性模型的完整指引，包括分布族、連結函數和應用。

## 概述

GLM 透過以下方式將線性迴歸擴展至非常態反應分布：
1. **分布族**：指定反應變數的條件分布
2. **連結函數**：將線性預測變數轉換至均值的尺度
3. **變異數函數**：將變異數與均值關聯

**一般形式**：g(μ) = Xβ，其中 g 是連結函數，μ = E(Y|X)

## 何時使用 GLM

- **二元結果**：邏輯斯迴歸（二項族配 logit 連結）
- **計數資料**：Poisson 或負二項迴歸
- **正值連續資料**：Gamma 或逆高斯
- **非常態分布**：當 OLS 假設違反時
- **連結函數**：需要預測變數與反應尺度間的非線性關係

## 分布族

### 二項族（Binomial Family）

用於二元結果（0/1）或比例（k/n）。

**使用時機：**
- 二元分類
- 成功/失敗結果
- 比例或率

**常用連結：**
- Logit（預設）：log(μ/(1-μ))
- Probit：Φ⁻¹(μ)
- Log：log(μ)

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 二元邏輯斯迴歸
model = sm.GLM(y, X, family=sm.families.Binomial())
results = model.fit()

# 公式 API
results = smf.glm('success ~ x1 + x2', data=df,
                  family=sm.families.Binomial()).fit()

# 存取預測（機率）
probs = results.predict(X_new)

# 分類（0.5 閾值）
predictions = (probs > 0.5).astype(int)
```

**解釋：**
```python
import numpy as np

# 勝算比（對於 logit 連結）
odds_ratios = np.exp(results.params)
print("Odds ratios:", odds_ratios)

# x 每增加 1 單位，勝算乘以 exp(beta)
```

### Poisson 族

用於計數資料（非負整數）。

**使用時機：**
- 計數結果（事件數）
- 稀有事件
- 率建模（含偏移量）

**常用連結：**
- Log（預設）：log(μ)
- Identity：μ
- Sqrt：√μ

```python
# Poisson 迴歸
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()

# 含暴露量/偏移量用於率
# 若建模率 = 計數/暴露量
model = sm.GLM(y, X, family=sm.families.Poisson(),
               offset=np.log(exposure))
results = model.fit()

# 解釋：exp(beta) = 對期望計數的乘法效果
import numpy as np
rate_ratios = np.exp(results.params)
print("Rate ratios:", rate_ratios)
```

**過度離散檢查：**
```python
# Poisson 的離差 / df 應約為 1
overdispersion = results.deviance / results.df_resid
print(f"Overdispersion: {overdispersion}")

# 若 >> 1，考慮負二項
if overdispersion > 1.5:
    print("Consider Negative Binomial model for overdispersion")
```

### 負二項族（Negative Binomial Family）

用於過度離散的計數資料。

**使用時機：**
- 變異數 > 均值的計數資料
- 過多零值或大變異數
- Poisson 模型顯示過度離散

```python
# 負二項 GLM
model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
results = model.fit()

# 替代方案：使用離散選擇模型估計 alpha
from statsmodels.discrete.discrete_model import NegativeBinomial
nb_model = NegativeBinomial(y, X)
nb_results = nb_model.fit()

print(f"Dispersion parameter alpha: {nb_results.params[-1]}")
```

### 高斯族（Gaussian Family）

等同於 OLS，但透過 IRLS（迭代重加權最小平方法）配適。

**使用時機：**
- 為一致性而使用 GLM 框架
- 需要穩健標準誤
- 與其他 GLM 比較

**常用連結：**
- Identity（預設）：μ
- Log：log(μ)
- Inverse：1/μ

```python
# 高斯 GLM（等同於 OLS）
model = sm.GLM(y, X, family=sm.families.Gaussian())
results = model.fit()

# 驗證與 OLS 等價
ols_results = sm.OLS(y, X).fit()
print("Parameters close:", np.allclose(results.params, ols_results.params))
```

### Gamma 族

用於正值連續資料，通常右偏。

**使用時機：**
- 正值結果（保險理賠、存活時間）
- 右偏分布
- 變異數與均值² 成比例

**常用連結：**
- Inverse（預設）：1/μ
- Log：log(μ)
- Identity：μ

```python
# Gamma 迴歸（常用於成本資料）
model = sm.GLM(y, X, family=sm.families.Gamma())
results = model.fit()

# Log 連結通常較佳於解釋
model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
results = model.fit()

# 使用 log 連結，exp(beta) = 乘法效果
import numpy as np
effects = np.exp(results.params)
```

### 逆高斯族（Inverse Gaussian Family）

用於具特定變異數結構的正值連續資料。

**使用時機：**
- 正偏結果
- 變異數與均值³ 成比例
- Gamma 的替代方案

**常用連結：**
- Inverse squared（預設）：1/μ²
- Log：log(μ)

```python
model = sm.GLM(y, X, family=sm.families.InverseGaussian())
results = model.fit()
```

### Tweedie 族

涵蓋多種分布的彈性族。

**使用時機：**
- 保險理賠（零值和連續值的混合）
- 半連續資料
- 需要彈性變異數函數

**特殊情況（power 參數 p）：**
- p=0：常態
- p=1：Poisson
- p=2：Gamma
- p=3：逆高斯
- 1<p<2：複合 Poisson-Gamma（保險常用）

```python
# Tweedie 配 power=1.5
model = sm.GLM(y, X, family=sm.families.Tweedie(link=sm.families.links.Log(),
                                                 var_power=1.5))
results = model.fit()
```

## 連結函數

連結函數連接線性預測變數與反應變數的均值。

### 可用連結

```python
from statsmodels.genmod import families

# Identity：g(μ) = μ
link = families.links.Identity()

# Log：g(μ) = log(μ)
link = families.links.Log()

# Logit：g(μ) = log(μ/(1-μ))
link = families.links.Logit()

# Probit：g(μ) = Φ⁻¹(μ)
link = families.links.Probit()

# Complementary log-log：g(μ) = log(-log(1-μ))
link = families.links.CLogLog()

# Inverse：g(μ) = 1/μ
link = families.links.InversePower()

# Inverse squared：g(μ) = 1/μ²
link = families.links.InverseSquared()

# Square root：g(μ) = √μ
link = families.links.Sqrt()

# Power：g(μ) = μ^p
link = families.links.Power(power=2)
```

### 選擇連結函數

**正則連結（每個族的預設）：**
- 二項 → Logit
- Poisson → Log
- Gamma → Inverse
- 高斯 → Identity
- 逆高斯 → Inverse squared

**何時使用非正則：**
- **二項配 Log 連結**：風險比而非勝算比
- **Identity 連結**：直接加法效果（當合理時）
- **Probit vs Logit**：類似結果，根據領域偏好
- **CLogLog**：非對稱關係，存活分析常用

```python
# 範例：log-binomial 模型的風險比
model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Log()))
results = model.fit()

# exp(beta) 現在給出風險比，不是勝算比
risk_ratios = np.exp(results.params)
```

## 模型配適與結果

### 基本工作流程

```python
import statsmodels.api as sm

# 添加常數
X = sm.add_constant(X_data)

# 指定族和連結
family = sm.families.Poisson(link=sm.families.links.Log())

# 使用 IRLS 配適模型
model = sm.GLM(y, X, family=family)
results = model.fit()

# 摘要
print(results.summary())
```

### 結果屬性

```python
# 參數和推論
results.params              # 係數
results.bse                 # 標準誤
results.tvalues            # Z 統計量
results.pvalues            # P 值
results.conf_int()         # 信賴區間

# 預測
results.fittedvalues       # 擬合值（μ）
results.predict(X_new)     # 新資料的預測

# 模型配適統計量
results.aic                # Akaike 資訊準則
results.bic                # Bayesian 資訊準則
results.deviance           # 離差
results.null_deviance      # 虛無模型離差
results.pearson_chi2       # Pearson 卡方統計量
results.df_resid           # 殘差自由度
results.llf                # 對數概似

# 殘差
results.resid_response     # 反應殘差（y - μ）
results.resid_pearson      # Pearson 殘差
results.resid_deviance     # 離差殘差
results.resid_anscombe     # Anscombe 殘差
results.resid_working      # 工作殘差
```

### 虛擬 R 平方

```python
# McFadden 虛擬 R 平方
pseudo_r2 = 1 - (results.deviance / results.null_deviance)
print(f"Pseudo R²: {pseudo_r2:.4f}")

# 調整後虛擬 R 平方
n = len(y)
k = len(results.params)
adj_pseudo_r2 = 1 - ((n-1)/(n-k)) * (results.deviance / results.null_deviance)
print(f"Adjusted Pseudo R²: {adj_pseudo_r2:.4f}")
```

## 診斷

### 配適度

```python
# 離差應約為具 df_resid 自由度的 χ² 分布
from scipy import stats

deviance_pval = 1 - stats.chi2.cdf(results.deviance, results.df_resid)
print(f"Deviance test p-value: {deviance_pval}")

# Pearson 卡方檢定
pearson_pval = 1 - stats.chi2.cdf(results.pearson_chi2, results.df_resid)
print(f"Pearson chi² test p-value: {pearson_pval}")

# 檢查過度離散/欠離散
dispersion = results.pearson_chi2 / results.df_resid
print(f"Dispersion: {dispersion}")
# 應約為 1；>1 表示過度離散，<1 表示欠離散
```

### 殘差分析

```python
import matplotlib.pyplot as plt

# 離差殘差 vs 擬合值
plt.figure(figsize=(10, 6))
plt.scatter(results.fittedvalues, results.resid_deviance, alpha=0.5)
plt.xlabel('Fitted values')
plt.ylabel('Deviance residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Deviance Residuals vs Fitted')
plt.show()

# 離差殘差的 Q-Q 圖
from statsmodels.graphics.gofplots import qqplot
qqplot(results.resid_deviance, line='s')
plt.title('Q-Q Plot of Deviance Residuals')
plt.show()

# 二元結果：分組殘差圖
if isinstance(results.model.family, sm.families.Binomial):
    from statsmodels.graphics.gofplots import qqplot
    # 分組預測並計算平均殘差
    # （需要自訂實作）
    pass
```

### 影響和離群值

```python
from statsmodels.stats.outliers_influence import GLMInfluence

influence = GLMInfluence(results)

# 槓桿
leverage = influence.hat_matrix_diag

# Cook's distance
cooks_d = influence.cooks_distance[0]

# DFFITS
dffits = influence.dffits[0]

# 找出影響觀測值
influential = np.where(cooks_d > 4/len(y))[0]
print(f"Influential observations: {influential}")
```

## 假設檢定

```python
# 單參數 Wald 檢定（摘要中自動顯示）

# 巢狀模型的概似比檢定
# 配適精簡模型
model_reduced = sm.GLM(y, X_reduced, family=family).fit()
model_full = sm.GLM(y, X_full, family=family).fit()

# LR 統計量
lr_stat = 2 * (model_full.llf - model_reduced.llf)
df = model_full.df_model - model_reduced.df_model

from scipy import stats
lr_pval = 1 - stats.chi2.cdf(lr_stat, df)
print(f"LR test p-value: {lr_pval}")

# 多參數 Wald 檢定
# 檢定 beta_1 = beta_2 = 0
R = [[0, 1, 0, 0], [0, 0, 1, 0]]
wald_test = results.wald_test(R)
print(wald_test)
```

## 穩健標準誤

```python
# 異質變異數穩健（三明治估計量）
results_robust = results.get_robustcov_results(cov_type='HC0')

# 群集穩健
results_cluster = results.get_robustcov_results(cov_type='cluster',
                                                groups=cluster_ids)

# 比較標準誤
print("Regular SE:", results.bse)
print("Robust SE:", results_robust.bse)
```

## 模型比較

```python
# 非巢狀模型的 AIC/BIC
models = [model1_results, model2_results, model3_results]
for i, res in enumerate(models, 1):
    print(f"Model {i}: AIC={res.aic:.2f}, BIC={res.bic:.2f}")

# 巢狀模型的概似比檢定（如上所示）

# 預測表現的交叉驗證
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_cv = sm.GLM(y_train, X_train, family=family).fit()
    pred_probs = model_cv.predict(X_val)

    score = log_loss(y_val, pred_probs)
    cv_scores.append(score)

print(f"CV Log Loss: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

## 預測

```python
# 點預測
predictions = results.predict(X_new)

# 分類：取得機率並轉換
if isinstance(family, sm.families.Binomial):
    probs = predictions
    class_predictions = (probs > 0.5).astype(int)

# 計數：預測為期望計數
if isinstance(family, sm.families.Poisson):
    expected_counts = predictions

# 透過 bootstrap 的預測區間
n_boot = 1000
boot_preds = np.zeros((n_boot, len(X_new)))

for i in range(n_boot):
    # Bootstrap 重抽樣
    boot_idx = np.random.choice(len(y), size=len(y), replace=True)
    X_boot, y_boot = X[boot_idx], y[boot_idx]

    # 配適並預測
    boot_model = sm.GLM(y_boot, X_boot, family=family).fit()
    boot_preds[i] = boot_model.predict(X_new)

# 95% 預測區間
pred_lower = np.percentile(boot_preds, 2.5, axis=0)
pred_upper = np.percentile(boot_preds, 97.5, axis=0)
```

## 常見應用

### 邏輯斯迴歸（二元分類）

```python
import statsmodels.api as sm

# 配適邏輯斯迴歸
X = sm.add_constant(X_data)
model = sm.GLM(y, X, family=sm.families.Binomial())
results = model.fit()

# 勝算比
odds_ratios = np.exp(results.params)
odds_ci = np.exp(results.conf_int())

# 分類指標
from sklearn.metrics import classification_report, roc_auc_score

probs = results.predict(X)
predictions = (probs > 0.5).astype(int)

print(classification_report(y, predictions))
print(f"AUC: {roc_auc_score(y, probs):.4f}")

# ROC 曲線
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y, probs)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

### Poisson 迴歸（計數資料）

```python
# 配適 Poisson 模型
X = sm.add_constant(X_data)
model = sm.GLM(y_counts, X, family=sm.families.Poisson())
results = model.fit()

# 率比
rate_ratios = np.exp(results.params)
print("Rate ratios:", rate_ratios)

# 檢查過度離散
dispersion = results.pearson_chi2 / results.df_resid
if dispersion > 1.5:
    print(f"Overdispersion detected ({dispersion:.2f}). Consider Negative Binomial.")
```

### Gamma 迴歸（成本/持續時間資料）

```python
# 配適 Gamma 模型配 log 連結
X = sm.add_constant(X_data)
model = sm.GLM(y_cost, X,
               family=sm.families.Gamma(link=sm.families.links.Log()))
results = model.fit()

# 乘法效果
effects = np.exp(results.params)
print("Multiplicative effects on mean:", effects)
```

## 最佳實務

1. **檢查分布假設**：繪製反應變數的直方圖和 Q-Q 圖
2. **驗證連結函數**：除非有理由否則使用正則連結
3. **檢查殘差**：離差殘差應約為常態
4. **檢定過度離散**：特別是 Poisson 模型
5. **適當使用偏移量**：用於具有不同暴露量的率建模
6. **考慮穩健標準誤**：當變異數假設有疑問時
7. **比較模型**：非巢狀使用 AIC/BIC，巢狀使用 LR 檢定
8. **在原始尺度上解釋**：轉換係數（例如 log 連結用 exp）
9. **檢查影響觀測值**：使用 Cook's distance
10. **驗證預測**：使用交叉驗證或保留集

## 常見陷阱

1. **忘記添加常數**：無截距項
2. **使用錯誤族**：檢查反應變數的分布
3. **忽略過度離散**：改用負二項而非 Poisson
4. **誤解係數**：記住連結函數轉換
5. **未檢查收斂**：IRLS 可能不收斂；檢查警告
6. **邏輯斯中的完全分離**：某些類別完美預測結果
7. **有界結果使用 identity 連結**：可能預測超出有效範圍
8. **使用不同樣本比較模型**：使用相同觀測值
9. **率模型忘記偏移量**：必須使用 log(exposure) 作為偏移量
10. **未考慮替代方案**：複雜資料考慮混合模型、零膨脹
