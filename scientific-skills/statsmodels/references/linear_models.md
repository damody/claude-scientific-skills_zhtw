# 線性迴歸模型參考

本文件提供 statsmodels 中線性迴歸模型的詳細指引，包括 OLS、GLS、WLS、分位數迴歸和特殊變體。

## 核心模型類別

### OLS（普通最小平方法）

假設獨立同分布誤差（Σ=I）。最適合具有同質變異數誤差的標準迴歸。

**使用時機：**
- 標準迴歸分析
- 誤差獨立且具有恆定變異數
- 無自相關或異質變異數
- 最常見的起點

**基本用法：**
```python
import statsmodels.api as sm
import numpy as np

# 準備資料 - 務必為截距添加常數
X = sm.add_constant(X_data)  # 為截距添加 1 的行

# 配適模型
model = sm.OLS(y, X)
results = model.fit()

# 查看結果
print(results.summary())
```

**主要結果屬性：**
```python
results.params           # 係數
results.bse              # 標準誤
results.tvalues          # T 統計量
results.pvalues          # P 值
results.rsquared         # R 平方
results.rsquared_adj     # 調整後 R 平方
results.fittedvalues     # 擬合值（訓練資料上的預測）
results.resid            # 殘差
results.conf_int()       # 參數的信賴區間
```

**帶信賴/預測區間的預測：**
```python
# 樣本內預測
pred = results.get_prediction(X)
pred_summary = pred.summary_frame()
print(pred_summary)  # 包含均值、標準誤、信賴區間

# 樣本外預測
X_new = sm.add_constant(X_new_data)
pred_new = results.get_prediction(X_new)
pred_summary = pred_new.summary_frame()

# 存取區間
mean_ci_lower = pred_summary["mean_ci_lower"]
mean_ci_upper = pred_summary["mean_ci_upper"]
obs_ci_lower = pred_summary["obs_ci_lower"]  # 預測區間
obs_ci_upper = pred_summary["obs_ci_upper"]
```

**公式 API（R 風格）：**
```python
import statsmodels.formula.api as smf

# 自動處理類別變數和交互作用
formula = 'y ~ x1 + x2 + C(category) + x1:x2'
results = smf.ols(formula, data=df).fit()
```

### WLS（加權最小平方法）

處理異質變異數誤差（對角 Σ），其中變異數在觀測值間不同。

**使用時機：**
- 已知異質變異數（非恆定誤差變異數）
- 不同觀測值具有不同可靠性
- 權重已知或可估計

**用法：**
```python
# 若您知道權重（變異數的倒數）
weights = 1 / error_variance
model = sm.WLS(y, X, weights=weights)
results = model.fit()

# 常見權重模式：
# - 1/variance：當變異數已知時
# - n_i：分組資料的樣本大小
# - 1/x：當變異數與 x 成比例時
```

**可行 WLS（估計權重）：**
```python
# 步驟 1：配適 OLS
ols_results = sm.OLS(y, X).fit()

# 步驟 2：建模平方殘差以估計變異數
abs_resid = np.abs(ols_results.resid)
variance_model = sm.OLS(np.log(abs_resid**2), X).fit()

# 步驟 3：使用估計的變異數作為權重
weights = 1 / np.exp(variance_model.fittedvalues)
wls_results = sm.WLS(y, X, weights=weights).fit()
```

### GLS（廣義最小平方法）

處理任意共變異數結構（Σ）。其他迴歸方法的超類別。

**使用時機：**
- 已知共變異數結構
- 相關誤差
- 比 WLS 更一般

**用法：**
```python
# 指定共變異數結構
# Sigma 應為 (n x n) 共變異數矩陣
model = sm.GLS(y, X, sigma=Sigma)
results = model.fit()
```

### GLSAR（帶自迴歸誤差的 GLS）

用於具有 AR(p) 誤差的時間序列資料的可行廣義最小平方法。

**使用時機：**
- 具有自相關誤差的時間序列迴歸
- 需要考慮序列相關
- 誤差獨立性的違反

**用法：**
```python
# AR(1) 誤差
model = sm.GLSAR(y, X, rho=1)  # rho=1 表示 AR(1)，rho=2 表示 AR(2)，等等
results = model.iterative_fit()  # 迭代估計 AR 參數

print(results.summary())
print(f"Estimated rho: {results.model.rho}")
```

### RLS（遞迴最小平方法）

序列參數估計，用於適應性或線上學習。

**使用時機：**
- 參數隨時間變化
- 線上/串流資料
- 想看參數演化

**用法：**
```python
from statsmodels.regression.recursive_ls import RecursiveLS

model = RecursiveLS(y, X)
results = model.fit()

# 存取時變參數
params_over_time = results.recursive_coefficients
cusum = results.cusum  # 結構斷點的 CUSUM 統計量
```

### 滾動迴歸

計算跨移動視窗的估計以偵測時變參數。

**使用時機：**
- 參數隨時間變化
- 想偵測結構變化
- 具演化關係的時間序列

**用法：**
```python
from statsmodels.regression.rolling import RollingOLS, RollingWLS

# 60 期視窗的滾動 OLS
rolling_model = RollingOLS(y, X, window=60)
rolling_results = rolling_model.fit()

# 萃取時變參數
rolling_params = rolling_results.params  # 含隨時間參數的 DataFrame
rolling_rsquared = rolling_results.rsquared

# 繪製參數演化
import matplotlib.pyplot as plt
rolling_params.plot()
plt.title('Time-Varying Coefficients')
plt.show()
```

### 分位數迴歸

分析條件分位數而非條件均值。

**使用時機：**
- 對分位數感興趣（中位數、第 90 百分位等）
- 對離群值穩健（中位數迴歸）
- 跨分位數的分布效果
- 異質效果

**用法：**
```python
from statsmodels.regression.quantile_regression import QuantReg

# 中位數迴歸（第 50 百分位）
model = QuantReg(y, X)
results_median = model.fit(q=0.5)

# 多個分位數
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
results_dict = {}
for q in quantiles:
    results_dict[q] = model.fit(q=q)

# 繪製分位數變化效果
import matplotlib.pyplot as plt
coef_dict = {q: res.params for q, res in results_dict.items()}
coef_df = pd.DataFrame(coef_dict).T
coef_df.plot()
plt.xlabel('Quantile')
plt.ylabel('Coefficient')
plt.show()
```

## 混合效果模型

用於具有隨機效果的階層/巢狀資料。

**使用時機：**
- 群集/分組資料（學校中的學生、醫院中的病患）
- 重複測量
- 需要隨機效果以考慮分組

**用法：**
```python
from statsmodels.regression.mixed_linear_model import MixedLM

# 隨機截距模型
model = MixedLM(y, X, groups=group_ids)
results = model.fit()

# 隨機截距和斜率
model = MixedLM(y, X, groups=group_ids, exog_re=X_random)
results = model.fit()

print(results.summary())
```

## 診斷和模型評估

### 殘差分析

```python
# 基本殘差圖
import matplotlib.pyplot as plt

# 殘差 vs 擬合值
plt.scatter(results.fittedvalues, results.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted')
plt.show()

# 常態性的 Q-Q 圖
from statsmodels.graphics.gofplots import qqplot
qqplot(results.resid, line='s')
plt.show()

# 殘差直方圖
plt.hist(results.resid, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()
```

### 規格檢定

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# 異質變異數檢定
lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(results.resid, X)
print(f"Breusch-Pagan test p-value: {lm_pval}")

# White 檢定
white_test = het_white(results.resid, X)
print(f"White test p-value: {white_test[1]}")

# 自相關
dw_stat = durbin_watson(results.resid)
print(f"Durbin-Watson statistic: {dw_stat}")
# DW ~ 2 表示無自相關
# DW < 2 表示正自相關
# DW > 2 表示負自相關

# 常態性檢定
jb_stat, jb_pval, skew, kurtosis = jarque_bera(results.resid)
print(f"Jarque-Bera test p-value: {jb_pval}")
```

### 多重共線性

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 計算每個變數的 VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
# VIF > 10 表示有問題的多重共線性
# VIF > 5 表示中度多重共線性

# 條件數（從摘要）
print(f"Condition number: {results.condition_number}")
# 條件數 > 20 表示多重共線性
# 條件數 > 30 表示嚴重問題
```

### 影響統計量

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = results.get_influence()

# 槓桿（hat 值）
leverage = influence.hat_matrix_diag
# 高槓桿：> 2*p/n（p=預測變數，n=觀測值）

# Cook's distance
cooks_d = influence.cooks_distance[0]
# 若 Cook's D > 4/n 則有影響

# DFFITS
dffits = influence.dffits[0]
# 若 |DFFITS| > 2*sqrt(p/n) 則有影響

# 建立影響圖
from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots(figsize=(12, 8))
influence_plot(results, ax=ax)
plt.show()
```

### 假設檢定

```python
# 檢定單一係數
# H0: beta_i = 0（摘要中自動顯示）

# 使用 F 檢定檢定多重限制
# 範例：檢定 beta_1 = beta_2 = 0
R = [[0, 1, 0, 0], [0, 0, 1, 0]]  # 限制矩陣
f_test = results.f_test(R)
print(f_test)

# 基於公式的假設檢定
f_test = results.f_test("x1 = x2 = 0")
print(f_test)

# 檢定線性組合：beta_1 + beta_2 = 1
r_matrix = [[0, 1, 1, 0]]
q_matrix = [1]  # 右側值
f_test = results.f_test((r_matrix, q_matrix))
print(f_test)

# Wald 檢定（等同於線性限制的 F 檢定）
wald_test = results.wald_test(R)
print(wald_test)
```

## 模型比較

```python
# 使用概似比檢定比較巢狀模型（若使用 MLE）
from statsmodels.stats.anova import anova_lm

# 配適受限和非受限模型
model_restricted = sm.OLS(y, X_restricted).fit()
model_full = sm.OLS(y, X_full).fit()

# 模型比較的 ANOVA 表
anova_results = anova_lm(model_restricted, model_full)
print(anova_results)

# 非巢狀模型比較的 AIC/BIC
print(f"Model 1 AIC: {model1.aic}, BIC: {model1.bic}")
print(f"Model 2 AIC: {model2.aic}, BIC: {model2.bic}")
# 較低的 AIC/BIC 表示較佳模型
```

## 穩健標準誤

處理異質變異數或群集而不重新加權。

```python
# 異質變異數穩健（HC）標準誤
results_hc = results.get_robustcov_results(cov_type='HC0')  # White's
results_hc1 = results.get_robustcov_results(cov_type='HC1')
results_hc2 = results.get_robustcov_results(cov_type='HC2')
results_hc3 = results.get_robustcov_results(cov_type='HC3')  # 最保守

# Newey-West HAC（異質變異數和自相關一致）
results_hac = results.get_robustcov_results(cov_type='HAC', maxlags=4)

# 群集穩健標準誤
results_cluster = results.get_robustcov_results(cov_type='cluster',
                                                groups=cluster_ids)

# 查看穩健結果
print(results_hc3.summary())
```

## 最佳實務

1. **務必添加常數**：使用 `sm.add_constant()` 除非您特別想排除截距
2. **檢查假設**：執行診斷檢定（異質變異數、自相關、常態性）
3. **類別變數使用公式 API**：`smf.ols()` 自動處理類別變數
4. **穩健標準誤**：當偵測到異質變異數但模型規格正確時使用
5. **模型選擇**：非巢狀模型使用 AIC/BIC，巢狀模型使用 F 檢定/概似比
6. **離群值和影響**：始終檢查 Cook's distance 和槓桿
7. **多重共線性**：解釋前檢查 VIF 和條件數
8. **時間序列**：對自相關誤差使用 `GLSAR` 或穩健 HAC 標準誤
9. **分組資料**：考慮混合效果模型或群集穩健標準誤
10. **分位數迴歸**：用於穩健估計或對分布效果感興趣時

## 常見陷阱

1. **忘記添加常數**：結果為無截距模型
2. **忽略異質變異數**：使用 WLS 或穩健標準誤
3. **對自相關誤差使用 OLS**：使用 GLSAR 或 HAC 標準誤
4. **多重共線性下過度解釋**：先檢查 VIF
5. **未檢查殘差**：始終繪製殘差 vs 擬合值圖
6. **使用 t-SNE/PCA 殘差**：殘差應來自原始空間
7. **混淆預測區間和信賴區間**：預測區間較寬
8. **未正確處理類別變數**：使用公式 API 或手動虛擬編碼
9. **使用不同樣本大小比較模型**：確保使用相同觀測值
10. **忽略影響觀測值**：檢查 Cook's distance 和 DFFITS
