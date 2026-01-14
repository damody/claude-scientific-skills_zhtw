# 離散選擇模型參考

本文件提供 statsmodels 中離散選擇模型的完整指引，包括二元、多項、計數和序數模型。

## 概述

離散選擇模型處理以下類型的結果：
- **二元（Binary）**：0/1、成功/失敗
- **多項（Multinomial）**：多個無序類別
- **序數（Ordinal）**：有序類別
- **計數（Count）**：非負整數

所有模型使用最大概似估計，並假設 i.i.d. 誤差。

## 二元模型

### Logit（邏輯斯迴歸）

使用邏輯斯分布處理二元結果。

**使用時機：**
- 二元分類（是/否、成功/失敗）
- 二元結果的機率估計
- 可解釋的勝算比

**模型**：P(Y=1|X) = 1 / (1 + exp(-Xβ))

```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

# 準備資料
X = sm.add_constant(X_data)

# 配適模型
model = Logit(y, X)
results = model.fit()

print(results.summary())
```

**解釋：**
```python
import numpy as np

# 勝算比
odds_ratios = np.exp(results.params)
print("Odds ratios:", odds_ratios)

# X 每增加 1 單位，勝算乘以 exp(β)
# OR > 1：增加成功的勝算
# OR < 1：減少成功的勝算
# OR = 1：無效果

# 勝算比的信賴區間
odds_ci = np.exp(results.conf_int())
print("Odds ratio 95% CI:")
print(odds_ci)
```

**邊際效應：**
```python
# 平均邊際效應（AME）
marginal_effects = results.get_margeff(at='mean')
print(marginal_effects.summary())

# 均值處的邊際效應（MEM）
marginal_effects_mem = results.get_margeff(at='mean', method='dydx')

# 特定值處的邊際效應
marginal_effects_custom = results.get_margeff(at='mean',
                                              atexog={'x1': 1, 'x2': 5})
```

**預測：**
```python
# 預測機率
probs = results.predict(X)

# 二元預測（0.5 閾值）
predictions = (probs > 0.5).astype(int)

# 自訂閾值
threshold = 0.3
predictions_custom = (probs > threshold).astype(int)

# 新資料
X_new = sm.add_constant(X_new_data)
new_probs = results.predict(X_new)
```

**模型評估：**
```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

# 分類報告
print(classification_report(y, predictions))

# 混淆矩陣
print(confusion_matrix(y, predictions))

# AUC-ROC
auc = roc_auc_score(y, probs)
print(f"AUC: {auc:.4f}")

# 虛擬 R 平方
print(f"McFadden's Pseudo R²: {results.prsquared:.4f}")
```

### Probit

使用常態分布處理二元結果。

**使用時機：**
- 二元結果
- 偏好常態分布假設
- 領域慣例（計量經濟學常用 probit）

**模型**：P(Y=1|X) = Φ(Xβ)，其中 Φ 是標準常態累積分布函數

```python
from statsmodels.discrete.discrete_model import Probit

model = Probit(y, X)
results = model.fit()

print(results.summary())
```

**與 Logit 的比較：**
- Probit 和 Logit 通常給出類似結果
- Probit：對稱，基於常態分布
- Logit：尾部稍重，解釋較容易（勝算比）
- 係數不可直接比較（尺度差異）

```python
# 邊際效應可比較
logit_me = logit_results.get_margeff().margeff
probit_me = probit_results.get_margeff().margeff

print("Logit marginal effects:", logit_me)
print("Probit marginal effects:", probit_me)
```

## 多項模型

### MNLogit（多項邏輯斯迴歸）

用於 3 個以上類別的無序類別結果。

**使用時機：**
- 多個無序類別（例如交通方式、品牌選擇）
- 類別間無自然順序
- 需要每個類別的機率

**模型**：P(Y=j|X) = exp(Xβⱼ) / Σₖ exp(Xβₖ)

```python
from statsmodels.discrete.discrete_model import MNLogit

# y 應為類別的整數 0, 1, 2, ...
model = MNLogit(y, X)
results = model.fit()

print(results.summary())
```

**解釋：**
```python
# 一個類別為參考（通常是類別 0）
# 係數表示相對於參考的對數勝算

# 對於類別 j 對參考：
# exp(β_j) = 類別 j 對參考的勝算比

# 每個類別的預測機率
probs = results.predict(X)  # 形狀：(n_samples, n_categories)

# 最可能的類別
predicted_categories = probs.argmax(axis=1)
```

**相對風險比：**
```python
# 對係數取指數得到相對風險比
import numpy as np
import pandas as pd

# 取得參數名稱和值
params_df = pd.DataFrame({
    'coef': results.params,
    'RRR': np.exp(results.params)
})
print(params_df)
```

### 條件 Logit

用於替代品具有特徵的選擇模型。

**使用時機：**
- 替代品特定回歸變數（跨選擇變化）
- 含選擇的追蹤資料
- 離散選擇實驗

```python
from statsmodels.discrete.conditional_models import ConditionalLogit

# 資料結構：長格式，含選擇指標
model = ConditionalLogit(y_choice, X_alternatives, groups=individual_id)
results = model.fit()
```

## 計數模型

### Poisson

計數資料的標準模型。

**使用時機：**
- 計數結果（事件、發生次數）
- 稀有事件
- 均值 ≈ 變異數

**模型**：P(Y=k|X) = exp(-λ) λᵏ / k!，其中 log(λ) = Xβ

```python
from statsmodels.discrete.count_model import Poisson

model = Poisson(y_counts, X)
results = model.fit()

print(results.summary())
```

**解釋：**
```python
# 率比（發生率比）
rate_ratios = np.exp(results.params)
print("Rate ratios:", rate_ratios)

# X 每增加 1 單位，期望計數乘以 exp(β)
```

**檢查過度離散：**
```python
# Poisson 的均值和變異數應相似
print(f"Mean: {y_counts.mean():.2f}")
print(f"Variance: {y_counts.var():.2f}")

# 正式檢定
from statsmodels.stats.stattools import durbin_watson

# 若變異數 >> 均值則過度離散
# 經驗法則：變異數/均值 > 1.5 表示過度離散
overdispersion_ratio = y_counts.var() / y_counts.mean()
print(f"Variance/Mean: {overdispersion_ratio:.2f}")

if overdispersion_ratio > 1.5:
    print("Consider Negative Binomial model")
```

**含偏移量（用於率）：**
```python
# 當建模具有不同暴露量的率時
# log(λ) = log(exposure) + Xβ

model = Poisson(y_counts, X, offset=np.log(exposure))
results = model.fit()
```

### 負二項（Negative Binomial）

用於過度離散的計數資料（變異數 > 均值）。

**使用時機：**
- 具過度離散的計數資料
- Poisson 無法解釋的過多變異數
- 計數中的異質性

**模型**：添加離散參數 α 以解釋過度離散

```python
from statsmodels.discrete.count_model import NegativeBinomial

model = NegativeBinomial(y_counts, X)
results = model.fit()

print(results.summary())
print(f"Dispersion parameter alpha: {results.params['alpha']:.4f}")
```

**與 Poisson 比較：**
```python
# 配適兩個模型
poisson_results = Poisson(y_counts, X).fit()
nb_results = NegativeBinomial(y_counts, X).fit()

# AIC 比較（較低較佳）
print(f"Poisson AIC: {poisson_results.aic:.2f}")
print(f"Negative Binomial AIC: {nb_results.aic:.2f}")

# 概似比檢定（若 NB 較佳）
from scipy import stats
lr_stat = 2 * (nb_results.llf - poisson_results.llf)
lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)  # 1 個額外參數（alpha）
print(f"LR test p-value: {lr_pval:.4f}")

if lr_pval < 0.05:
    print("Negative Binomial significantly better")
```

### 零膨脹模型

用於具有過多零值的計數資料。

**使用時機：**
- 比 Poisson/NB 預期更多的零值
- 兩個過程：一個產生零值，一個產生計數
- 範例：就醫次數、保險理賠

**模型：**
- ZeroInflatedPoisson（ZIP）
- ZeroInflatedNegativeBinomialP（ZINB）

```python
from statsmodels.discrete.count_model import (ZeroInflatedPoisson,
                                               ZeroInflatedNegativeBinomialP)

# ZIP 模型
zip_model = ZeroInflatedPoisson(y_counts, X, exog_infl=X_inflation)
zip_results = zip_model.fit()

# ZINB 模型（用於過度離散 + 過多零值）
zinb_model = ZeroInflatedNegativeBinomialP(y_counts, X, exog_infl=X_inflation)
zinb_results = zinb_model.fit()

print(zip_results.summary())
```

**模型的兩個部分：**
```python
# 1. 膨脹模型：P(Y=0 由於膨脹)
# 2. 計數模型：計數的分布

# 膨脹的預測機率
inflation_probs = zip_results.predict(X, which='prob')

# 預測計數
predicted_counts = zip_results.predict(X, which='mean')
```

### 障礙模型（Hurdle Models）

兩階段模型：是否有任何計數，然後有多少。

**使用時機：**
- 過多零值
- 零值與正計數有不同過程
- 零值在結構上與正值不同

```python
from statsmodels.discrete.count_model import HurdleCountModel

# 指定計數分布和零膨脹
model = HurdleCountModel(y_counts, X,
                         exog_infl=X_hurdle,
                         dist='poisson')  # 或 'negbin'
results = model.fit()

print(results.summary())
```

## 序數模型

### 有序 Logit/Probit

用於有序類別結果。

**使用時機：**
- 有序類別（例如低/中/高、評分 1-5）
- 自然順序很重要
- 想尊重序數結構

**模型**：具有截點的累積機率模型

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

# y 應為有序整數：0, 1, 2, ...
model = OrderedModel(y_ordered, X, distr='logit')  # 或 'probit'
results = model.fit(method='bfgs')

print(results.summary())
```

**解釋：**
```python
# 截點（類別間的閾值）
cutpoints = results.params[-n_categories+1:]
print("Cutpoints:", cutpoints)

# 係數
coefficients = results.params[:-n_categories+1]
print("Coefficients:", coefficients)

# 每個類別的預測機率
probs = results.predict(X)  # 形狀：(n_samples, n_categories)

# 最可能的類別
predicted_categories = probs.argmax(axis=1)
```

**比例勝算假設：**
```python
# 檢定係數在截點間是否相同
# （Brant 檢定 - 手動實作或檢查殘差）

# 檢查：分別建模每個截點並比較係數
```

## 模型診斷

### 配適度

```python
# 虛擬 R 平方（McFadden）
print(f"Pseudo R²: {results.prsquared:.4f}")

# 模型比較的 AIC/BIC
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")

# 對數概似
print(f"Log-likelihood: {results.llf:.2f}")

# 對虛無模型的概似比檢定
lr_stat = 2 * (results.llf - results.llnull)
from scipy import stats
lr_pval = 1 - stats.chi2.cdf(lr_stat, results.df_model)
print(f"LR test p-value: {lr_pval}")
```

### 分類指標（二元）

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

# 預測
probs = results.predict(X)
predictions = (probs > 0.5).astype(int)

# 指標
print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
print(f"Precision: {precision_score(y, predictions):.4f}")
print(f"Recall: {recall_score(y, predictions):.4f}")
print(f"F1: {f1_score(y, predictions):.4f}")
print(f"AUC: {roc_auc_score(y, probs):.4f}")
```

### 分類指標（多項）

```python
from sklearn.metrics import accuracy_score, classification_report, log_loss

# 預測類別
probs = results.predict(X)
predictions = probs.argmax(axis=1)

# 準確率
accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy:.4f}")

# 分類報告
print(classification_report(y, predictions))

# 對數損失
logloss = log_loss(y, probs)
print(f"Log Loss: {logloss:.4f}")
```

### 計數模型診斷

```python
# 觀察與預測頻率
observed = pd.Series(y_counts).value_counts().sort_index()
predicted = results.predict(X)
predicted_counts = pd.Series(np.round(predicted)).value_counts().sort_index()

# 比較分布
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
observed.plot(kind='bar', alpha=0.5, label='Observed', ax=ax)
predicted_counts.plot(kind='bar', alpha=0.5, label='Predicted', ax=ax)
ax.legend()
ax.set_xlabel('Count')
ax.set_ylabel('Frequency')
plt.show()

# 根圖（Rootogram）（更好的視覺化）
from statsmodels.graphics.agreement import mean_diff_plot
# 需要自訂根圖實作
```

### 影響和離群值

```python
# 標準化殘差
std_resid = (y - results.predict(X)) / np.sqrt(results.predict(X))

# 檢查離群值（|std_resid| > 2）
outliers = np.where(np.abs(std_resid) > 2)[0]
print(f"Number of outliers: {len(outliers)}")

# 槓桿（hat 值）- 用於 logit/probit
# from statsmodels.stats.outliers_influence
```

## 假設檢定

```python
# 單參數檢定（摘要中自動顯示）

# 多參數：Wald 檢定
# 檢定 H0: β₁ = β₂ = 0
R = [[0, 1, 0, 0], [0, 0, 1, 0]]
wald_test = results.wald_test(R)
print(wald_test)

# 巢狀模型的概似比檢定
model_reduced = Logit(y, X_reduced).fit()
model_full = Logit(y, X_full).fit()

lr_stat = 2 * (model_full.llf - model_reduced.llf)
df = model_full.df_model - model_reduced.df_model
from scipy import stats
lr_pval = 1 - stats.chi2.cdf(lr_stat, df)
print(f"LR test p-value: {lr_pval:.4f}")
```

## 模型選擇與比較

```python
# 配適多個模型
models = {
    'Logit': Logit(y, X).fit(),
    'Probit': Probit(y, X).fit(),
    # 添加更多模型
}

# 比較 AIC/BIC
comparison = pd.DataFrame({
    'AIC': {name: model.aic for name, model in models.items()},
    'BIC': {name: model.bic for name, model in models.items()},
    'Pseudo R²': {name: model.prsquared for name, model in models.items()}
})
print(comparison.sort_values('AIC'))

# 預測表現的交叉驗證
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 使用 sklearn 包裝器或手動交叉驗證
```

## 公式 API

使用 R 風格公式以更容易的規格。

```python
import statsmodels.formula.api as smf

# 使用公式的 Logit
formula = 'y ~ x1 + x2 + C(category) + x1:x2'
results = smf.logit(formula, data=df).fit()

# 使用公式的 MNLogit
results = smf.mnlogit(formula, data=df).fit()

# 使用公式的 Poisson
results = smf.poisson(formula, data=df).fit()

# 使用公式的負二項
results = smf.negativebinomial(formula, data=df).fit()
```

## 常見應用

### 二元分類（行銷回應）

```python
# 預測客戶購買機率
X = sm.add_constant(customer_features)
model = Logit(purchased, X)
results = model.fit()

# 目標定位：選擇最可能購買的前 20%
probs = results.predict(X)
top_20_pct_idx = np.argsort(probs)[-int(0.2*len(probs)):]
```

### 多項選擇（交通方式）

```python
# 預測交通方式選擇
model = MNLogit(mode_choice, X)
results = model.fit()

# 新通勤者的預測方式
new_commuter = sm.add_constant(new_features)
mode_probs = results.predict(new_commuter)
predicted_mode = mode_probs.argmax(axis=1)
```

### 計數資料（就醫次數）

```python
# 建模醫療利用
model = NegativeBinomial(num_visits, X)
results = model.fit()

# 新病患的預期就診次數
expected_visits = results.predict(new_patient_X)
```

### 零膨脹（保險理賠）

```python
# 許多人有零理賠
# 零膨脹：有些人永不理賠
# 計數過程：可能理賠的人

zip_model = ZeroInflatedPoisson(claims, X_count, exog_infl=X_inflation)
results = zip_model.fit()

# P(永不提出理賠)
never_claim_prob = results.predict(X, which='prob-zero')

# 預期理賠
expected_claims = results.predict(X, which='mean')
```

## 最佳實務

1. **檢查資料類型**：確保反應變數符合模型（二元、計數、類別）
2. **添加常數**：務必使用 `sm.add_constant()` 除非不要截距
3. **縮放連續預測變數**：改善收斂和解釋
4. **檢查收斂**：注意收斂警告
5. **使用公式 API**：處理類別變數和交互作用
6. **邊際效應**：報告邊際效應，不只是係數
7. **模型比較**：使用 AIC/BIC 和交叉驗證
8. **驗證**：預測模型使用保留集或交叉驗證
9. **檢查過度離散**：計數模型檢定 Poisson 假設
10. **考慮替代方案**：過多零值使用零膨脹、障礙模型

## 常見陷阱

1. **忘記常數**：無截距項
2. **完美分離**：Logit/probit 可能不收斂
3. **過度離散使用 Poisson**：檢查並使用負二項
4. **誤解係數**：記住它們在對數勝算/對數尺度上
5. **未檢查收斂**：最佳化可能靜默失敗
6. **錯誤分布**：配合資料類型（二元/計數/類別）
7. **忽略過多零值**：適當時使用 ZIP/ZINB
8. **未驗證預測**：始終檢查樣本外表現
9. **比較非巢狀模型**：使用 AIC/BIC，不是概似比檢定
10. **序數當名義**：有序類別使用 OrderedModel
