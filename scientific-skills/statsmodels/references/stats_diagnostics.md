# 統計檢定和診斷參考

本文件提供 statsmodels 中統計檢定、診斷和工具的完整指引。

## 概述

Statsmodels 提供廣泛的統計檢定功能：
- 殘差診斷和規格檢定
- 假設檢定（母數和無母數）
- 配適度檢定
- 多重比較和事後檢定
- 檢定力和樣本大小計算
- 穩健共變異數矩陣
- 影響和離群值偵測

## 殘差診斷

### 自相關檢定

**Ljung-Box 檢定**：檢定殘差中的自相關

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# 檢定殘差的自相關
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(lb_test)

# H0：直到滯後 k 無自相關
# 若 p 值 < 0.05，拒絕 H0（存在自相關）
```

**Durbin-Watson 檢定**：檢定一階自相關

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson: {dw_stat:.4f}")

# DW ≈ 2：無自相關
# DW < 2：正自相關
# DW > 2：負自相關
# 精確臨界值取決於 n 和 k
```

**Breusch-Godfrey 檢定**：更一般的自相關檢定

```python
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

bg_test = acorr_breusch_godfrey(results, nlags=5)
lm_stat, lm_pval, f_stat, f_pval = bg_test

print(f"LM statistic: {lm_stat:.4f}, p-value: {lm_pval:.4f}")
# H0：直到滯後 k 無自相關
```

### 異質變異數檢定

**Breusch-Pagan 檢定**：檢定異質變異數

```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(residuals, exog)
lm_stat, lm_pval, f_stat, f_pval = bp_test

print(f"Breusch-Pagan test p-value: {lm_pval:.4f}")
# H0：同質變異數（變異數恆定）
# 若 p 值 < 0.05，拒絕 H0（存在異質變異數）
```

**White 檢定**：更一般的異質變異數檢定

```python
from statsmodels.stats.diagnostic import het_white

white_test = het_white(residuals, exog)
lm_stat, lm_pval, f_stat, f_pval = white_test

print(f"White test p-value: {lm_pval:.4f}")
# H0：同質變異數
```

**ARCH 檢定**：檢定自迴歸條件異質變異數

```python
from statsmodels.stats.diagnostic import het_arch

arch_test = het_arch(residuals, nlags=5)
lm_stat, lm_pval, f_stat, f_pval = arch_test

print(f"ARCH test p-value: {lm_pval:.4f}")
# H0：無 ARCH 效應
# 若顯著，考慮 GARCH 模型
```

### 常態性檢定

**Jarque-Bera 檢定**：使用偏態和峰度檢定常態性

```python
from statsmodels.stats.stattools import jarque_bera

jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)

print(f"Jarque-Bera statistic: {jb_stat:.4f}")
print(f"p-value: {jb_pval:.4f}")
print(f"Skewness: {skew:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# H0：殘差呈常態分布
# 常態：偏態 ≈ 0，峰度 ≈ 3
```

**Omnibus 檢定**：另一個常態性檢定（同樣基於偏態/峰度）

```python
from statsmodels.stats.stattools import omni_normtest

omni_stat, omni_pval = omni_normtest(residuals)
print(f"Omnibus test p-value: {omni_pval:.4f}")
# H0：常態性
```

**Anderson-Darling 檢定**：分布配適檢定

```python
from statsmodels.stats.diagnostic import normal_ad

ad_stat, ad_pval = normal_ad(residuals)
print(f"Anderson-Darling test p-value: {ad_pval:.4f}")
```

**Lilliefors 檢定**：修改的 Kolmogorov-Smirnov 檢定

```python
from statsmodels.stats.diagnostic import lilliefors

lf_stat, lf_pval = lilliefors(residuals, dist='norm')
print(f"Lilliefors test p-value: {lf_pval:.4f}")
```

### 線性和規格檢定

**Ramsey RESET 檢定**：檢定函數形式誤設

```python
from statsmodels.stats.diagnostic import linear_reset

reset_test = linear_reset(results, power=2)
f_stat, f_pval = reset_test

print(f"RESET test p-value: {f_pval:.4f}")
# H0：模型正確指定（線性）
# 若被拒絕，可能需要多項式項或轉換
```

**Harvey-Collier 檢定**：檢定線性

```python
from statsmodels.stats.diagnostic import linear_harvey_collier

hc_stat, hc_pval = linear_harvey_collier(results)
print(f"Harvey-Collier test p-value: {hc_pval:.4f}")
# H0：線性規格正確
```

## 多重共線性偵測

**變異數膨脹因子（VIF）**：

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# 計算每個變數的 VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print(vif_data.sort_values('VIF', ascending=False))

# 解釋：
# VIF = 1：與其他預測變數無相關
# VIF > 5：中度多重共線性
# VIF > 10：嚴重多重共線性問題
# VIF > 20：嚴重多重共線性（考慮移除變數）
```

**條件數**：從迴歸結果

```python
print(f"Condition number: {results.condition_number:.2f}")

# 解釋：
# < 10：無多重共線性擔憂
# 10-30：中度多重共線性
# > 30：強多重共線性
# > 100：嚴重多重共線性
```

## 影響和離群值偵測

### 槓桿

高槓桿點具有極端的預測變數值。

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = results.get_influence()

# Hat 值（槓桿）
leverage = influence.hat_matrix_diag

# 經驗法則：槓桿 > 2*p/n 或 3*p/n 為高
# p = 參數數，n = 樣本大小
threshold = 2 * len(results.params) / len(y)
high_leverage = np.where(leverage > threshold)[0]

print(f"High leverage observations: {high_leverage}")
```

### Cook's Distance

測量每個觀測值的整體影響。

```python
# Cook's distance
cooks_d = influence.cooks_distance[0]

# 經驗法則：Cook's D > 4/n 為有影響
threshold = 4 / len(y)
influential = np.where(cooks_d > threshold)[0]

print(f"Influential observations (Cook's D): {influential}")

# 繪圖
import matplotlib.pyplot as plt
plt.stem(range(len(cooks_d)), cooks_d)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (4/n)')
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.legend()
plt.show()
```

### DFFITS

測量對擬合值的影響。

```python
# DFFITS
dffits = influence.dffits[0]

# 經驗法則：|DFFITS| > 2*sqrt(p/n) 為有影響
p = len(results.params)
n = len(y)
threshold = 2 * np.sqrt(p / n)

influential_dffits = np.where(np.abs(dffits) > threshold)[0]
print(f"Influential observations (DFFITS): {influential_dffits}")
```

### DFBETAs

測量對每個係數的影響。

```python
# DFBETAs（每個參數一個）
dfbetas = influence.dfbetas

# 經驗法則：|DFBETA| > 2/sqrt(n)
threshold = 2 / np.sqrt(n)

for i, param_name in enumerate(results.params.index):
    influential = np.where(np.abs(dfbetas[:, i]) > threshold)[0]
    if len(influential) > 0:
        print(f"Influential for {param_name}: {influential}")
```

### 影響圖

```python
from statsmodels.graphics.regressionplots import influence_plot

fig, ax = plt.subplots(figsize=(12, 8))
influence_plot(results, ax=ax, criterion='cooks')
plt.show()

# 結合槓桿、殘差和 Cook's distance
# 大氣泡 = 高 Cook's distance
# 遠離 x=0 = 高槓桿
# 遠離 y=0 = 大殘差
```

### 學生化殘差

```python
# 學生化殘差（離群值）
student_resid = influence.resid_studentized_internal

# 外部學生化殘差（更保守）
student_resid_external = influence.resid_studentized_external

# 離群值：|學生化殘差| > 3（或 > 2.5）
outliers = np.where(np.abs(student_resid_external) > 3)[0]
print(f"Outliers: {outliers}")
```

## 假設檢定

### t 檢定

**單樣本 t 檢定**：檢定均值是否等於特定值

```python
from scipy import stats

# H0：母體均值 = mu_0
t_stat, p_value = stats.ttest_1samp(data, popmean=mu_0)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**雙樣本 t 檢定**：比較兩組的均值

```python
# H0：mean1 = mean2（變異數相等）
t_stat, p_value = stats.ttest_ind(group1, group2)

# Welch's t 檢定（變異數不等）
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**配對 t 檢定**：比較配對觀測值

```python
# H0：均值差 = 0
t_stat, p_value = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### 比例檢定

**單比例檢定**：

```python
from statsmodels.stats.proportion import proportions_ztest

# H0：比例 = p0
count = 45  # 成功數
nobs = 100  # 總觀測數
p0 = 0.5    # 假設比例

z_stat, p_value = proportions_ztest(count, nobs, value=p0)

print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**雙比例檢定**：

```python
# H0：proportion1 = proportion2
counts = [45, 60]
nobs = [100, 120]

z_stat, p_value = proportions_ztest(counts, nobs)
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### 卡方檢定

**獨立性卡方檢定**：

```python
from scipy.stats import chi2_contingency

# 列聯表
contingency_table = pd.crosstab(variable1, variable2)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# H0：變數間獨立
```

**配適度卡方檢定**：

```python
from scipy.stats import chisquare

# 觀察頻率
observed = [20, 30, 25, 25]

# 期望頻率（預設相等）
expected = [25, 25, 25, 25]

chi2, p_value = chisquare(observed, expected)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")

# H0：資料遵循期望分布
```

### 無母數檢定

**Mann-Whitney U 檢定**（獨立樣本）：

```python
from scipy.stats import mannwhitneyu

# H0：分布相等
u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

print(f"U statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**Wilcoxon 符號等級檢定**（配對樣本）：

```python
from scipy.stats import wilcoxon

# H0：中位數差 = 0
w_stat, p_value = wilcoxon(before, after)

print(f"W statistic: {w_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**Kruskal-Wallis H 檢定**（>2 組）：

```python
from scipy.stats import kruskal

# H0：所有組別具有相同分布
h_stat, p_value = kruskal(group1, group2, group3)

print(f"H statistic: {h_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**符號檢定**：

```python
from statsmodels.stats.descriptivestats import sign_test

# H0：中位數 = m0
result = sign_test(data, m0=0)
print(result)
```

### 變異數分析（ANOVA）

**單因子變異數分析**：

```python
from scipy.stats import f_oneway

# H0：所有組別均值相等
f_stat, p_value = f_oneway(group1, group2, group3)

print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

**雙因子變異數分析**（使用 statsmodels）：

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 配適模型
model = ols('response ~ C(factor1) + C(factor2) + C(factor1):C(factor2)',
            data=df).fit()

# ANOVA 表
anova_table = anova_lm(model, typ=2)
print(anova_table)
```

**重複測量變異數分析**：

```python
from statsmodels.stats.anova import AnovaRM

# 需要長格式資料
aovrm = AnovaRM(df, depvar='score', subject='subject_id', within=['time'])
results = aovrm.fit()

print(results.summary())
```

## 多重比較

### 事後檢定

**Tukey HSD**（Honest Significant Difference）：

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 執行 Tukey HSD 檢定
tukey = pairwise_tukeyhsd(data, groups, alpha=0.05)

print(tukey.summary())

# 繪製信賴區間
tukey.plot_simultaneous()
plt.show()
```

**Bonferroni 校正**：

```python
from statsmodels.stats.multitest import multipletests

# 多重檢定的 P 值
p_values = [0.01, 0.03, 0.04, 0.15, 0.001]

# 應用校正
reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
    p_values,
    alpha=0.05,
    method='bonferroni'
)

print("Rejected:", reject)
print("Corrected p-values:", pvals_corrected)
```

**偽發現率（FDR）**：

```python
# FDR 校正（比 Bonferroni 較不保守）
reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'  # Benjamini-Hochberg
)

print("Rejected:", reject)
print("Corrected p-values:", pvals_corrected)
```

## 穩健共變異數矩陣

### 異質變異數一致（HC）標準誤

```python
# 配適 OLS 後
results = sm.OLS(y, X).fit()

# HC0（White 異質變異數一致標準誤）
results_hc0 = results.get_robustcov_results(cov_type='HC0')

# HC1（自由度調整）
results_hc1 = results.get_robustcov_results(cov_type='HC1')

# HC2（槓桿調整）
results_hc2 = results.get_robustcov_results(cov_type='HC2')

# HC3（最保守，建議用於小樣本）
results_hc3 = results.get_robustcov_results(cov_type='HC3')

print("Standard OLS SEs:", results.bse)
print("Robust HC3 SEs:", results_hc3.bse)
```

### HAC（異質變異數和自相關一致）

**Newey-West 標準誤**：

```python
# 用於具有自相關和異質變異數的時間序列
results_hac = results.get_robustcov_results(cov_type='HAC', maxlags=4)

print("HAC (Newey-West) SEs:", results_hac.bse)
print(results_hac.summary())
```

### 群集穩健標準誤

```python
# 用於群集/分組資料
results_cluster = results.get_robustcov_results(
    cov_type='cluster',
    groups=cluster_ids
)

print("Cluster-robust SEs:", results_cluster.bse)
```

## 描述統計

**基本描述統計**：

```python
from statsmodels.stats.api import DescrStatsW

# 完整描述統計
desc = DescrStatsW(data)

print("Mean:", desc.mean)
print("Std Dev:", desc.std)
print("Variance:", desc.var)
print("Confidence interval:", desc.tconfint_mean())

# 分位數
print("Median:", desc.quantile(0.5))
print("IQR:", desc.quantile([0.25, 0.75]))
```

**加權統計**：

```python
# 帶權重
desc_weighted = DescrStatsW(data, weights=weights)

print("Weighted mean:", desc_weighted.mean)
print("Weighted std:", desc_weighted.std)
```

**比較兩組**：

```python
from statsmodels.stats.weightstats import CompareMeans

# 建立比較物件
cm = CompareMeans(DescrStatsW(group1), DescrStatsW(group2))

# t 檢定
print("t-test:", cm.ttest_ind())

# 差異的信賴區間
print("CI for difference:", cm.tconfint_diff())

# 變異數相等檢定
print("Equal variance test:", cm.test_equal_var())
```

## 檢定力分析和樣本大小

**t 檢定的檢定力**：

```python
from statsmodels.stats.power import tt_ind_solve_power

# 求解樣本大小
effect_size = 0.5  # Cohen's d
alpha = 0.05
power = 0.8

n = tt_ind_solve_power(effect_size=effect_size,
                        alpha=alpha,
                        power=power,
                        alternative='two-sided')

print(f"Required sample size per group: {n:.0f}")

# 給定 n 求解檢定力
power = tt_ind_solve_power(effect_size=0.5,
                           nobs1=50,
                           alpha=0.05,
                           alternative='two-sided')

print(f"Power: {power:.4f}")
```

**比例檢定的檢定力**：

```python
from statsmodels.stats.power import zt_ind_solve_power

# 用於比例檢定（z 檢定）
effect_size = 0.3  # 比例差
alpha = 0.05
power = 0.8

n = zt_ind_solve_power(effect_size=effect_size,
                        alpha=alpha,
                        power=power,
                        alternative='two-sided')

print(f"Required sample size per group: {n:.0f}")
```

**檢定力曲線**：

```python
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt

# 建立檢定力分析物件
analysis = TTestIndPower()

# 繪製不同樣本大小的檢定力曲線
sample_sizes = range(10, 200, 10)
effect_sizes = [0.2, 0.5, 0.8]  # 小、中、大

fig, ax = plt.subplots(figsize=(10, 6))

for es in effect_sizes:
    power = [analysis.solve_power(effect_size=es, nobs1=n, alpha=0.05)
             for n in sample_sizes]
    ax.plot(sample_sizes, power, label=f'Effect size = {es}')

ax.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8')
ax.set_xlabel('Sample size per group')
ax.set_ylabel('Power')
ax.set_title('Power Curves for Two-Sample t-test')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## 效果量

**Cohen's d**（標準化均值差）：

```python
def cohens_d(group1, group2):
    """計算獨立樣本的 Cohen's d"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 合併標準差
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d

d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.4f}")

# 解釋：
# |d| < 0.2：可忽略
# |d| ~ 0.2：小
# |d| ~ 0.5：中
# |d| ~ 0.8：大
```

**Eta 平方**（用於 ANOVA）：

```python
# 從 ANOVA 表
# η² = SS_between / SS_total

def eta_squared(anova_table):
    return anova_table['sum_sq'][0] / anova_table['sum_sq'].sum()

# 執行 ANOVA 後
eta_sq = eta_squared(anova_table)
print(f"Eta-squared: {eta_sq:.4f}")

# 解釋：
# 0.01：小效果
# 0.06：中效果
# 0.14：大效果
```

## 列聯表和關聯

**McNemar 檢定**（配對二元資料）：

```python
from statsmodels.stats.contingency_tables import mcnemar

# 2x2 列聯表
table = [[a, b],
         [c, d]]

result = mcnemar(table, exact=True)  # 或大樣本用 exact=False
print(f"p-value: {result.pvalue:.4f}")

# H0：邊際機率相等
```

**Cochran-Mantel-Haenszel 檢定**：

```python
from statsmodels.stats.contingency_tables import StratifiedTable

# 用於分層 2x2 表
strat_table = StratifiedTable(tables_list)
result = strat_table.test_null_odds()

print(f"p-value: {result.pvalue:.4f}")
```

## 處理效應和因果推論

**傾向分數配對**：

```python
from statsmodels.treatment import propensity_score

# 估計傾向分數
ps_model = sm.Logit(treatment, X).fit()
propensity_scores = ps_model.predict(X)

# 用於配對或加權
# （需要手動實作配對）
```

**雙重差分法**：

```python
# Did 公式：outcome ~ treatment * post
model = ols('outcome ~ treatment + post + treatment:post', data=df).fit()

# DiD 估計是交互作用係數
did_estimate = model.params['treatment:post']
print(f"DiD estimate: {did_estimate:.4f}")
```

## 最佳實務

1. **始終檢查假設**：解釋結果前進行檢定
2. **報告效果量**：不只是 p 值
3. **使用適當檢定**：配合資料類型和分布
4. **多重比較校正**：進行多個檢定時
5. **檢查樣本大小**：確保足夠檢定力
6. **視覺檢查**：檢定前繪製資料
7. **報告信賴區間**：連同點估計
8. **考慮替代方案**：假設違反時使用無母數
9. **穩健標準誤**：當存在異質變異數/自相關時使用
10. **記錄決策**：註記使用哪些檢定及原因

## 常見陷阱

1. **未檢查檢定假設**：可能使結果無效
2. **多重檢定未校正**：膨脹型一誤差
3. **對非常態資料使用母數檢定**：考慮無母數
4. **忽略異質變異數**：使用穩健標準誤
5. **混淆統計顯著性和實務顯著性**：檢查效果量
6. **未報告信賴區間**：只有 p 值不夠
7. **使用錯誤檢定**：配合研究問題
8. **檢定力不足**：型二誤差風險（偽陰性）
9. **p 值操弄（p-hacking）**：測試多種規格直到顯著
10. **過度解釋 p 值**：記住 NHST 的限制
