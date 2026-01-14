# 效果量和統計考驗力分析

本文件提供計算、解讀和報告效果量，以及進行統計考驗力分析以規劃研究的指南。

## 為什麼效果量很重要

1. **統計顯著性 ≠ 實際顯著性**：p 值只告訴效果是否存在，而非大小
2. **與樣本大小相關**：大樣本下，瑣碎的效果也會變「顯著」
3. **解讀**：效果量提供大小和實際重要性
4. **統合分析**：效果量使跨研究結果合併成為可能
5. **統計考驗力分析**：確定樣本大小所需

**黃金法則**：始終報告效果量連同 p 值。

---

## 各分析類型的效果量

### T 檢定和平均差異

#### Cohen's d（標準化平均差異）

**公式**：
- 獨立組：d = (M₁ - M₂) / SD_pooled
- 配對組：d = M_diff / SD_diff

**解讀**（Cohen, 1988）：
- 小：|d| = 0.20
- 中等：|d| = 0.50
- 大：|d| = 0.80

**情境相關解讀**：
- 在教育領域：d = 0.40 是成功介入的典型值
- 在心理學領域：d = 0.40 被認為有意義
- 在醫學領域：小效果量可能具有臨床重要性

**Python 計算**：
```python
import pingouin as pg
import numpy as np

# 含效果量的獨立 t 檢定
result = pg.ttest(group1, group2, correction=False)
cohens_d = result['cohen-d'].values[0]

# 手動計算
mean_diff = np.mean(group1) - np.mean(group2)
pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
cohens_d = mean_diff / pooled_std

# 配對 t 檢定
result = pg.ttest(pre, post, paired=True)
cohens_d = result['cohen-d'].values[0]
```

**d 的信賴區間**：
```python
from pingouin import compute_effsize_from_t

d, ci = compute_effsize_from_t(t_statistic, nx=n1, ny=n2, eftype='cohen')
```

---

#### Hedges' g（偏差校正的 d）

**為何使用**：Cohen's d 在小樣本（n < 20）時有輕微向上偏差

**公式**：g = d × correction_factor，其中 correction_factor = 1 - 3/(4df - 1)

**Python 計算**：
```python
result = pg.ttest(group1, group2, correction=False)
hedges_g = result['hedges'].values[0]
```

**使用 Hedges' g 當**：
- 樣本大小小（每組 n < 20）
- 進行統合分析（統合分析的標準）

---

#### Glass's Δ（Delta）

**何時使用**：當一組是具有已知變異性的對照組時

**公式**：Δ = (M₁ - M₂) / SD_control

**使用案例**：
- 臨床試驗（使用對照組 SD）
- 當處理影響變異性時

---

### 變異數分析

#### Eta 平方（η²）

**測量內容**：因子解釋的總變異比例

**公式**：η² = SS_effect / SS_total

**解讀**：
- 小：η² = 0.01（1% 變異）
- 中等：η² = 0.06（6% 變異）
- 大：η² = 0.14（14% 變異）

**限制**：多因子時有偏差（總和 > 1.0）

**Python 計算**：
```python
import pingouin as pg

# 單因子變異數分析
aov = pg.anova(dv='value', between='group', data=df)
eta_squared = aov['SS'][0] / aov['SS'].sum()

# 或直接使用 pingouin
aov = pg.anova(dv='value', between='group', data=df, detailed=True)
eta_squared = aov['np2'][0]  # 注意：pingouin 報告偏 eta 平方
```

---

#### 偏 Eta 平方（η²_p）

**測量內容**：因子解釋的變異比例，排除其他因子

**公式**：η²_p = SS_effect / (SS_effect + SS_error)

**解讀**：與 η² 相同的基準

**何時使用**：多因子變異數分析（因子設計的標準）

**Python 計算**：
```python
aov = pg.anova(dv='value', between=['factor1', 'factor2'], data=df)
# pingouin 預設報告偏 eta 平方
partial_eta_sq = aov['np2']
```

---

#### Omega 平方（ω²）

**測量內容**：母體變異解釋的較少偏差估計

**為何使用**：η² 高估效果量；ω² 提供更好的母體估計

**公式**：ω² = (SS_effect - df_effect × MS_error) / (SS_total + MS_error)

**解讀**：與 η² 相同的基準，但通常值較小

**Python 計算**：
```python
def omega_squared(aov_table):
    ss_effect = aov_table.loc[0, 'SS']
    ss_total = aov_table['SS'].sum()
    ms_error = aov_table.loc[aov_table.index[-1], 'MS']  # 殘差 MS
    df_effect = aov_table.loc[0, 'DF']

    omega_sq = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
    return omega_sq
```

---

#### Cohen's f

**測量內容**：變異數分析的效果量（類似於 Cohen's d）

**公式**：f = √(η² / (1 - η²))

**解讀**：
- 小：f = 0.10
- 中等：f = 0.25
- 大：f = 0.40

**Python 計算**：
```python
eta_squared = 0.06  # 從變異數分析
cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
```

**在統計考驗力分析中使用**：變異數分析統計考驗力計算所需

---

### 相關

#### Pearson's r / Spearman's ρ

**解讀**：
- 小：|r| = 0.10
- 中等：|r| = 0.30
- 大：|r| = 0.50

**重要注意事項**：
- r² = 決定係數（解釋的變異比例）
- r = 0.30 表示 9% 的共享變異（0.30² = 0.09）
- 考慮方向（正/負）和情境

**Python 計算**：
```python
import pingouin as pg

# 含 CI 的 Pearson 相關
result = pg.corr(x, y, method='pearson')
r = result['r'].values[0]
ci = [result['CI95%'][0][0], result['CI95%'][0][1]]

# Spearman 相關
result = pg.corr(x, y, method='spearman')
rho = result['r'].values[0]
```

---

### 迴歸

#### R²（決定係數）

**測量內容**：模型解釋 Y 中變異的比例

**解讀**：
- 小：R² = 0.02
- 中等：R² = 0.13
- 大：R² = 0.26

**情境相關**：
- 物理科學：預期 R² > 0.90
- 社會科學：R² > 0.30 被認為良好
- 行為預測：R² > 0.10 可能有意義

**Python 計算**：
```python
from sklearn.metrics import r2_score
from statsmodels.api import OLS

# 使用 statsmodels
model = OLS(y, X).fit()
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj

# 手動
r_squared = 1 - (SS_residual / SS_total)
```

---

#### 調整後 R²

**為何使用**：添加預測變數時 R² 會人為增加；調整後 R² 懲罰模型複雜度

**公式**：R²_adj = 1 - (1 - R²) × (n - 1) / (n - k - 1)

**何時使用**：多元迴歸始終連同 R² 報告

---

#### 標準化迴歸係數（β）

**測量內容**：預測變數一個 SD 變化對結果（以 SD 為單位）的效果

**解讀**：類似於 Cohen's d
- 小：|β| = 0.10
- 中等：|β| = 0.30
- 大：|β| = 0.50

**Python 計算**：
```python
from scipy import stats

# 先標準化變數
X_std = (X - X.mean()) / X.std()
y_std = (y - y.mean()) / y.std()

model = OLS(y_std, X_std).fit()
beta = model.params
```

---

#### f²（迴歸的 Cohen's f 平方）

**測量內容**：個別預測變數或模型比較的效果量

**公式**：f² = R²_AB - R²_A / (1 - R²_AB)

其中：
- R²_AB = 含預測變數的完整模型 R²
- R²_A = 不含預測變數的簡約模型 R²

**解讀**：
- 小：f² = 0.02
- 中等：f² = 0.15
- 大：f² = 0.35

**Python 計算**：
```python
# 比較兩個巢狀模型
model_full = OLS(y, X_full).fit()
model_reduced = OLS(y, X_reduced).fit()

r2_full = model_full.rsquared
r2_reduced = model_reduced.rsquared

f_squared = (r2_full - r2_reduced) / (1 - r2_full)
```

---

### 類別資料分析

#### Cramér's V

**測量內容**：χ² 檢定的關聯強度（適用於任何表格大小）

**公式**：V = √(χ² / (n × (k - 1)))

其中 k = min(列, 欄)

**解讀**（對於 k > 2）：
- 小：V = 0.07
- 中等：V = 0.21
- 大：V = 0.35

**對於 2×2 表**：使用 phi 係數（φ）

**Python 計算**：
```python
from scipy.stats.contingency import association

# Cramér's V
cramers_v = association(contingency_table, method='cramer')

# Phi 係數（用於 2x2）
phi = association(contingency_table, method='pearson')
```

---

#### 勝算比（OR）和風險比（RR）

**用於 2×2 列聯表**：

|           | 結果 + | 結果 - |
|-----------|--------|--------|
| 暴露      | a      | b      |
| 未暴露    | c      | d      |

**勝算比**：OR = (a/b) / (c/d) = ad / bc

**解讀**：
- OR = 1：無關聯
- OR > 1：正向關聯（勝算增加）
- OR < 1：負向關聯（勝算減少）
- OR = 2：兩倍勝算
- OR = 0.5：一半勝算

**風險比**：RR = (a/(a+b)) / (c/(c+d))

**何時使用**：
- 世代研究：使用 RR（更易解讀）
- 病例-對照研究：使用 OR（無法取得 RR）
- 邏輯斯迴歸：OR 是自然輸出

**Python 計算**：
```python
import statsmodels.api as sm

# 從列聯表
odds_ratio = (a * d) / (b * c)

# 信賴區間
table = np.array([[a, b], [c, d]])
oddsratio, pvalue = stats.fisher_exact(table)

# 從邏輯斯迴歸
model = sm.Logit(y, X).fit()
odds_ratios = np.exp(model.params)  # 指數化係數
ci = np.exp(model.conf_int())  # 指數化 CI
```

---

### 貝氏效果量

#### 貝氏因子（BF）

**測量內容**：對立假設 vs. 虛無假設的證據比率

**解讀**：
- BF₁₀ = 1：H₁ 和 H₀ 的證據相等
- BF₁₀ = 3：H₁ 比 H₀ 可能 3 倍（中度證據）
- BF₁₀ = 10：H₁ 比 H₀ 可能 10 倍（強證據）
- BF₁₀ = 100：H₁ 比 H₀ 可能 100 倍（決定性證據）
- BF₁₀ = 0.33：H₀ 比 H₁ 可能 3 倍
- BF₁₀ = 0.10：H₀ 比 H₁ 可能 10 倍

**分類**（Jeffreys, 1961）：
- 1-3：軼事性證據
- 3-10：中度證據
- 10-30：強證據
- 30-100：非常強證據
- >100：決定性證據

**Python 計算**：
```python
import pingouin as pg

# 貝氏 t 檢定
result = pg.ttest(group1, group2, correction=False)
# 注意：pingouin 不包含 BF；使用其他套件

# 使用 JASP 或 BayesFactor（R）透過 rpy2
# 或使用數值積分實作
```

---

## 統計考驗力分析

### 概念

**統計考驗力**：如果效果存在時偵測到它的機率（1 - β）

**傳統標準**：
- 考驗力 = 0.80（80% 偵測效果的機會）
- α = 0.05（5% 型一誤差率）

**四個相互關聯的參數**（給定 3 個，可求解第 4 個）：
1. 樣本大小（n）
2. 效果量（d、f 等）
3. 顯著水準（α）
4. 考驗力（1 - β）

---

### 事前統計考驗力分析（規劃）

**目的**：在研究前確定所需樣本大小

**步驟**：
1. 指定預期效果量（來自文獻、試驗資料或最小有意義效果）
2. 設定 α 水準（通常 0.05）
3. 設定期望考驗力（通常 0.80）
4. 計算所需 n

**Python 實作**：
```python
from statsmodels.stats.power import (
    tt_ind_solve_power,
    zt_ind_solve_power,
    FTestAnovaPower,
    NormalIndPower
)

# T 檢定統計考驗力分析
n_required = tt_ind_solve_power(
    effect_size=0.5,  # Cohen's d
    alpha=0.05,
    power=0.80,
    ratio=1.0,  # 相等組大小
    alternative='two-sided'
)

# 變異數分析統計考驗力分析
anova_power = FTestAnovaPower()
n_per_group = anova_power.solve_power(
    effect_size=0.25,  # Cohen's f
    ngroups=3,
    alpha=0.05,
    power=0.80
)

# 相關統計考驗力分析
from pingouin import power_corr
n_required = power_corr(r=0.30, power=0.80, alpha=0.05)
```

---

### 事後統計考驗力分析（研究後）

**⚠️ 注意**：事後考驗力有爭議且通常不建議

**為何有問題**：
- 觀察到的考驗力是 p 值的直接函數
- 如果 p > 0.05，考驗力總是低
- 不提供超出 p 值的額外資訊
- 可能誤導

**可能可接受時**：
- 為未來研究規劃
- 使用來自多個研究（不僅僅是您自己的）的效果量
- 明確目標是複製的樣本大小

**更好的替代方案**：
- 報告效果量的信賴區間
- 進行敏感度分析
- 報告最小可偵測效果量

---

### 敏感度分析

**目的**：給定研究參數確定最小可偵測效果量

**何時使用**：研究完成後，了解研究的能力

**Python 實作**：
```python
# 每組 n=50 時我們可以偵測什麼效果量？
detectable_effect = tt_ind_solve_power(
    effect_size=None,  # 求解此項
    nobs1=50,
    alpha=0.05,
    power=0.80,
    ratio=1.0,
    alternative='two-sided'
)

print(f"每組 n=50 時，我們可以偵測 d ≥ {detectable_effect:.2f}")
```

---

## 報告效果量

### APA 風格指南

**T 檢定範例**：
> 「A 組（M = 75.2，SD = 8.5）的分數顯著高於 B 組（M = 68.3，SD = 9.2），t(98) = 3.82，p < .001，d = 0.77，95% CI [0.36, 1.18]。」

**變異數分析範例**：
> 「處理條件對測驗分數有顯著主效果，F(2, 87) = 8.45，p < .001，η²p = .16。使用 Tukey's HSD 的事後比較顯示...」

**相關範例**：
> 「學習時間和考試分數之間有中度正相關，r(148) = .42，p < .001，95% CI [.27, .55]。」

**迴歸範例**：
> 「迴歸模型顯著預測考試分數，F(3, 146) = 45.2，p < .001，R² = .48。學習時數（β = .52，p < .001）和先前 GPA（β = .31，p < .001）是顯著預測變數。」

**貝氏範例**：
> 「貝氏獨立樣本 t 檢定提供了組間差異的強證據，BF₁₀ = 23.5，表示資料在 H₁ 下比 H₀ 下可能 23.5 倍。」

---

## 效果量陷阱

1. **不要只依賴基準**：情境很重要；小效果可能有意義
2. **報告信賴區間**：CI 顯示效果量估計的精確度
3. **區分統計 vs. 實際顯著性**：大 n 可以使瑣碎效果「顯著」
4. **考慮成本-效益**：如果介入成本低，即使小效果也可能有價值
5. **多個結果**：效果量因結果而異；報告所有
6. **不要挑選**：報告所有計畫分析的效果
7. **發表偏差**：發表的效果通常被高估

---

## 快速參考表

| 分析 | 效果量 | 小 | 中等 | 大 |
|----------|-------------|-------|--------|-------|
| T 檢定 | Cohen's d | 0.20 | 0.50 | 0.80 |
| 變異數分析 | η²、ω² | 0.01 | 0.06 | 0.14 |
| 變異數分析 | Cohen's f | 0.10 | 0.25 | 0.40 |
| 相關 | r、ρ | 0.10 | 0.30 | 0.50 |
| 迴歸 | R² | 0.02 | 0.13 | 0.26 |
| 迴歸 | f² | 0.02 | 0.15 | 0.35 |
| 卡方檢定 | Cramér's V | 0.07 | 0.21 | 0.35 |
| 卡方檢定（2×2） | φ | 0.10 | 0.30 | 0.50 |

---

## 資源

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- Lakens, D. (2013). Calculating and reporting effect sizes
- Ellis, P. D. (2010). *The Essential Guide to Effect Sizes*
