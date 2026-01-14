# PyMC 分布參考

此參考提供 PyMC 中可用機率分布的完整目錄，按類別組織。建構貝氏模型時使用此參考選擇適當的先驗和概似函數分布。

## 連續分布

連續分布定義實值域上的機率密度。

### 常見連續分布

**`pm.Normal(name, mu, sigma)`**
- 常態（高斯）分布（Normal/Gaussian distribution）
- 參數：`mu`（平均值）、`sigma`（標準差）
- 支持：(-∞, ∞)
- 常見用途：無界參數的預設先驗、具有加性雜訊的連續資料概似函數

**`pm.HalfNormal(name, sigma)`**
- 半常態分布（Half-normal distribution，常態分布的正半部分）
- 參數：`sigma`（標準差）
- 支持：[0, ∞)
- 常見用途：尺度/標準差參數的先驗

**`pm.Uniform(name, lower, upper)`**
- 均勻分布（Uniform distribution）
- 參數：`lower`、`upper`（邊界）
- 支持：[lower, upper]
- 常見用途：參數必須有界時的弱資訊先驗

**`pm.Beta(name, alpha, beta)`**
- Beta 分布
- 參數：`alpha`、`beta`（形狀參數）
- 支持：[0, 1]
- 常見用途：機率和比例的先驗

**`pm.Gamma(name, alpha, beta)`**
- Gamma 分布
- 參數：`alpha`（形狀）、`beta`（率）
- 支持：(0, ∞)
- 常見用途：正值參數、率參數的先驗

**`pm.Exponential(name, lam)`**
- 指數分布（Exponential distribution）
- 參數：`lam`（率參數）
- 支持：[0, ∞)
- 常見用途：尺度參數、等待時間的先驗

**`pm.LogNormal(name, mu, sigma)`**
- 對數常態分布（Log-normal distribution）
- 參數：`mu`、`sigma`（底層常態分布的參數）
- 支持：(0, ∞)
- 常見用途：具有乘法效應的正值參數先驗

**`pm.StudentT(name, nu, mu, sigma)`**
- Student's t 分布
- 參數：`nu`（自由度）、`mu`（位置）、`sigma`（尺度）
- 支持：(-∞, ∞)
- 常見用途：對離群值穩健模型的常態分布替代

**`pm.Cauchy(name, alpha, beta)`**
- 柯西分布（Cauchy distribution）
- 參數：`alpha`（位置）、`beta`（尺度）
- 支持：(-∞, ∞)
- 常見用途：常態分布的重尾替代

### 特殊連續分布

**`pm.Laplace(name, mu, b)`** - 拉普拉斯（雙指數）分布（Laplace/double exponential distribution）

**`pm.AsymmetricLaplace(name, kappa, mu, b)`** - 非對稱拉普拉斯分布（Asymmetric Laplace distribution）

**`pm.InverseGamma(name, alpha, beta)`** - 逆 Gamma 分布（Inverse gamma distribution）

**`pm.Weibull(name, alpha, beta)`** - Weibull 分布，用於可靠性分析

**`pm.Logistic(name, mu, s)`** - 邏輯分布（Logistic distribution）

**`pm.LogitNormal(name, mu, sigma)`** - Logit-常態分布，支持 (0,1)

**`pm.Pareto(name, alpha, m)`** - 帕雷托分布（Pareto distribution），用於冪律現象

**`pm.ChiSquared(name, nu)`** - 卡方分布（Chi-squared distribution）

**`pm.ExGaussian(name, mu, sigma, nu)`** - 指數修正高斯分布（Exponentially modified Gaussian）

**`pm.VonMises(name, mu, kappa)`** - Von Mises 分布（圓形常態分布）

**`pm.SkewNormal(name, mu, sigma, alpha)`** - 偏態常態分布（Skew-normal distribution）

**`pm.Triangular(name, lower, c, upper)`** - 三角分布（Triangular distribution）

**`pm.Gumbel(name, mu, beta)`** - Gumbel 分布，用於極值

**`pm.Rice(name, nu, sigma)`** - Rice（Rician）分布

**`pm.Moyal(name, mu, sigma)`** - Moyal 分布

**`pm.Kumaraswamy(name, a, b)`** - Kumaraswamy 分布（Beta 替代）

**`pm.Interpolated(name, x_points, pdf_points)`** - 從插值建立的自訂分布

## 離散分布

離散分布定義整數值域上的機率。

### 常見離散分布

**`pm.Bernoulli(name, p)`**
- 伯努利分布（Bernoulli distribution，二元結果）
- 參數：`p`（成功機率）
- 支持：{0, 1}
- 常見用途：二元分類、擲硬幣

**`pm.Binomial(name, n, p)`**
- 二項分布（Binomial distribution）
- 參數：`n`（試驗次數）、`p`（成功機率）
- 支持：{0, 1, ..., n}
- 常見用途：固定試驗中的成功次數

**`pm.Poisson(name, mu)`**
- 卜瓦松分布（Poisson distribution）
- 參數：`mu`（率參數）
- 支持：{0, 1, 2, ...}
- 常見用途：計數資料、率、發生次數

**`pm.Categorical(name, p)`**
- 類別分布（Categorical distribution）
- 參數：`p`（機率向量）
- 支持：{0, 1, ..., K-1}
- 常見用途：多類別分類

**`pm.DiscreteUniform(name, lower, upper)`**
- 離散均勻分布（Discrete uniform distribution）
- 參數：`lower`、`upper`（邊界）
- 支持：{lower, ..., upper}
- 常見用途：有限整數上的均勻先驗

**`pm.NegativeBinomial(name, mu, alpha)`**
- 負二項分布（Negative binomial distribution）
- 參數：`mu`（平均值）、`alpha`（離散度）
- 支持：{0, 1, 2, ...}
- 常見用途：過度離散的計數資料

**`pm.Geometric(name, p)`**
- 幾何分布（Geometric distribution）
- 參數：`p`（成功機率）
- 支持：{0, 1, 2, ...}
- 常見用途：首次成功前的失敗次數

### 特殊離散分布

**`pm.BetaBinomial(name, alpha, beta, n)`** - Beta-二項分布（過度離散的二項分布）

**`pm.HyperGeometric(name, N, k, n)`** - 超幾何分布（Hypergeometric distribution）

**`pm.DiscreteWeibull(name, q, beta)`** - 離散 Weibull 分布

**`pm.OrderedLogistic(name, eta, cutpoints)`** - 有序邏輯分布，用於有序資料

**`pm.OrderedProbit(name, eta, cutpoints)`** - 有序 Probit 分布，用於有序資料

## 多變量分布

多變量分布定義向量值隨機變數的聯合機率分布。

### 常見多變量分布

**`pm.MvNormal(name, mu, cov)`**
- 多變量常態分布（Multivariate normal distribution）
- 參數：`mu`（平均向量）、`cov`（共變異數矩陣）
- 常見用途：相關連續變數、高斯過程

**`pm.Dirichlet(name, a)`**
- 狄利克雷分布（Dirichlet distribution）
- 參數：`a`（集中參數）
- 支持：單純形（和為 1）
- 常見用途：機率向量的先驗、主題模型

**`pm.Multinomial(name, n, p)`**
- 多項分布（Multinomial distribution）
- 參數：`n`（試驗次數）、`p`（機率向量）
- 常見用途：多類別的計數資料

**`pm.MvStudentT(name, nu, mu, cov)`**
- 多變量 Student's t 分布
- 參數：`nu`（自由度）、`mu`（位置）、`cov`（尺度矩陣）
- 常見用途：穩健的多變量建模

### 特殊多變量分布

**`pm.LKJCorr(name, n, eta)`** - LKJ 相關矩陣先驗（用於相關矩陣）

**`pm.LKJCholeskyCov(name, n, eta, sd_dist)`** - 具有 Cholesky 分解的 LKJ 先驗

**`pm.Wishart(name, nu, V)`** - Wishart 分布（用於共變異數矩陣）

**`pm.InverseWishart(name, nu, V)`** - 逆 Wishart 分布

**`pm.MatrixNormal(name, mu, rowcov, colcov)`** - 矩陣常態分布

**`pm.KroneckerNormal(name, mu, covs, sigma)`** - Kronecker 結構常態分布

**`pm.CAR(name, mu, W, alpha, tau)`** - 條件自迴歸分布（空間）

**`pm.ICAR(name, W, sigma)`** - 內在條件自迴歸分布（空間）

## 混合分布

混合分布組合多個成分分布。

**`pm.Mixture(name, w, comp_dists)`**
- 一般混合分布
- 參數：`w`（權重）、`comp_dists`（成分分布）
- 常見用途：聚類、多模態資料

**`pm.NormalMixture(name, w, mu, sigma)`**
- 常態分布混合
- 常見用途：高斯混合聚類

### 零膨脹和門檻模型

**`pm.ZeroInflatedPoisson(name, psi, mu)`** - 計數資料中的過多零值

**`pm.ZeroInflatedBinomial(name, psi, n, p)`** - 零膨脹二項分布

**`pm.ZeroInflatedNegativeBinomial(name, psi, mu, alpha)`** - 零膨脹負二項分布

**`pm.HurdlePoisson(name, psi, mu)`** - 門檻卜瓦松分布（兩部分模型）

**`pm.HurdleGamma(name, psi, alpha, beta)`** - 門檻 Gamma 分布

**`pm.HurdleLogNormal(name, psi, mu, sigma)`** - 門檻對數常態分布

## 時間序列分布

專為時序資料和序列建模設計的分布。

**`pm.AR(name, rho, sigma, init_dist)`**
- 自迴歸過程（Autoregressive process）
- 參數：`rho`（AR 係數）、`sigma`（創新標準差）、`init_dist`（初始分布）
- 常見用途：時間序列建模、序列資料

**`pm.GaussianRandomWalk(name, mu, sigma, init_dist)`**
- 高斯隨機漫步（Gaussian random walk）
- 參數：`mu`（漂移）、`sigma`（步長）、`init_dist`（初始值）
- 常見用途：累積過程、隨機漫步先驗

**`pm.MvGaussianRandomWalk(name, mu, cov, init_dist)`**
- 多變量高斯隨機漫步

**`pm.GARCH11(name, omega, alpha_1, beta_1)`**
- GARCH(1,1) 波動率模型
- 常見用途：金融時間序列、波動率建模

**`pm.EulerMaruyama(name, dt, sde_fn, sde_pars, init_dist)`**
- 透過 Euler-Maruyama 離散化的隨機微分方程
- 常見用途：連續時間過程

## 特殊分布

**`pm.Deterministic(name, var)`**
- 確定性轉換（非隨機變數）
- 用於從其他變數衍生的計算量

**`pm.Potential(name, logp)`**
- 添加任意對數機率貢獻
- 用於自訂概似成分或約束

**`pm.Flat(name)`**
- 不適當的均勻先驗（常數密度）
- 謹慎使用；可能導致抽樣問題

**`pm.HalfFlat(name)`**
- 正實數上的不適當均勻先驗
- 謹慎使用；可能導致抽樣問題

## 分布修飾符

**`pm.Truncated(name, dist, lower, upper)`**
- 將任何分布截斷到指定邊界

**`pm.Censored(name, dist, lower, upper)`**
- 處理設限觀察（觀察到邊界而非精確值）

**`pm.CustomDist(name, ..., logp, random)`**
- 使用使用者指定的對數機率和隨機抽樣函數定義自訂分布

**`pm.Simulator(name, fn, params, ...)`**
- 透過模擬的自訂分布（用於無概似推斷）

## 使用提示

### 選擇先驗

1. **尺度參數**（σ、τ）：使用 `HalfNormal`、`HalfCauchy`、`Exponential` 或 `Gamma`
2. **機率**：使用 `Beta` 或 `Uniform(0, 1)`
3. **無界參數**：使用 `Normal` 或 `StudentT`（用於穩健性）
4. **正值參數**：使用 `LogNormal`、`Gamma` 或 `Exponential`
5. **相關矩陣**：使用 `LKJCorr`
6. **計數資料**：使用 `Poisson` 或 `NegativeBinomial`（用於過度離散）

### 形狀廣播

PyMC 分布支援 NumPy 風格的廣播。使用 `shape` 參數建立隨機變數的向量或陣列：

```python
# 5 個獨立常態分布的向量
beta = pm.Normal('beta', mu=0, sigma=1, shape=5)

# 3x4 獨立 Gamma 分布的矩陣
tau = pm.Gamma('tau', alpha=2, beta=1, shape=(3, 4))
```

### 使用 dims 命名維度

使用 `dims` 而非 shape 使模型更易讀：

```python
with pm.Model(coords={'predictors': ['age', 'income', 'education']}) as model:
    beta = pm.Normal('beta', mu=0, sigma=1, dims='predictors')
```
