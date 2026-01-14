# 貝氏統計分析

本文件提供進行和解讀貝氏統計分析的指南，這是頻率學派（古典）統計的替代框架。

## 貝氏與頻率學派哲學

### 基本差異

| 面向 | 頻率學派 | 貝氏 |
|--------|-------------|----------|
| **機率解讀** | 事件的長期頻率 | 信念/不確定性程度 |
| **參數** | 固定但未知 | 具有分布的隨機變數 |
| **推論** | 基於抽樣分布 | 基於後驗分布 |
| **主要輸出** | p 值、信賴區間 | 後驗機率、可信區間 |
| **先驗資訊** | 不正式納入 | 透過先驗分布明確納入 |
| **假設檢定** | 拒絕/未能拒絕虛無假設 | 給定資料下假設的機率 |
| **樣本大小** | 通常需要最小值 | 可用於任何樣本大小 |
| **解讀** | 間接（給定 H₀ 下資料的機率） | 直接（給定資料下假設的機率） |

### 關鍵問題差異

**頻率學派**：「如果虛無假設為真，觀察到這麼極端或更極端資料的機率是多少？」

**貝氏**：「給定觀察到的資料，假設為真的機率是多少？」

貝氏問題更直觀，直接回答研究者想知道的問題。

---

## 貝氏定理

**公式**：
```
P(θ|D) = P(D|θ) × P(θ) / P(D)
```

**白話說明**：
```
後驗 = 概似 × 先驗 / 證據
```

其中：
- **θ（theta）**：感興趣的參數（例如，平均差異、相關）
- **D**：觀察到的資料
- **P(θ|D)**：後驗分布（看到資料後對 θ 的信念）
- **P(D|θ)**：概似函數（給定 θ 下資料的機率）
- **P(θ)**：先驗分布（看到資料前對 θ 的信念）
- **P(D)**：邊際概似/證據（正規化常數）

---

## 先驗分布

### 先驗類型

#### 1. 訊息性先驗

**何時使用**：當您有來自以下來源的實質先驗知識：
- 先前研究
- 專家知識
- 理論
- 試驗資料

**範例**：統合分析顯示效果量 d ≈ 0.40，SD = 0.15
- 先驗：Normal(0.40, 0.15)

**優點**：
- 納入現有知識
- 更有效率（需要較小樣本）
- 可以用小資料穩定估計

**缺點**：
- 主觀（但主觀性可以是優勢）
- 必須合理且透明
- 如果強先驗與資料衝突可能有爭議

---

#### 2. 弱訊息性先驗

**何時使用**：大多數應用的預設選擇

**特性**：
- 正則化估計（防止極端值）
- 在中等資料下對後驗影響最小
- 防止計算問題

**先驗範例**：
- 效果量：Normal(0, 1) 或 Cauchy(0, 0.707)
- 變異數：Half-Cauchy(0, 1)
- 相關：Uniform(-1, 1) 或 Beta(2, 2)

**優點**：
- 平衡客觀性和正則化
- 計算穩定
- 廣泛可接受

---

#### 3. 無訊息性（平坦/均勻）先驗

**何時使用**：當試圖「客觀」時

**範例**：Uniform(-∞, ∞) 對任何值

**⚠️ 注意**：
- 可能導致不適當的後驗
- 可能產生不合理的結果
- 不是真正「無訊息性」的（仍然做出假設）
- 在現代貝氏實踐中通常不建議

**更好的替代方案**：使用弱訊息性先驗

---

### 先驗敏感度分析

**務必進行**：測試結果如何隨不同先驗變化

**過程**：
1. 用預設/計畫的先驗擬合模型
2. 用更分散的先驗擬合模型
3. 用更集中的先驗擬合模型
4. 比較後驗分布

**報告**：
- 如果結果相似：證據是穩健的
- 如果結果有實質差異：資料不夠強以壓倒先驗

**Python 範例**：
```python
import pymc as pm

# 使用不同先驗的模型
priors = [
    ('weakly_informative', pm.Normal.dist(0, 1)),
    ('diffuse', pm.Normal.dist(0, 10)),
    ('informative', pm.Normal.dist(0.5, 0.3))
]

results = {}
for name, prior in priors:
    with pm.Model():
        effect = pm.Normal('effect', mu=prior.mu, sigma=prior.sigma)
        # ... 模型其餘部分
        trace = pm.sample()
        results[name] = trace
```

---

## 貝氏假設檢定

### 貝氏因子（BF）

**定義**：兩個競爭假設的證據比率

**公式**：
```
BF₁₀ = P(D|H₁) / P(D|H₀)
```

**解讀**：

| BF₁₀ | 證據 |
|------|----------|
| >100 | 決定性支持 H₁ |
| 30-100 | 非常強支持 H₁ |
| 10-30 | 強支持 H₁ |
| 3-10 | 中度支持 H₁ |
| 1-3 | 軼事性支持 H₁ |
| 1 | 無證據 |
| 1/3-1 | 軼事性支持 H₀ |
| 1/10-1/3 | 中度支持 H₀ |
| 1/30-1/10 | 強支持 H₀ |
| 1/100-1/30 | 非常強支持 H₀ |
| <1/100 | 決定性支持 H₀ |

**相對於 p 值的優勢**：
1. 可以提供虛無假設的證據
2. 不依賴抽樣意圖（沒有「偷看」問題）
3. 直接量化證據
4. 可以用更多資料更新

**Python 計算**：
```python
import pingouin as pg

# 注意：Python 中 BF 支援有限
# 更好的選項：R 套件（BayesFactor）、JASP 軟體

# 從 t 統計量近似 BF
# 使用 Jeffreys-Zellner-Siow 先驗
from scipy import stats

def bf_from_t(t, n1, n2, r_scale=0.707):
    """
    從 t 統計量近似貝氏因子
    r_scale：Cauchy 先驗尺度（預設 0.707 為中等效果）
    """
    # 這是簡化版；使用專用套件獲得精確計算
    df = n1 + n2 - 2
    # 實作需要數值積分
    pass
```

---

### 實際等價區間（ROPE）

**目的**：定義可忽略效果量的範圍

**過程**：
1. 定義 ROPE（例如，d ∈ [-0.1, 0.1] 為可忽略效果）
2. 計算後驗在 ROPE 內的百分比
3. 做出決策：
   - >95% 在 ROPE 內：接受實際等價
   - >95% 在 ROPE 外：拒絕等價
   - 否則：無定論

**優勢**：直接檢定實際顯著性

**Python 範例**：
```python
# 定義 ROPE
rope_lower, rope_upper = -0.1, 0.1

# 計算後驗在 ROPE 內的百分比
in_rope = np.mean((posterior_samples > rope_lower) &
                  (posterior_samples < rope_upper))

print(f"{in_rope*100:.1f}% 後驗在 ROPE 內")
```

---

## 貝氏估計

### 可信區間

**定義**：包含參數機率為 X% 的區間

**95% 可信區間解讀**：
> 「真實參數落在此區間內的機率為 95%。」

**這正是人們認為信賴區間的意義**（但在頻率學派框架中不是）

**類型**：

#### 等尾區間（ETI）
- 第 2.5 百分位到第 97.5 百分位
- 計算簡單
- 對偏斜分布可能不包含眾數

#### 最高密度區間（HDI）
- 包含 95% 分布的最窄區間
- 始終包含眾數
- 對偏斜分布更好

**Python 計算**：
```python
import arviz as az

# 等尾區間
eti = np.percentile(posterior_samples, [2.5, 97.5])

# HDI
hdi = az.hdi(posterior_samples, hdi_prob=0.95)
```

---

### 後驗分布

**解讀後驗分布**：

1. **集中趨勢**：
   - 平均值：後驗平均值
   - 中位數：第 50 百分位
   - 眾數：最可能值（MAP - 最大後驗估計）

2. **不確定性**：
   - SD：後驗的離散程度
   - 可信區間：量化不確定性

3. **形狀**：
   - 對稱：類似常態
   - 偏斜：不對稱不確定性
   - 多峰：多個可能值

**視覺化**：
```python
import matplotlib.pyplot as plt
import arviz as az

# 含 HDI 的後驗圖
az.plot_posterior(trace, hdi_prob=0.95)

# 軌跡圖（檢查收斂）
az.plot_trace(trace)

# 森林圖（多個參數）
az.plot_forest(trace)
```

---

## 常見貝氏分析

### 貝氏 T 檢定

**目的**：比較兩組（t 檢定的貝氏替代方法）

**輸出**：
1. 平均差異的後驗分布
2. 95% 可信區間
3. 貝氏因子（BF₁₀）
4. 方向假設的機率（例如，P(μ₁ > μ₂)）

**Python 實作**：
```python
import pymc as pm
import arviz as az

# 貝氏獨立樣本 t 檢定
with pm.Model() as model:
    # 組平均值的先驗
    mu1 = pm.Normal('mu1', mu=0, sigma=10)
    mu2 = pm.Normal('mu2', mu=0, sigma=10)

    # 合併標準差的先驗
    sigma = pm.HalfNormal('sigma', sigma=10)

    # 概似函數
    y1 = pm.Normal('y1', mu=mu1, sigma=sigma, observed=group1)
    y2 = pm.Normal('y2', mu=mu2, sigma=sigma, observed=group2)

    # 衍生量：平均差異
    diff = pm.Deterministic('diff', mu1 - mu2)

    # 抽樣後驗
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# 分析結果
print(az.summary(trace, var_names=['mu1', 'mu2', 'diff']))

# group1 > group2 的機率
prob_greater = np.mean(trace.posterior['diff'].values > 0)
print(f"P(μ₁ > μ₂) = {prob_greater:.3f}")

# 繪製後驗
az.plot_posterior(trace, var_names=['diff'], ref_val=0)
```

---

### 貝氏變異數分析

**目的**：比較三組以上

**模型**：
```python
import pymc as pm

with pm.Model() as anova_model:
    # 超先驗
    mu_global = pm.Normal('mu_global', mu=0, sigma=10)
    sigma_between = pm.HalfNormal('sigma_between', sigma=5)
    sigma_within = pm.HalfNormal('sigma_within', sigma=5)

    # 組平均值（階層式）
    group_means = pm.Normal('group_means',
                            mu=mu_global,
                            sigma=sigma_between,
                            shape=n_groups)

    # 概似函數
    y = pm.Normal('y',
                  mu=group_means[group_idx],
                  sigma=sigma_within,
                  observed=data)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# 後驗對比
contrast_1_2 = trace.posterior['group_means'][:,:,0] - trace.posterior['group_means'][:,:,1]
```

---

### 貝氏相關

**目的**：估計兩個變數之間的相關

**優勢**：提供相關值的分布

**Python 實作**：
```python
import pymc as pm

with pm.Model() as corr_model:
    # 相關的先驗
    rho = pm.Uniform('rho', lower=-1, upper=1)

    # 轉換為共變異數矩陣
    cov_matrix = pm.math.stack([[1, rho],
                                [rho, 1]])

    # 概似函數（雙變量常態）
    obs = pm.MvNormal('obs',
                     mu=[0, 0],
                     cov=cov_matrix,
                     observed=np.column_stack([x, y]))

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# 摘要相關
print(az.summary(trace, var_names=['rho']))

# 相關為正的機率
prob_positive = np.mean(trace.posterior['rho'].values > 0)
```

---

### 貝氏線性迴歸

**目的**：建立預測變數和結果之間的關係模型

**優勢**：
- 所有參數的不確定性
- 自然正則化（透過先驗）
- 可以納入先驗知識
- 預測的可信區間

**Python 實作**：
```python
import pymc as pm

with pm.Model() as regression_model:
    # 係數的先驗
    alpha = pm.Normal('alpha', mu=0, sigma=10)  # 截距
    beta = pm.Normal('beta', mu=0, sigma=10, shape=n_predictors)
    sigma = pm.HalfNormal('sigma', sigma=10)

    # 期望值
    mu = alpha + pm.math.dot(X, beta)

    # 概似函數
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# 後驗預測檢驗
with regression_model:
    ppc = pm.sample_posterior_predictive(trace)

az.plot_ppc(ppc)

# 含不確定性的預測
with regression_model:
    pm.set_data({'X': X_new})
    posterior_pred = pm.sample_posterior_predictive(trace)
```

---

## 階層（多層次）模型

**何時使用**：
- 巢狀/群集資料（學校中的學生）
- 重複測量
- 統合分析
- 跨組的變化效果

**關鍵概念**：部分合併
- 完全合併：忽略組別（有偏）
- 無合併：分別分析組別（高變異）
- 部分合併：跨組借用強度（貝氏）

**範例：變化截距**：
```python
with pm.Model() as hierarchical_model:
    # 超先驗
    mu_global = pm.Normal('mu_global', mu=0, sigma=10)
    sigma_between = pm.HalfNormal('sigma_between', sigma=5)
    sigma_within = pm.HalfNormal('sigma_within', sigma=5)

    # 組層級截距
    alpha = pm.Normal('alpha',
                     mu=mu_global,
                     sigma=sigma_between,
                     shape=n_groups)

    # 概似函數
    y_obs = pm.Normal('y_obs',
                     mu=alpha[group_idx],
                     sigma=sigma_within,
                     observed=y)

    trace = pm.sample()
```

---

## 模型比較

### 方法

#### 1. 貝氏因子
- 直接比較模型證據
- 對先驗設定敏感
- 計算可能密集

#### 2. 資訊準則

**WAIC（廣泛適用資訊準則）**：
- AIC 的貝氏類比
- 越低越好
- 考慮有效參數數量

**LOO（留一法交叉驗證）**：
- 估計樣本外預測誤差
- 越低越好
- 比 WAIC 更穩健

**Python 計算**：
```python
import arviz as az

# 計算 WAIC 和 LOO
waic = az.waic(trace)
loo = az.loo(trace)

print(f"WAIC: {waic.elpd_waic:.2f}")
print(f"LOO: {loo.elpd_loo:.2f}")

# 比較多個模型
comparison = az.compare({
    'model1': trace1,
    'model2': trace2,
    'model3': trace3
})
print(comparison)
```

---

## 檢驗貝氏模型

### 1. 收斂診斷

**R-hat（Gelman-Rubin 統計量）**：
- 比較鏈內和鏈間變異
- 接近 1.0 的值表示收斂
- R-hat < 1.01：良好
- R-hat > 1.05：收斂不良

**有效樣本大小（ESS）**：
- 獨立樣本數量
- 越高越好
- 建議每條鏈 ESS > 400

**軌跡圖**：
- 應該看起來像「模糊的毛毛蟲」
- 沒有趨勢，沒有卡住的鏈

**Python 檢驗**：
```python
# 含診斷的自動摘要
print(az.summary(trace, var_names=['parameter']))

# 視覺診斷
az.plot_trace(trace)
az.plot_rank(trace)  # 等級圖
```

---

### 2. 後驗預測檢驗

**目的**：模型是否產生與觀察資料相似的資料？

**過程**：
1. 從後驗產生預測
2. 與實際資料比較
3. 尋找系統性差異

**Python 實作**：
```python
with model:
    ppc = pm.sample_posterior_predictive(trace)

# 視覺檢驗
az.plot_ppc(ppc, num_pp_samples=100)

# 定量檢驗
obs_mean = np.mean(observed_data)
pred_means = [np.mean(sample) for sample in ppc.posterior_predictive['y_obs']]
p_value = np.mean(pred_means >= obs_mean)  # 貝氏 p 值
```

---

## 報告貝氏結果

### T 檢定報告範例

> 「進行貝氏獨立樣本 t 檢定以比較 A 組和 B 組。使用弱訊息性先驗：平均差異為 Normal(0, 1)，合併標準差為 Half-Cauchy(0, 1)。平均差異的後驗分布平均值為 5.2（95% CI [2.3, 8.1]），表示 A 組分數高於 B 組。貝氏因子 BF₁₀ = 23.5 提供了組間差異的強證據，A 組平均值超過 B 組平均值的機率為 99.7%。」

### 迴歸報告範例

> 「使用弱訊息性先驗（係數為 Normal(0, 10)，殘差 SD 為 Half-Cauchy(0, 5)）擬合貝氏線性迴歸。模型解釋了相當大的變異（R² = 0.47，95% CI [0.38, 0.55]）。學習時數（β = 0.52，95% CI [0.38, 0.66]）和先前 GPA（β = 0.31，95% CI [0.17, 0.45]）是可信的預測變數（95% CI 排除零）。後驗預測檢驗顯示模型配適良好。收斂診斷令人滿意（所有 R-hat < 1.01，ESS > 1000）。」

---

## 優勢和限制

### 優勢

1. **直觀解讀**：關於參數的直接機率陳述
2. **納入先驗知識**：使用所有可用資訊
3. **靈活**：輕鬆處理複雜模型
4. **沒有 p-hacking**：可以在資料到達時查看
5. **量化不確定性**：完整的後驗分布
6. **小樣本**：可用於任何樣本大小

### 限制

1. **計算**：需要 MCMC 抽樣（可能很慢）
2. **先驗設定**：需要思考和合理化
3. **複雜性**：學習曲線較陡
4. **軟體**：工具比頻率學派方法少
5. **溝通**：可能需要教育審查者/讀者

---

## 主要 Python 套件

- **PyMC**：完整的貝氏建模框架
- **ArviZ**：視覺化和診斷
- **Bambi**：迴歸模型的高層介面
- **PyStan**：Stan 的 Python 介面
- **TensorFlow Probability**：使用 TensorFlow 的貝氏推論

---

## 何時使用貝氏方法

**使用貝氏當**：
- 您有先驗資訊要納入
- 您想要直接的機率陳述
- 樣本大小小
- 模型複雜（階層式、缺失資料等）
- 您想要在資料到達時更新分析

**頻率學派可能足夠當**：
- 大樣本的標準分析
- 沒有先驗資訊
- 計算資源有限
- 審查者不熟悉貝氏方法
