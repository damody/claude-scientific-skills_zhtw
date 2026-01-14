# 統計假設和診斷程序

本文件提供檢驗和驗證各種分析統計假設的完整指南。

## 一般原則

1. **在解讀檢定結果前務必檢驗假設**
2. **使用多種診斷方法**（視覺化 + 形式檢定）
3. **考慮穩健性**：某些檢定在特定條件下對違反具有穩健性
4. **記錄所有假設檢驗**於分析報告中
5. **報告違反情況和採取的補救措施**

## 跨檢定的常見假設

### 1. 觀測值獨立性

**意義**：每個觀測值都是獨立的；對一個受試者的測量不會影響對另一個受試者的測量。

**如何檢驗**：
- 檢視研究設計和資料收集程序
- 對於時間序列：檢查自相關（ACF/PACF 圖、Durbin-Watson 檢定）
- 對於群集資料：考慮組內相關係數（ICC）

**違反時該怎麼做**：
- 對群集/階層資料使用混合效果模型
- 對時間相依資料使用時間序列方法
- 對相關資料使用廣義估計方程式（GEE）

**嚴重程度**：高 - 違反可能嚴重膨脹型一誤差

---

### 2. 常態性

**意義**：資料或殘差遵循常態（高斯）分布。

**何時需要**：
- t 檢定（對於小樣本；n > 30 時穩健）
- 變異數分析（對於小樣本；n > 30 時穩健）
- 線性迴歸（對於殘差）
- 某些相關檢定（Pearson）

**如何檢驗**：

**視覺方法**（主要）：
- Q-Q（分位-分位）圖：點應落在對角線上
- 含常態曲線疊加的直方圖
- 核密度圖

**形式檢定**（次要）：
- Shapiro-Wilk 檢定（建議用於 n < 50）
- Kolmogorov-Smirnov 檢定
- Anderson-Darling 檢定

**Python 實作**：
```python
from scipy import stats
import matplotlib.pyplot as plt

# Shapiro-Wilk 檢定
statistic, p_value = stats.shapiro(data)

# Q-Q 圖
stats.probplot(data, dist="norm", plot=plt)
```

**解讀指南**：
- 對於 n < 30：視覺和形式檢定都很重要
- 對於 30 ≤ n < 100：視覺檢查為主，形式檢定為輔
- 對於 n ≥ 100：形式檢定過度敏感；依賴視覺檢查
- 注意嚴重偏態、離群值或雙峰

**違反時該怎麼做**：
- **輕微違反**（輕微偏態）：如果每組 n > 30 則繼續
- **中度違反**：使用無母數替代方法（Mann-Whitney、Kruskal-Wallis、Wilcoxon）
- **嚴重違反**：
  - 轉換資料（對數、平方根、Box-Cox）
  - 使用無母數方法
  - 使用穩健迴歸方法
  - 考慮拔靴法

**嚴重程度**：中等 - 在足夠樣本量下，母數檢定通常對輕微違反具有穩健性

---

### 3. 變異數同質性（同方差性）

**意義**：各組間或預測變數範圍內的變異數相等。

**何時需要**：
- 獨立樣本 t 檢定
- 變異數分析
- 線性迴歸（殘差變異數恆定）

**如何檢驗**：

**視覺方法**（主要）：
- 分組箱形圖（用於 t 檢定/變異數分析）
- 殘差 vs. 擬合值圖（用於迴歸）- 應顯示隨機散佈
- 尺度-位置圖（標準化殘差的平方根 vs. 擬合值）

**形式檢定**（次要）：
- Levene 檢定（對非常態性穩健）
- Bartlett 檢定（對非常態性敏感，不建議）
- Brown-Forsythe 檢定（基於中位數的 Levene 版本）
- Breusch-Pagan 檢定（用於迴歸）

**Python 實作**：
```python
from scipy import stats
import pingouin as pg

# Levene 檢定
statistic, p_value = stats.levene(group1, group2, group3)

# 對於迴歸
# Breusch-Pagan 檢定
from statsmodels.stats.diagnostic import het_breuschpagan
_, p_value, _, _ = het_breuschpagan(residuals, exog)
```

**解讀指南**：
- 變異數比（最大/最小）< 2-3：通常可接受
- 對於變異數分析：如果組別大小相等，檢定具有穩健性
- 對於迴歸：在殘差圖中尋找漏斗形態

**違反時該怎麼做**：
- **t 檢定**：使用 Welch t 檢定（不假設變異數相等）
- **變異數分析**：使用 Welch 變異數分析或 Brown-Forsythe 變異數分析
- **迴歸**：
  - 轉換依變數（對數、平方根）
  - 使用加權最小平方法（WLS）
  - 使用穩健標準誤（HC3）
  - 使用具有適當變異數函數的廣義線性模型（GLM）

**嚴重程度**：中等 - 在樣本大小相等時，檢定可以是穩健的

---

## 特定檢定的假設

### T 檢定

**假設**：
1. 觀測值獨立性
2. 常態性（獨立 t 檢定的每組；配對 t 檢定的差異）
3. 變異數同質性（僅獨立 t 檢定）

**診斷工作流程**：
```python
import scipy.stats as stats
import pingouin as pg

# 檢驗每組的常態性
stats.shapiro(group1)
stats.shapiro(group2)

# 檢驗變異數同質性
stats.levene(group1, group2)

# 如果假設被違反：
# 選項 1：Welch t 檢定（變異數不等）
pg.ttest(group1, group2, correction=False)  # Welch's

# 選項 2：無母數替代方法
pg.mwu(group1, group2)  # Mann-Whitney U
```

---

### 變異數分析

**假設**：
1. 組內和組間觀測值獨立性
2. 每組的常態性
3. 跨組的變異數同質性

**額外考量**：
- 對於重複測量變異數分析：球形性假設（Mauchly 檢定）

**診斷工作流程**：
```python
import pingouin as pg

# 檢驗每組的常態性
for group in df['group'].unique():
    data = df[df['group'] == group]['value']
    stats.shapiro(data)

# 檢驗變異數同質性
pg.homoscedasticity(df, dv='value', group='group')

# 對於重複測量：檢驗球形性
# 在 pingouin 的 rm_anova 中自動檢定
```

**球形性違反時該怎麼做**（重複測量）：
- Greenhouse-Geisser 校正（ε < 0.75）
- Huynh-Feldt 校正（ε > 0.75）
- 使用多變量方法（MANOVA）

---

### 線性迴歸

**假設**：
1. **線性**：X 和 Y 之間的關係是線性的
2. **獨立性**：殘差是獨立的
3. **同方差性**：殘差變異數恆定
4. **常態性**：殘差呈常態分布
5. **無多重共線性**：預測變數之間不高度相關（多元迴歸）

**診斷工作流程**：

**1. 線性**：
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Y vs 每個 X 的散佈圖
# 殘差 vs. 擬合值（應隨機散佈）
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='r', linestyle='--')
```

**2. 獨立性**：
```python
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson 檢定（用於時間序列）
dw_statistic = durbin_watson(residuals)
# 值在 1.5-2.5 之間表示獨立
```

**3. 同方差性**：
```python
# Breusch-Pagan 檢定
from statsmodels.stats.diagnostic import het_breuschpagan
_, p_value, _, _ = het_breuschpagan(residuals, exog)

# 視覺：尺度-位置圖
plt.scatter(fitted_values, np.sqrt(np.abs(std_residuals)))
```

**4. 殘差常態性**：
```python
# 殘差的 Q-Q 圖
stats.probplot(residuals, dist="norm", plot=plt)

# Shapiro-Wilk 檢定
stats.shapiro(residuals)
```

**5. 多重共線性**：
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 計算每個預測變數的 VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# VIF > 10 表示嚴重多重共線性
# VIF > 5 表示中度多重共線性
```

**違反時該怎麼做**：
- **非線性**：添加多項式項，使用 GAM，或轉換變數
- **異方差性**：轉換 Y，使用 WLS，使用穩健 SE
- **非常態殘差**：轉換 Y，使用穩健方法，檢查離群值
- **多重共線性**：移除相關預測變數，使用 PCA，脊迴歸

---

### 邏輯斯迴歸

**假設**：
1. **獨立性**：觀測值是獨立的
2. **線性**：對數勝算與連續預測變數之間呈線性關係
3. **無完美多重共線性**：預測變數之間不完美相關
4. **大樣本量**：每個預測變數至少 10-20 個事件

**診斷工作流程**：

**1. 對數的線性**：
```python
# Box-Tidwell 檢定：添加與連續預測變數對數的交互作用
# 如果交互作用顯著，則線性被違反
```

**2. 多重共線性**：
```python
# 使用與線性迴歸相同的 VIF
```

**3. 影響觀測值**：
```python
# Cook's distance、DFBetas、槓桿
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(model)
cooks_d = influence.cooks_distance
```

**4. 模型配適**：
```python
# Hosmer-Lemeshow 檢定
# 虛擬 R 平方
# 分類指標（準確率、AUC-ROC）
```

---

## 離群值偵測

**方法**：
1. **視覺**：箱形圖、散佈圖
2. **統計**：
   - Z 分數：|z| > 3 表示離群值
   - IQR 方法：值 < Q1 - 1.5×IQR 或 > Q3 + 1.5×IQR
   - 使用中位數絕對偏差的修正 Z 分數（對離群值穩健）

**對於迴歸**：
- **槓桿**：高槓桿點（帽值）
- **影響**：Cook's distance > 4/n 表示影響點
- **離群值**：學生化殘差 > ±3

**該怎麼做**：
1. 調查資料輸入錯誤
2. 考慮離群值是否為有效觀測值
3. 報告敏感度分析（含和不含離群值的結果）
4. 如果離群值是合理的，使用穩健方法

---

## 樣本大小考量

### 最小樣本大小（經驗法則）

- **T 檢定**：每組 n ≥ 30 以對非常態性穩健
- **變異數分析**：每組 n ≥ 30
- **相關**：n ≥ 30 以獲得足夠統計考驗力
- **簡單迴歸**：n ≥ 50
- **多元迴歸**：每個預測變數 n ≥ 10-20（最少 10 + k 個預測變數）
- **邏輯斯迴歸**：每個預測變數 n ≥ 10-20 個事件

### 小樣本考量

對於小樣本：
- 假設變得更加關鍵
- 可用時使用精確檢定（Fisher 精確檢定、精確邏輯斯迴歸）
- 考慮無母數替代方法
- 使用排列檢定或拔靴法
- 解讀時保守

---

## 報告假設檢驗

報告分析時，包括：

1. **檢驗的假設陳述**：列出所有檢驗的假設
2. **使用的方法**：描述採用的視覺和形式檢定
3. **診斷檢定結果**：報告檢定統計量和 p 值
4. **評估**：陳述假設是否滿足或被違反
5. **採取的行動**：如果違反，描述補救措施（轉換、替代檢定、穩健方法）

**報告陳述範例**：
> 「使用 Shapiro-Wilk 檢定和 Q-Q 圖評估常態性。A 組（W = 0.97，p = .18）和 B 組（W = 0.96，p = .12）的資料未顯示顯著偏離常態性。使用 Levene 檢定評估變異數同質性，結果不顯著（F(1, 58) = 1.23，p = .27），表示各組變異數相等。因此，獨立樣本 t 檢定的假設獲得滿足。」
