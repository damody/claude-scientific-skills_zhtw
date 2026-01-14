# Pymoo 約束與決策參考

pymoo 中約束處理和多準則決策的參考。

## 約束處理

### 定義約束

約束在 Problem 定義中指定：

```python
from pymoo.core.problem import ElementwiseProblem
import numpy as np

class ConstrainedProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=2,    # 不等式約束數量
            n_eq_constr=1,      # 等式約束數量
            xl=np.array([0, 0]),
            xu=np.array([5, 5])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # 目標
        f1 = x[0]**2 + x[1]**2
        f2 = (x[0]-1)**2 + (x[1]-1)**2

        out["F"] = [f1, f2]

        # 不等式約束（公式為 g(x) <= 0）
        g1 = x[0] + x[1] - 5  # x[0] + x[1] >= 5 → -(x[0] + x[1] - 5) <= 0
        g2 = x[0]**2 + x[1]**2 - 25  # x[0]^2 + x[1]^2 <= 25

        out["G"] = [g1, g2]

        # 等式約束（公式為 h(x) = 0）
        h1 = x[0] - 2*x[1]

        out["H"] = [h1]
```

**約束公式規則：**
- 不等式：`g(x) <= 0`（負值或零時可行）
- 等式：`h(x) = 0`（為零時可行）
- 將 `g(x) >= 0` 轉換為 `-g(x) <= 0`

### 約束處理技術

#### 1. 可行性優先（預設）
**機制：** 始終偏好可行解而非不可行解
**比較：**
1. 兩者都可行 → 按目標值比較
2. 一個可行，一個不可行 → 可行者勝出
3. 兩者都不可行 → 按約束違反比較

**用法：**
```python
from pymoo.algorithms.moo.nsga2 import NSGA2

# 可行性優先是大多數演算法的預設
algorithm = NSGA2(pop_size=100)
```

**優點：**
- 適用於任何基於排序的演算法
- 簡單有效
- 無需參數調整

**缺點：**
- 可行區域小時可能困難
- 可能忽略好的不可行解

#### 2. 懲罰方法
**機制：** 根據約束違反向目標添加懲罰
**公式：** `F_penalized = F + penalty_factor * violation`

**用法：**
```python
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.constraints.as_penalty import ConstraintsAsPenalty

# 用懲罰包裝問題
problem_with_penalty = ConstraintsAsPenalty(problem, penalty=1e6)

algorithm = GA(pop_size=100)
```

**參數：**
- `penalty`：懲罰係數（根據問題尺度調整）

**優點：**
- 將約束問題轉換為無約束問題
- 適用於任何最佳化演算法

**缺點：**
- 懲罰參數敏感
- 可能需要針對問題調整

#### 3. 約束作為目標
**機制：** 將約束違反視為額外目標
**結果：** 具有 M+1 個目標的多目標問題（M 個原始目標 + 約束）

**用法：**
```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.constraints.as_obj import ConstraintsAsObjective

# 將約束違反作為目標添加
problem_with_cv_obj = ConstraintsAsObjective(problem)

algorithm = NSGA2(pop_size=100)
```

**優點：**
- 無需參數調整
- 保留可能有用的不可行解
- 可行區域小時效果好

**缺點：**
- 增加問題維度
- 更複雜的 Pareto 前沿分析

#### 4. Epsilon 約束處理
**機制：** 動態可行性閾值
**概念：** 隨著世代逐漸收緊約束容差

**優點：**
- 平滑過渡到可行區域
- 有助於處理困難的約束地形

**缺點：**
- 特定演算法的實作
- 需要參數調整

#### 5. 修復運算子
**機制：** 修改不可行解以滿足約束
**應用：** 交叉/突變後修復子代

**用法：**
```python
from pymoo.core.repair import Repair

class MyRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # 將 X 投影到可行區域
        # 範例：裁剪到邊界
        X = np.clip(X, problem.xl, problem.xu)
        return X

from pymoo.algorithms.soo.nonconvex.ga import GA

algorithm = GA(pop_size=100, repair=MyRepair())
```

**優點：**
- 在整個最佳化過程中維持可行性
- 可以編碼領域知識

**缺點：**
- 需要針對問題的實作
- 可能限制搜尋

### 約束處理演算法

一些演算法有內建的約束處理：

#### SRES（隨機排序演化策略）
**用途：** 單目標約束最佳化
**機制：** 隨機排序平衡目標和約束

**用法：**
```python
from pymoo.algorithms.soo.nonconvex.sres import SRES

algorithm = SRES()
```

#### ISRES（改進的 SRES）
**用途：** 增強的約束最佳化
**改進：** 更好的參數適應

**用法：**
```python
from pymoo.algorithms.soo.nonconvex.isres import ISRES

algorithm = ISRES()
```

### 約束處理指南

**根據以下選擇技術：**

| 問題特徵 | 建議技術 |
|------------------------|----------------------|
| 大的可行區域 | 可行性優先 |
| 小的可行區域 | 約束作為目標、修復 |
| 重度約束 | SRES/ISRES、Epsilon 約束 |
| 線性約束 | 修復（投影） |
| 非線性約束 | 可行性優先、懲罰 |
| 已知可行解 | 有偏差的初始化 |

## 多準則決策（MCDM）

獲得 Pareto 前沿後，MCDM 幫助選擇偏好的解。

### 決策背景

**Pareto 前沿特徵：**
- 多個非支配解
- 每個代表不同的權衡
- 沒有客觀上「最佳」的解
- 需要決策者偏好

### Pymoo 中的 MCDM 方法

#### 1. 偽權重
**概念：** 對每個目標加權，選擇最小化加權和的解
**公式：** `score = w1*f1 + w2*f2 + ... + wM*fM`

**用法：**
```python
from pymoo.mcdm.pseudo_weights import PseudoWeights

# 定義權重（必須總和為 1）
weights = np.array([0.3, 0.7])  # f1 權重 30%，f2 權重 70%

dm = PseudoWeights(weights)
best_idx = dm.do(result.F)
best_solution = result.X[best_idx]
```

**何時使用：**
- 有明確的偏好表達
- 目標可比較
- 可接受線性權衡

**限制：**
- 需要指定權重
- 線性假設可能無法捕捉偏好
- 對目標縮放敏感

#### 2. 妥協規劃
**概念：** 選擇最接近理想點的解
**度量：** 到理想點的距離（例如歐氏、Tchebycheff）

**用法：**
```python
from pymoo.mcdm.compromise_programming import CompromiseProgramming

dm = CompromiseProgramming()
best_idx = dm.do(result.F, ideal=ideal_point, nadir=nadir_point)
```

**何時使用：**
- 已知或可估計理想目標值
- 平衡考慮所有目標
- 沒有明確的權重偏好

#### 3. 互動式決策
**概念：** 迭代偏好細化
**過程：**
1. 向決策者展示代表性解
2. 收集偏好回饋
3. 將搜尋集中在偏好區域
4. 重複直到找到滿意的解

**方法：**
- 參考點方法
- 權衡分析
- 漸進式偏好表達

### 決策工作流程

**步驟 1：正規化目標**
```python
# 正規化到 [0, 1] 以進行公平比較
F_norm = (result.F - result.F.min(axis=0)) / (result.F.max(axis=0) - result.F.min(axis=0))
```

**步驟 2：分析權衡**
```python
from pymoo.visualization.scatter import Scatter

plot = Scatter()
plot.add(result.F)
plot.show()

# 識別膝點、極端解
```

**步驟 3：應用 MCDM 方法**
```python
from pymoo.mcdm.pseudo_weights import PseudoWeights

weights = np.array([0.4, 0.6])  # 根據偏好
dm = PseudoWeights(weights)
selected = dm.do(F_norm)
```

**步驟 4：驗證選擇**
```python
# 視覺化選定的解
from pymoo.visualization.petal import Petal

plot = Petal()
plot.add(result.F[selected], label="選定")
# 添加其他候選解以進行比較
plot.show()
```

### 進階 MCDM 技術

#### 膝點偵測
**概念：** 一個目標的小改進會導致其他目標大幅退化的解

**用法：**
```python
from pymoo.mcdm.knee import KneePoint

km = KneePoint()
knee_idx = km.do(result.F)
knee_solutions = result.X[knee_idx]
```

**何時使用：**
- 沒有明確偏好
- 需要平衡的權衡
- 凸 Pareto 前沿

#### 超體積貢獻
**概念：** 選擇對超體積貢獻最大的解
**用例：** 維持多樣化的解子集

**用法：**
```python
from pymoo.indicators.hv import HV

hv = HV(ref_point=reference_point)
hv_contributions = hv.calc_contributions(result.F)

# 選擇貢獻最大的
top_k = 5
top_indices = np.argsort(hv_contributions)[-top_k:]
selected_solutions = result.X[top_indices]
```

### 決策指南

**當決策者有：**

| 偏好資訊 | 建議方法 |
|------------------------|-------------------|
| 明確的目標權重 | 偽權重 |
| 理想目標值 | 妥協規劃 |
| 沒有先驗偏好 | 膝點、視覺檢查 |
| 衝突準則 | 互動方法 |
| 需要多樣化子集 | 超體積貢獻 |

**最佳實務：**
1. **MCDM 前正規化目標**
2. **視覺化 Pareto 前沿**以了解權衡
3. **考慮多種方法**以進行穩健選擇
4. **與領域專家驗證結果**
5. **記錄假設**和偏好來源
6. **對權重/參數進行敏感度分析**

### 整合範例

包含約束處理和決策的完整工作流程：

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.mcdm.pseudo_weights import PseudoWeights
import numpy as np

# 定義約束問題
problem = MyConstrainedProblem()

# 設定具有可行性優先約束處理的演算法
algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)

# 最佳化
result = minimize(
    problem,
    algorithm,
    ('n_gen', 200),
    seed=1,
    verbose=True
)

# 只篩選可行解
feasible_mask = result.CV[:, 0] == 0  # 約束違反 = 0
F_feasible = result.F[feasible_mask]
X_feasible = result.X[feasible_mask]

# 正規化目標
F_norm = (F_feasible - F_feasible.min(axis=0)) / (F_feasible.max(axis=0) - F_feasible.min(axis=0))

# 應用 MCDM
weights = np.array([0.5, 0.5])
dm = PseudoWeights(weights)
best_idx = dm.do(F_norm)

# 獲取最終解
best_solution = X_feasible[best_idx]
best_objectives = F_feasible[best_idx]

print(f"選定的解：{best_solution}")
print(f"目標值：{best_objectives}")
```
