# Pymoo 演算法參考

pymoo 中可用最佳化演算法的完整參考。

## 單目標最佳化演算法

### 遺傳演算法（GA）
**用途：** 通用單目標演化最佳化
**最適合：** 連續、離散或混合變數問題
**演算法類型：** (μ+λ) 遺傳演算法

**關鍵參數：**
- `pop_size`：族群大小（預設：100）
- `sampling`：初始族群生成策略
- `selection`：親代選擇機制（預設：Tournament）
- `crossover`：重組運算子（預設：SBX）
- `mutation`：變異運算子（預設：Polynomial）
- `eliminate_duplicates`：移除重複解（預設：True）
- `n_offsprings`：每代的子代數量

**用法：**
```python
from pymoo.algorithms.soo.nonconvex.ga import GA
algorithm = GA(pop_size=100, eliminate_duplicates=True)
```

### 差分演化（DE）
**用途：** 單目標連續最佳化
**最適合：** 具有良好全域搜尋能力的連續參數最佳化
**演算法類型：** 基於族群的差分演化

**變體：** 多種 DE 策略可用（rand/1/bin、best/1/bin 等）

### 粒子群最佳化（PSO）
**用途：** 透過群體智慧的單目標最佳化
**最適合：** 連續問題，在平滑地形上快速收斂

### CMA-ES
**用途：** 共變異數矩陣適應演化策略
**最適合：** 連續最佳化，特別適合雜訊或病態問題

### Pattern Search
**用途：** 直接搜尋方法
**最適合：** 梯度資訊不可用的問題

### Nelder-Mead
**用途：** 基於單純形的最佳化
**最適合：** 連續函數的局部最佳化

## 多目標最佳化演算法

### NSGA-II（非支配排序遺傳演算法 II）
**用途：** 具有 2-3 個目標的多目標最佳化
**最適合：** 需要良好分布 Pareto 前沿的雙目標和三目標問題
**選擇策略：** 非支配排序 + 擁擠度距離

**關鍵特徵：**
- 快速非支配排序
- 使用擁擠度距離維持多樣性
- 菁英主義方法
- 二元競賽配對選擇

**關鍵參數：**
- `pop_size`：族群大小（預設：100）
- `sampling`：初始族群策略
- `crossover`：連續變數預設 SBX
- `mutation`：預設多項式突變
- `survival`：RankAndCrowding

**用法：**
```python
from pymoo.algorithms.moo.nsga2 import NSGA2
algorithm = NSGA2(pop_size=100)
```

**何時使用：**
- 2-3 個目標
- 需要跨 Pareto 前沿的分布解
- 標準多目標基準

### NSGA-III
**用途：** 高維目標最佳化（4+ 個目標）
**最適合：** 需要均勻 Pareto 前沿覆蓋的 4 個或更多目標問題
**選擇策略：** 基於參考方向的多樣性維持

**關鍵特徵：**
- 參考方向引導族群
- 在高維目標空間維持多樣性
- 透過參考點進行生態位保留
- 低代表參考方向選擇

**關鍵參數：**
- `ref_dirs`：參考方向（必需）
- `pop_size`：預設為參考方向數量
- `crossover`：預設 SBX
- `mutation`：預設多項式突變

**用法：**
```python
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

ref_dirs = get_reference_directions("das-dennis", n_dim=4, n_partitions=12)
algorithm = NSGA3(ref_dirs=ref_dirs)
```

**NSGA-II vs NSGA-III：**
- 2-3 個目標使用 NSGA-II
- 4+ 個目標使用 NSGA-III
- NSGA-III 提供更均勻的分布
- NSGA-II 計算開銷較低

### R-NSGA-II（基於參考點的 NSGA-II）
**用途：** 具有偏好表達的多目標最佳化
**最適合：** 當決策者有偏好的 Pareto 前沿區域時

### U-NSGA-III（統一 NSGA-III）
**用途：** 處理各種情況的改進版本
**最適合：** 具有額外穩健性的高維目標問題

### MOEA/D（基於分解的多目標演化演算法）
**用途：** 基於分解的多目標最佳化
**最適合：** 分解為標量子問題有效的問題

### AGE-MOEA
**用途：** 自適應幾何估計
**最適合：** 具有自適應機制的多目標和高維目標問題

### RVEA（參考向量引導演化演算法）
**用途：** 基於參考向量的高維目標最佳化
**最適合：** 具有自適應參考向量的高維目標問題

### SMS-EMOA
**用途：** S-度量選擇演化多目標演算法
**最適合：** 超體積指標關鍵的問題
**選擇：** 使用支配超體積貢獻

## 動態多目標演算法

### D-NSGA-II
**用途：** 動態多目標問題
**最適合：** 隨時間變化的目標函數或約束

### KGB-DMOEA
**用途：** 知識引導的動態多目標最佳化
**最適合：** 利用歷史資訊的動態問題

## 約束最佳化

### SRES（隨機排序演化策略）
**用途：** 單目標約束最佳化
**最適合：** 重度約束問題

### ISRES（改進的 SRES）
**用途：** 增強的約束最佳化
**最適合：** 複雜的約束地形

## 演算法選擇指南

**單目標問題：**
- 通用問題從 GA 開始
- 連續最佳化使用 DE
- 平滑問題上更快收斂嘗試 PSO
- 困難/雜訊地形使用 CMA-ES

**多目標問題：**
- 2-3 個目標：NSGA-II
- 4+ 個目標：NSGA-III
- 偏好表達：R-NSGA-II
- 適合分解：MOEA/D
- 超體積重點：SMS-EMOA

**約束問題：**
- 基於可行性的存活選擇（適用於大多數演算法）
- 重度約束：SRES/ISRES
- 懲罰方法以確保演算法相容性

**動態問題：**
- 隨時間變化：D-NSGA-II
- 歷史知識有用：KGB-DMOEA
