# Pymoo 遺傳運算子參考

pymoo 中遺傳運算子的完整參考。

## 抽樣運算子

抽樣運算子在最佳化開始時初始化族群。

### 隨機抽樣
**用途：** 生成隨機初始解
**類型：**
- `FloatRandomSampling`：連續變數
- `BinaryRandomSampling`：二進位變數
- `IntegerRandomSampling`：整數變數
- `PermutationRandomSampling`：基於排列的問題

**用法：**
```python
from pymoo.operators.sampling.rnd import FloatRandomSampling
sampling = FloatRandomSampling()
```

### 拉丁超立方抽樣（LHS）
**用途：** 空間填充的初始族群
**優點：** 比隨機抽樣更好地覆蓋搜尋空間
**類型：**
- `LHS`：標準拉丁超立方

**用法：**
```python
from pymoo.operators.sampling.lhs import LHS
sampling = LHS()
```

### 自訂抽樣
透過 Population 物件或 NumPy 陣列提供初始族群

## 選擇運算子

選擇運算子選擇用於繁殖的親代。

### 競賽選擇
**用途：** 透過競賽比賽選擇親代
**機制：** 隨機選擇 k 個個體，選擇最佳者
**參數：**
- `pressure`：競賽大小（預設：2）
- `func_comp`：比較函數

**用法：**
```python
from pymoo.operators.selection.tournament import TournamentSelection
selection = TournamentSelection(pressure=2)
```

### 隨機選擇
**用途：** 均勻隨機親代選擇
**用例：** 基準或探索導向的演算法

**用法：**
```python
from pymoo.operators.selection.rnd import RandomSelection
selection = RandomSelection()
```

## 交叉運算子

交叉運算子重組親代解以建立子代。

### 連續變數

#### 模擬二進位交叉（SBX）
**用途：** 連續最佳化的主要交叉
**機制：** 模擬二進位編碼變數的單點交叉
**參數：**
- `prob`：交叉機率（預設：0.9）
- `eta`：分布指數（預設：15）
  - 較高的 eta → 子代更接近親代
  - 較低的 eta → 更多探索

**用法：**
```python
from pymoo.operators.crossover.sbx import SBX
crossover = SBX(prob=0.9, eta=15)
```

**字串簡寫：** `"real_sbx"`

#### 差分演化交叉
**用途：** DE 特定的重組
**變體：**
- `DE/rand/1/bin`
- `DE/best/1/bin`
- `DE/current-to-best/1/bin`

**參數：**
- `CR`：交叉率
- `F`：縮放因子

### 二進位變數

#### 單點交叉
**用途：** 在一個點切割並交換
**用法：**
```python
from pymoo.operators.crossover.pntx import SinglePointCrossover
crossover = SinglePointCrossover()
```

#### 雙點交叉
**用途：** 在兩個點之間切割並交換
**用法：**
```python
from pymoo.operators.crossover.pntx import TwoPointCrossover
crossover = TwoPointCrossover()
```

#### K 點交叉
**用途：** 多個切割點
**參數：**
- `n_points`：交叉點數量

#### 均勻交叉
**用途：** 每個基因獨立來自任一親代
**參數：**
- `prob`：每個基因的交換機率（預設：0.5）

**用法：**
```python
from pymoo.operators.crossover.ux import UniformCrossover
crossover = UniformCrossover(prob=0.5)
```

#### 半均勻交叉（HUX）
**用途：** 精確交換一半不同的基因
**優點：** 維持基因多樣性

### 排列

#### 順序交叉（OX）
**用途：** 保留親代的相對順序
**用例：** 旅行推銷員、排程問題

**用法：**
```python
from pymoo.operators.crossover.ox import OrderCrossover
crossover = OrderCrossover()
```

#### 邊重組交叉（ERX）
**用途：** 保留親代的邊資訊
**用例：** 邊連接重要的路由問題

#### 部分映射交叉（PMX）
**用途：** 交換片段同時維持排列有效性

## 突變運算子

突變運算子引入變異以維持多樣性。

### 連續變數

#### 多項式突變（PM）
**用途：** 連續最佳化的主要突變
**機制：** 多項式機率分布
**參數：**
- `prob`：每個變數的突變機率
- `eta`：分布指數（預設：20）
  - 較高的 eta → 較小的擾動
  - 較低的 eta → 較大的擾動

**用法：**
```python
from pymoo.operators.mutation.pm import PM
mutation = PM(prob=None, eta=20)  # prob=None 表示 1/n_var
```

**字串簡寫：** `"real_pm"`

**機率指南：**
- `None` 或 `1/n_var`：標準建議
- 較高以獲得更多探索
- 較低以獲得更多利用

### 二進位變數

#### 位元翻轉突變
**用途：** 以指定機率翻轉位元
**參數：**
- `prob`：每個位元的翻轉機率

**用法：**
```python
from pymoo.operators.mutation.bitflip import BitflipMutation
mutation = BitflipMutation(prob=0.05)
```

### 整數變數

#### 整數多項式突變
**用途：** 適應整數的 PM
**確保：** 突變後為有效整數值

### 排列

#### 反轉突變
**用途：** 反轉排列的一個片段
**用例：** 維持一些順序結構

**用法：**
```python
from pymoo.operators.mutation.inversion import InversionMutation
mutation = InversionMutation()
```

#### 打亂突變
**用途：** 隨機打亂一個片段

### 自訂突變
透過繼承 `Mutation` 類別定義自訂突變

## 修復運算子

修復運算子修正約束違反或確保解的可行性。

### 四捨五入修復
**用途：** 四捨五入到最近的有效值
**用例：** 具有邊界約束的整數/離散變數

### 反彈修復
**用途：** 將超出邊界的值反射回可行區域
**用例：** 箱約束的連續問題

### 投影修復
**用途：** 將不可行解投影到可行區域
**用例：** 線性約束

### 自訂修復
**用途：** 特定領域的約束處理
**實作：** 繼承 `Repair` 類別

**範例：**
```python
from pymoo.core.repair import Repair

class MyRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # 修改 X 以滿足約束
        # 回傳修復後的 X
        return X
```

## 運算子設定指南

### 參數調整

**交叉機率：**
- 高（0.8-0.95）：大多數問題的標準
- 較低：更強調突變

**突變機率：**
- `1/n_var`：標準建議
- 較高：更多探索，較慢收斂
- 較低：更快收斂，過早收斂的風險

**分布指數（eta）：**
- 交叉 eta（15-30）：較高用於局部搜尋
- 突變 eta（20-50）：較高用於利用

### 問題特定選擇

**連續問題：**
- 交叉：SBX
- 突變：多項式突變
- 選擇：競賽

**二進位問題：**
- 交叉：雙點或均勻
- 突變：位元翻轉
- 選擇：競賽

**排列問題：**
- 交叉：順序交叉（OX）
- 突變：反轉或打亂
- 選擇：競賽

**混合變數問題：**
- 對每種變數類型使用適當的運算子
- 確保運算子相容性

### 基於字串的設定

Pymoo 支援方便的基於字串的運算子指定：

```python
from pymoo.algorithms.soo.nonconvex.ga import GA

algorithm = GA(
    pop_size=100,
    sampling="real_random",
    crossover="real_sbx",
    mutation="real_pm"
)
```

**可用字串：**
- 抽樣：`"real_random"`、`"real_lhs"`、`"bin_random"`、`"perm_random"`
- 交叉：`"real_sbx"`、`"real_de"`、`"int_sbx"`、`"bin_ux"`、`"bin_hux"`
- 突變：`"real_pm"`、`"int_pm"`、`"bin_bitflip"`、`"perm_inv"`

## 運算子組合範例

### 標準連續 GA：
```python
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection

sampling = FloatRandomSampling()
crossover = SBX(prob=0.9, eta=15)
mutation = PM(eta=20)
selection = TournamentSelection()
```

### 二進位 GA：
```python
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

sampling = BinaryRandomSampling()
crossover = TwoPointCrossover()
mutation = BitflipMutation(prob=0.05)
```

### 排列 GA（TSP）：
```python
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation

sampling = PermutationRandomSampling()
crossover = OrderCrossover()
mutation = InversionMutation()
```
