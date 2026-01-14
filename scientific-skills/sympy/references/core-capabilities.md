# SymPy 核心能力

本文件涵蓋 SymPy 的基本操作：符號計算基礎、代數、微積分、簡化和方程式求解。

## 建立符號和基本操作

### 符號建立

**單一符號：**
```python
from sympy import symbols, Symbol
x = Symbol('x')
# 或更常見的：
x, y, z = symbols('x y z')
```

**含假設：**
```python
x = symbols('x', real=True, positive=True)
n = symbols('n', integer=True)
```

常見假設：`real`、`positive`、`negative`、`integer`、`rational`、`prime`、`even`、`odd`、`complex`

### 基本算術

SymPy 支援標準 Python 運算符用於符號運算式：
- 加法：`x + y`
- 減法：`x - y`
- 乘法：`x * y`
- 除法：`x / y`
- 指數：`x**y`

**重要注意事項：**使用 `sympy.Rational()` 或 `S()` 獲得精確有理數：
```python
from sympy import Rational, S
expr = Rational(1, 2) * x  # 正確：精確的 1/2
expr = S(1)/2 * x          # 正確：精確的 1/2
expr = 0.5 * x             # 建立浮點近似
```

### 替換和評估

**替換值：**
```python
expr = x**2 + 2*x + 1
expr.subs(x, 3)  # 返回 16
expr.subs({x: 2, y: 3})  # 多重替換
```

**數值評估：**
```python
from sympy import pi, sqrt
expr = sqrt(8)
expr.evalf()      # 2.82842712474619
expr.evalf(20)    # 2.8284271247461900976（20 位數）
pi.evalf(100)     # 100 位數的 pi
```

## 簡化

SymPy 提供多種簡化函式，每種有不同的策略：

### 一般簡化

```python
from sympy import simplify, expand, factor, collect, cancel, trigsimp

# 一般簡化（嘗試多種方法）
simplify(sin(x)**2 + cos(x)**2)  # 返回 1

# 展開乘積和冪次
expand((x + 1)**3)  # x**3 + 3*x**2 + 3*x + 1

# 多項式因式分解
factor(x**3 - x**2 + x - 1)  # (x - 1)*(x**2 + 1)

# 按變數收集項
collect(x*y + x - 3 + 2*x**2 - z*x**2 + x**3, x)

# 約分有理運算式中的公因子
cancel((x**2 + 2*x + 1)/(x**2 + x))  # (x + 1)/x
```

### 三角簡化

```python
from sympy import sin, cos, tan, trigsimp, expand_trig

# 簡化三角運算式
trigsimp(sin(x)**2 + cos(x)**2)  # 1
trigsimp(sin(x)/cos(x))          # tan(x)

# 展開三角函式
expand_trig(sin(x + y))  # sin(x)*cos(y) + sin(y)*cos(x)
```

### 冪次和對數簡化

```python
from sympy import powsimp, powdenest, log, expand_log, logcombine

# 簡化冪次
powsimp(x**a * x**b)  # x**(a + b)

# 展開對數
expand_log(log(x*y))  # log(x) + log(y)

# 合併對數
logcombine(log(x) + log(y))  # log(x*y)
```

## 微積分

### 微分

```python
from sympy import diff, Derivative

# 一階微分
diff(x**2, x)  # 2*x

# 高階微分
diff(x**4, x, x, x)  # 24*x（三階微分）
diff(x**4, x, 3)     # 24*x（同上）

# 偏微分
diff(x**2*y**3, x, y)  # 6*x*y**2

# 未評估的微分（用於顯示）
d = Derivative(x**2, x)
d.doit()  # 評估為 2*x
```

### 積分

**不定積分：**
```python
from sympy import integrate

integrate(x**2, x)           # x**3/3
integrate(exp(x)*sin(x), x)  # exp(x)*sin(x)/2 - exp(x)*cos(x)/2
integrate(1/x, x)            # log(x)
```

**注意：**SymPy 不包含積分常數。如需要請手動添加 `+ C`。

**定積分：**
```python
from sympy import oo, pi, exp, sin

integrate(x**2, (x, 0, 1))    # 1/3
integrate(exp(-x), (x, 0, oo)) # 1
integrate(sin(x), (x, 0, pi))  # 2
```

**多重積分：**
```python
integrate(x*y, (x, 0, 1), (y, 0, x))  # 1/12
```

**數值積分（當符號失敗時）：**
```python
integrate(x**x, (x, 0, 1)).evalf()  # 0.783430510712134
```

### 極限

```python
from sympy import limit, oo, sin

# 基本極限
limit(sin(x)/x, x, 0)  # 1
limit(1/x, x, oo)      # 0

# 單邊極限
limit(1/x, x, 0, '+')  # oo
limit(1/x, x, 0, '-')  # -oo

# 在奇異點使用 limit()（而非 subs()）
limit((x**2 - 1)/(x - 1), x, 1)  # 2
```

**重要：**在奇異點使用 `limit()` 而非 `subs()`，因為無窮物件不能可靠地追蹤增長率。

### 級數展開

```python
from sympy import series, sin, exp, cos

# 泰勒級數展開
expr = sin(x)
expr.series(x, 0, 6)  # x - x**3/6 + x**5/120 + O(x**6)

# 在某點展開
exp(x).series(x, 1, 4)  # 在 x=1 處展開

# 移除 O() 項
series(exp(x), x, 0, 4).removeO()  # 1 + x + x**2/2 + x**3/6
```

### 有限差分（數值微分）

```python
from sympy import Function, differentiate_finite
f = Function('f')

# 使用有限差分近似微分
differentiate_finite(f(x), x)
f(x).as_finite_difference()
```

## 方程式求解

### 代數方程式 - solveset

**主要函式：**`solveset(equation, variable, domain)`

```python
from sympy import solveset, Eq, S

# 基本求解（假設方程式 = 0）
solveset(x**2 - 1, x)  # {-1, 1}
solveset(x**2 + 1, x)  # {-I, I}（複數解）

# 使用明確方程式
solveset(Eq(x**2, 4), x)  # {-2, 2}

# 指定定義域
solveset(x**2 - 1, x, domain=S.Reals)  # {-1, 1}
solveset(x**2 + 1, x, domain=S.Reals)  # EmptySet（無實數解）
```

**返回類型：**有限集合、區間或像集

### 方程組

**線性方程組 - linsolve：**
```python
from sympy import linsolve, Matrix

# 從方程式
linsolve([x + y - 2, x - y], x, y)  # {(1, 1)}

# 從增廣矩陣
linsolve(Matrix([[1, 1, 2], [1, -1, 0]]), x, y)

# 從 A*x = b 形式
A = Matrix([[1, 1], [1, -1]])
b = Matrix([2, 0])
linsolve((A, b), x, y)
```

**非線性方程組 - nonlinsolve：**
```python
from sympy import nonlinsolve

nonlinsolve([x**2 + y - 2, x + y**2 - 3], x, y)
```

**注意：**目前 nonlinsolve 不返回 LambertW 形式的解。

### 多項式根

```python
from sympy import roots, solve

# 取得含重數的根
roots(x**3 - 6*x**2 + 9*x, x)  # {0: 1, 3: 2}
# 表示 x=0（重數 1）、x=3（重數 2）
```

### 通用求解器 - solve

對於超越方程式更靈活的替代方案：
```python
from sympy import solve, exp, log

solve(exp(x) - 3, x)     # [log(3)]
solve(x**2 - 4, x)       # [-2, 2]
solve([x + y - 1, x - y + 1], [x, y])  # {x: 0, y: 1}
```

### 微分方程式 - dsolve

```python
from sympy import Function, dsolve, Derivative, Eq

# 定義函式
f = symbols('f', cls=Function)

# 求解 ODE
dsolve(Derivative(f(x), x) - f(x), f(x))
# 返回：Eq(f(x), C1*exp(x))

# 含初始條件
dsolve(Derivative(f(x), x) - f(x), f(x), ics={f(0): 1})
# 返回：Eq(f(x), exp(x))

# 二階 ODE
dsolve(Derivative(f(x), x, 2) + f(x), f(x))
# 返回：Eq(f(x), C1*sin(x) + C2*cos(x))
```

## 常見模式和最佳實踐

### 模式 1：漸進式建立複雜運算式
```python
from sympy import *
x, y = symbols('x y')

# 逐步建立
expr = x**2
expr = expr + 2*x + 1
expr = simplify(expr)
```

### 模式 2：使用假設
```python
# 定義含物理約束的符號
x = symbols('x', positive=True, real=True)
y = symbols('y', real=True)

# SymPy 可以使用這些進行簡化
sqrt(x**2)  # 返回 x（而非 Abs(x)）因為有 positive 假設
```

### 模式 3：轉換為數值函式
```python
from sympy import lambdify
import numpy as np

expr = x**2 + 2*x + 1
f = lambdify(x, expr, 'numpy')

# 現在可以與 numpy 陣列一起使用
x_vals = np.linspace(0, 10, 100)
y_vals = f(x_vals)
```

### 模式 4：美觀列印
```python
from sympy import init_printing, pprint
init_printing()  # 在終端機/notebook 啟用美觀列印

expr = Integral(sqrt(1/x), x)
pprint(expr)  # 顯示格式化良好的輸出
```
