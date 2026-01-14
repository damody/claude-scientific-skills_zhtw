# SymPy 進階主題

本文件涵蓋 SymPy 的進階數學能力，包括幾何、數論、組合學、邏輯和集合、統計、多項式和特殊函數。

## 幾何

### 2D 幾何

```python
from sympy.geometry import Point, Line, Circle, Triangle, Polygon

# 點
p1 = Point(0, 0)
p2 = Point(1, 1)
p3 = Point(1, 0)

# 點之間的距離
dist = p1.distance(p2)

# 線
line = Line(p1, p2)
line_from_eq = Line(Point(0, 0), slope=2)

# 線的性質
line.slope       # 斜率
line.equation()  # 線的方程式
line.length      # oo（線為無限）

# 線段
from sympy.geometry import Segment
seg = Segment(p1, p2)
seg.length       # 有限長度
seg.midpoint     # 中點

# 交點
line2 = Line(Point(0, 1), Point(1, 0))
intersection = line.intersection(line2)  # [Point(1/2, 1/2)]

# 圓
circle = Circle(Point(0, 0), 5)  # 圓心、半徑
circle.area           # 25*pi
circle.circumference  # 10*pi

# 三角形
tri = Triangle(p1, p2, p3)
tri.area       # 面積
tri.perimeter  # 周長
tri.angles     # 角度字典
tri.vertices   # 頂點元組

# 多邊形
poly = Polygon(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))
poly.area
poly.perimeter
poly.vertices
```

### 幾何查詢

```python
# 檢查點是否在線/曲線上
point = Point(0.5, 0.5)
line.contains(point)

# 檢查是否平行/垂直
line1 = Line(Point(0, 0), Point(1, 1))
line2 = Line(Point(0, 1), Point(1, 2))
line1.is_parallel(line2)  # True
line1.is_perpendicular(line2)  # False

# 切線
from sympy.geometry import Circle, Point
circle = Circle(Point(0, 0), 5)
point = Point(5, 0)
tangents = circle.tangent_lines(point)
```

### 3D 幾何

```python
from sympy.geometry import Point3D, Line3D, Plane

# 3D 點
p1 = Point3D(0, 0, 0)
p2 = Point3D(1, 1, 1)
p3 = Point3D(1, 0, 0)

# 3D 線
line = Line3D(p1, p2)

# 平面
plane = Plane(p1, p2, p3)  # 從 3 個點
plane = Plane(Point3D(0, 0, 0), normal_vector=(1, 0, 0))  # 從點和法向量

# 平面方程式
plane.equation()

# 點到平面的距離
point = Point3D(2, 3, 4)
dist = plane.distance(point)

# 平面和線的交點
intersection = plane.intersection(line)
```

### 曲線和橢圓

```python
from sympy.geometry import Ellipse, Curve
from sympy import sin, cos, pi

# 橢圓
ellipse = Ellipse(Point(0, 0), hradius=3, vradius=2)
ellipse.area          # 6*pi
ellipse.eccentricity  # 離心率

# 參數曲線
from sympy.abc import t
curve = Curve((cos(t), sin(t)), (t, 0, 2*pi))  # 圓
```

## 數論

### 質數

```python
from sympy.ntheory import isprime, primerange, prime, nextprime, prevprime

# 檢查是否為質數
isprime(7)    # True
isprime(10)   # False

# 在範圍內生成質數
list(primerange(10, 50))  # [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# 第 n 個質數
prime(10)     # 29（第 10 個質數）

# 下一個和上一個質數
nextprime(10)  # 11
prevprime(10)  # 7
```

### 質因數分解

```python
from sympy import factorint, primefactors, divisors

# 質因數分解
factorint(60)  # {2: 2, 3: 1, 5: 1} 表示 2^2 * 3^1 * 5^1

# 質因數列表
primefactors(60)  # [2, 3, 5]

# 所有因數
divisors(60)  # [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
```

### GCD 和 LCM

```python
from sympy import gcd, lcm, igcd, ilcm

# 最大公因數
gcd(60, 48)   # 12
igcd(60, 48)  # 12（整數版本）

# 最小公倍數
lcm(60, 48)   # 240
ilcm(60, 48)  # 240（整數版本）

# 多個參數
gcd(60, 48, 36)  # 12
```

### 模運算

```python
from sympy.ntheory import mod_inverse, totient, is_primitive_root

# 模逆元（找 x 使得 a*x ≡ 1 (mod m)）
mod_inverse(3, 7)  # 5（因為 3*5 = 15 ≡ 1 (mod 7)）

# 歐拉函數
totient(10)  # 4（小於 10 且與 10 互質的數：1,3,7,9）

# 原根
is_primitive_root(2, 5)  # True
```

### 丟番圖方程式

```python
from sympy.solvers.diophantine import diophantine
from sympy.abc import x, y, z

# 線性丟番圖方程式：ax + by = c
diophantine(3*x + 4*y - 5)  # {(4*t_0 - 5, -3*t_0 + 5)}

# 二次形式
diophantine(x**2 + y**2 - 25)  # 畢達哥拉斯型方程式

# 更複雜的方程式
diophantine(x**2 - 4*x*y + 8*y**2 - 3*x + 7*y - 5)
```

### 連分數

```python
from sympy import nsimplify, continued_fraction_iterator
from sympy import Rational, pi

# 轉換為連分數
cf = continued_fraction_iterator(Rational(415, 93))
list(cf)  # [4, 2, 6, 7]

# 近似無理數
cf_pi = continued_fraction_iterator(pi.evalf(20))
```

## 組合學

### 排列和組合

```python
from sympy import factorial, binomial, factorial2
from sympy.functions.combinatorial.numbers import nC, nP

# 階乘
factorial(5)  # 120

# 二項係數（n 選 k）
binomial(5, 2)  # 10

# 排列 nPk = n!/(n-k)!
nP(5, 2)  # 20

# 組合 nCk = n!/(k!(n-k)!)
nC(5, 2)  # 10

# 雙階乘 n!!
factorial2(5)  # 15 (5*3*1)
factorial2(6)  # 48 (6*4*2)
```

### 排列物件

```python
from sympy.combinatorics import Permutation

# 建立排列（循環表示法）
p = Permutation([1, 2, 0, 3])  # 送 0->1, 1->2, 2->0, 3->3
p = Permutation(0, 1, 2)(3)    # 循環表示法：(0 1 2)(3)

# 排列運算
p.order()       # 排列的階
p.is_even       # 如果是偶排列則為 True
p.inversions()  # 逆序數

# 組合排列
q = Permutation([2, 0, 1, 3])
r = p * q       # 組合
```

### 分割

```python
from sympy.utilities.iterables import partitions
from sympy.functions.combinatorial.numbers import partition

# 整數分割數
partition(5)  # 7 (5, 4+1, 3+2, 3+1+1, 2+2+1, 2+1+1+1, 1+1+1+1+1)

# 生成所有分割
list(partitions(4))
# {4: 1}, {3: 1, 1: 1}, {2: 2}, {2: 1, 1: 2}, {1: 4}
```

### 卡塔蘭數和費波那契數

```python
from sympy import catalan, fibonacci, lucas

# 卡塔蘭數
catalan(5)  # 42

# 費波那契數
fibonacci(10)  # 55
lucas(10)      # 123（盧卡斯數）
```

### 群論

```python
from sympy.combinatorics import PermutationGroup, Permutation

# 建立排列群
p1 = Permutation([1, 0, 2])
p2 = Permutation([0, 2, 1])
G = PermutationGroup(p1, p2)

# 群的性質
G.order()        # 群的階
G.is_abelian     # 檢查是否為交換群
G.is_cyclic()    # 檢查是否為循環群
G.elements       # 所有群元素
```

## 邏輯和集合

### 布林邏輯

```python
from sympy import symbols, And, Or, Not, Xor, Implies, Equivalent
from sympy.logic.boolalg import truth_table, simplify_logic

# 定義布林變數
x, y, z = symbols('x y z', bool=True)

# 邏輯運算
expr = And(x, Or(y, Not(z)))
expr = Implies(x, y)  # x -> y
expr = Equivalent(x, y)  # x <-> y
expr = Xor(x, y)  # 互斥或

# 簡化
expr = (x & y) | (x & ~y)
simplified = simplify_logic(expr)  # 返回 x

# 真值表
expr = Implies(x, y)
print(truth_table(expr, [x, y]))
```

### 集合

```python
from sympy import FiniteSet, Interval, Union, Intersection, Complement
from sympy import S  # 用於特殊集合

# 有限集合
A = FiniteSet(1, 2, 3, 4)
B = FiniteSet(3, 4, 5, 6)

# 集合運算
union = Union(A, B)              # {1, 2, 3, 4, 5, 6}
intersection = Intersection(A, B)  # {3, 4}
difference = Complement(A, B)     # {1, 2}

# 區間
I = Interval(0, 1)              # [0, 1]
I_open = Interval.open(0, 1)    # (0, 1)
I_lopen = Interval.Lopen(0, 1)  # (0, 1]
I_ropen = Interval.Ropen(0, 1)  # [0, 1)

# 特殊集合
S.Reals        # 所有實數
S.Integers     # 所有整數
S.Naturals     # 自然數
S.EmptySet     # 空集合
S.Complexes    # 複數

# 集合成員
3 in A  # True
7 in A  # False

# 子集和超集
A.is_subset(B)    # False
A.is_superset(B)  # False
```

### 集合論運算

```python
from sympy import ImageSet, Lambda
from sympy.abc import x

# 像集（函數值的集合）
squares = ImageSet(Lambda(x, x**2), S.Integers)
# {x^2 | x ∈ ℤ}

# 冪集
from sympy.sets import FiniteSet
A = FiniteSet(1, 2, 3)
# 注意：SymPy 沒有直接的冪集，但可以生成
```

## 多項式

### 多項式操作

```python
from sympy import Poly, symbols, factor, expand, roots
x, y = symbols('x y')

# 建立多項式
p = Poly(x**2 + 2*x + 1, x)

# 多項式性質
p.degree()       # 2
p.coeffs()       # [1, 2, 1]
p.as_expr()      # 轉換回運算式

# 算術
p1 = Poly(x**2 + 1, x)
p2 = Poly(x + 1, x)
p3 = p1 + p2
p4 = p1 * p2
q, r = div(p1, p2)  # 商和餘數
```

### 多項式根

```python
from sympy import roots, real_roots, count_roots

p = Poly(x**3 - 6*x**2 + 11*x - 6, x)

# 所有根
r = roots(p)  # {1: 1, 2: 1, 3: 1}

# 僅實根
r = real_roots(p)

# 計算區間內的根數
count_roots(p, a, b)  # [a, b] 內的根數
```

### 多項式 GCD 和因式分解

```python
from sympy import gcd, lcm, factor, factor_list

p1 = Poly(x**2 - 1, x)
p2 = Poly(x**2 - 2*x + 1, x)

# GCD 和 LCM
g = gcd(p1, p2)
l = lcm(p1, p2)

# 因式分解
f = factor(x**3 - x**2 + x - 1)  # (x - 1)*(x**2 + 1)
factors = factor_list(x**3 - x**2 + x - 1)  # 列表形式
```

### Groebner 基

```python
from sympy import groebner, symbols

x, y, z = symbols('x y z')
polynomials = [x**2 + y**2 + z**2 - 1, x*y - z]

# 計算 Groebner 基
gb = groebner(polynomials, x, y, z)
```

## 統計

### 隨機變數

```python
from sympy.stats import (
    Normal, Uniform, Exponential, Poisson, Binomial,
    P, E, variance, density, sample
)

# 定義隨機變數
X = Normal('X', 0, 1)  # Normal(平均值, 標準差)
Y = Uniform('Y', 0, 1)  # Uniform(a, b)
Z = Exponential('Z', 1)  # Exponential(率)

# 機率
P(X > 0)  # 1/2
P((X > 0) & (X < 1))

# 期望值
E(X)  # 0
E(X**2)  # 1

# 變異數
variance(X)  # 1

# 機率密度函數
density(X)(x)  # sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))
```

### 離散分布

```python
from sympy.stats import Die, Bernoulli, Binomial, Poisson

# 骰子
D = Die('D', 6)
P(D > 3)  # 1/2

# 伯努利分布
B = Bernoulli('B', 0.5)
P(B)  # 1/2

# 二項分布
X = Binomial('X', 10, 0.5)
P(X == 5)  # 10 次試驗中恰好 5 次成功的機率

# 泊松分布
Y = Poisson('Y', 3)
P(Y < 2)  # 少於 2 個事件的機率
```

### 聯合分布

```python
from sympy.stats import Normal, P, E
from sympy import symbols

# 獨立隨機變數
X = Normal('X', 0, 1)
Y = Normal('Y', 0, 1)

# 聯合機率
P((X > 0) & (Y > 0))  # 1/4

# 共變異數
from sympy.stats import covariance
covariance(X, Y)  # 0（獨立）
```

## 特殊函數

### 常見特殊函數

```python
from sympy import (
    gamma,      # Gamma 函數
    beta,       # Beta 函數
    erf,        # 誤差函數
    besselj,    # 第一類 Bessel 函數
    bessely,    # 第二類 Bessel 函數
    hermite,    # Hermite 多項式
    legendre,   # Legendre 多項式
    laguerre,   # Laguerre 多項式
    chebyshevt, # Chebyshev 多項式（第一類）
    zeta        # Riemann zeta 函數
)

# Gamma 函數
gamma(5)  # 24（等於 4!）
gamma(1/2)  # sqrt(pi)

# Bessel 函數
besselj(0, x)  # J_0(x)
bessely(1, x)  # Y_1(x)

# 正交多項式
hermite(3, x)    # 8*x**3 - 12*x
legendre(2, x)   # (3*x**2 - 1)/2
laguerre(2, x)   # x**2/2 - 2*x + 1
chebyshevt(3, x) # 4*x**3 - 3*x
```

### 超幾何函數

```python
from sympy import hyper, meijerg

# 超幾何函數
hyper([1, 2], [3], x)

# Meijer G 函數
meijerg([[1, 1], []], [[1], [0]], x)
```

## 常見模式

### 模式 1：符號幾何問題

```python
from sympy.geometry import Point, Triangle
from sympy import symbols

# 定義符號三角形
a, b = symbols('a b', positive=True)
tri = Triangle(Point(0, 0), Point(a, 0), Point(0, b))

# 符號計算性質
area = tri.area  # a*b/2
perimeter = tri.perimeter  # a + b + sqrt(a**2 + b**2)
```

### 模式 2：數論計算

```python
from sympy.ntheory import factorint, totient, isprime

# 因式分解和分析
n = 12345
factors = factorint(n)
phi = totient(n)
is_prime = isprime(n)
```

### 模式 3：組合生成

```python
from sympy.utilities.iterables import multiset_permutations, combinations

# 生成所有排列
perms = list(multiset_permutations([1, 2, 3]))

# 生成組合
combs = list(combinations([1, 2, 3, 4], 2))
```

### 模式 4：機率計算

```python
from sympy.stats import Normal, P, E, variance

X = Normal('X', mu, sigma)

# 計算統計量
mean = E(X)
var = variance(X)
prob = P(X > a)
```

## 重要注意事項

1. **假設：**許多運算受益於符號假設（例如 `positive=True`、`integer=True`）。

2. **符號 vs 數值：**這些運算是符號的。使用 `evalf()` 獲得數值結果。

3. **效能：**複雜的符號運算可能很慢。考慮對大規模計算使用數值方法。

4. **精確算術：**SymPy 維持精確表示（例如 `sqrt(2)` 而非 `1.414...`）。
