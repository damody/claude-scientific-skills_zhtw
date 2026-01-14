# SymPy 矩陣和線性代數

本文件涵蓋 SymPy 的矩陣運算、線性代數能力，以及線性方程組求解。

## 矩陣建立

### 基本矩陣構建

```python
from sympy import Matrix, eye, zeros, ones, diag

# 從列的列表
M = Matrix([[1, 2], [3, 4]])
M = Matrix([
    [1, 2, 3],
    [4, 5, 6]
])

# 行向量
v = Matrix([1, 2, 3])

# 列向量
v = Matrix([[1, 2, 3]])
```

### 特殊矩陣

```python
# 單位矩陣
I = eye(3)  # 3x3 單位矩陣
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# 零矩陣
Z = zeros(2, 3)  # 2 列、3 行的零

# 全一矩陣
O = ones(3, 2)   # 3 列、2 行的一

# 對角矩陣
D = diag(1, 2, 3)
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]

# 區塊對角矩陣
from sympy import BlockDiagMatrix
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])
BD = BlockDiagMatrix(A, B)
```

## 矩陣性質和存取

### 形狀和維度

```python
M = Matrix([[1, 2, 3], [4, 5, 6]])

M.shape  # (2, 3) - 返回元組 (列, 行)
M.rows   # 2
M.cols   # 3
```

### 存取元素

```python
M = Matrix([[1, 2, 3], [4, 5, 6]])

# 單一元素
M[0, 0]  # 1（從零開始索引）
M[1, 2]  # 6

# 列存取
M[0, :]   # Matrix([[1, 2, 3]])
M.row(0)  # 同上

# 行存取
M[:, 1]   # Matrix([[2], [5]])
M.col(1)  # 同上

# 切片
M[0:2, 0:2]  # 左上角 2x2 子矩陣
```

### 修改

```python
M = Matrix([[1, 2], [3, 4]])

# 插入列
M = M.row_insert(1, Matrix([[5, 6]]))
# [[1, 2],
#  [5, 6],
#  [3, 4]]

# 插入行
M = M.col_insert(1, Matrix([7, 8]))

# 刪除列
M = M.row_del(0)

# 刪除行
M = M.col_del(1)
```

## 基本矩陣運算

### 算術運算

```python
from sympy import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

# 加法
C = A + B

# 減法
C = A - B

# 純量乘法
C = 2 * A

# 矩陣乘法
C = A * B

# 元素對元素乘法（Hadamard 乘積）
C = A.multiply_elementwise(B)

# 冪次
C = A**2  # 同 A * A
C = A**3  # A * A * A
```

### 轉置和共軛

```python
M = Matrix([[1, 2], [3, 4]])

# 轉置
M.T
# [[1, 3],
#  [2, 4]]

# 共軛（用於複數矩陣）
M.conjugate()

# 共軛轉置（Hermitian 轉置）
M.H  # 同 M.conjugate().T
```

### 反矩陣

```python
M = Matrix([[1, 2], [3, 4]])

# 反矩陣
M_inv = M**-1
M_inv = M.inv()

# 驗證
M * M_inv  # 返回單位矩陣

# 檢查是否可逆
M.is_invertible()  # True 或 False
```

## 進階線性代數

### 行列式

```python
M = Matrix([[1, 2], [3, 4]])
M.det()  # -2

# 用於符號矩陣
from sympy import symbols
a, b, c, d = symbols('a b c d')
M = Matrix([[a, b], [c, d]])
M.det()  # a*d - b*c
```

### 跡

```python
M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
M.trace()  # 1 + 5 + 9 = 15
```

### 列梯形式

```python
M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 簡化列梯形式
rref_M, pivot_cols = M.rref()
# rref_M 是 RREF 矩陣
# pivot_cols 是樞紐行索引的元組
```

### 秩

```python
M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
M.rank()  # 2（此矩陣秩不足）
```

### 零空間和行空間

```python
M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 零空間（核）
null = M.nullspace()
# 返回零空間的基向量列表

# 行空間
col = M.columnspace()
# 返回行空間的基向量列表

# 列空間
row = M.rowspace()
# 返回列空間的基向量列表
```

### 正交化

```python
# Gram-Schmidt 正交化
vectors = [Matrix([1, 2, 3]), Matrix([4, 5, 6])]
ortho = Matrix.orthogonalize(*vectors)

# 含正規化
ortho_norm = Matrix.orthogonalize(*vectors, normalize=True)
```

## 特徵值和特徵向量

### 計算特徵值

```python
M = Matrix([[1, 2], [2, 1]])

# 含重數的特徵值
eigenvals = M.eigenvals()
# 返回字典：{特徵值: 重數}
# 範例：{3: 1, -1: 1}

# 僅特徵值作為列表
eigs = list(M.eigenvals().keys())
```

### 計算特徵向量

```python
M = Matrix([[1, 2], [2, 1]])

# 含特徵值的特徵向量
eigenvects = M.eigenvects()
# 返回元組列表：(特徵值, 重數, [特徵向量])
# 範例：[(3, 1, [Matrix([1, 1])]), (-1, 1, [Matrix([1, -1])])]

# 存取個別特徵向量
for eigenval, multiplicity, eigenvecs in M.eigenvects():
    print(f"特徵值：{eigenval}")
    print(f"特徵向量：{eigenvecs}")
```

### 對角化

```python
M = Matrix([[1, 2], [2, 1]])

# 檢查是否可對角化
M.is_diagonalizable()  # True 或 False

# 對角化（M = P*D*P^-1）
P, D = M.diagonalize()
# P：特徵向量矩陣
# D：特徵值的對角矩陣

# 驗證
P * D * P**-1 == M  # True
```

### 特徵多項式

```python
from sympy import symbols
lam = symbols('lambda')

M = Matrix([[1, 2], [2, 1]])
charpoly = M.charpoly(lam)
# 返回特徵多項式
```

### Jordan 標準形

```python
M = Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])

# Jordan 形式（用於不可對角化矩陣）
P, J = M.jordan_form()
# J 是 Jordan 標準形
# P 是變換矩陣
```

## 矩陣分解

### LU 分解

```python
M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])

# LU 分解
L, U, perm = M.LUdecomposition()
# L：下三角矩陣
# U：上三角矩陣
# perm：排列索引
```

### QR 分解

```python
M = Matrix([[1, 2], [3, 4], [5, 6]])

# QR 分解
Q, R = M.QRdecomposition()
# Q：正交矩陣
# R：上三角矩陣
```

### Cholesky 分解

```python
# 用於正定對稱矩陣
M = Matrix([[4, 2], [2, 3]])

L = M.cholesky()
# L 是下三角矩陣使得 M = L*L.T
```

### 奇異值分解（SVD）

```python
M = Matrix([[1, 2], [3, 4], [5, 6]])

# SVD（注意：可能需要數值評估）
U, S, V = M.singular_value_decomposition()
# M = U * S * V
```

## 求解線性方程組

### 使用矩陣方程式

```python
# 求解 Ax = b
A = Matrix([[1, 2], [3, 4]])
b = Matrix([5, 6])

# 解
x = A.solve(b)  # 或 A**-1 * b

# 最小平方（用於超定系統）
x = A.solve_least_squares(b)
```

### 使用 linsolve

```python
from sympy import linsolve, symbols

x, y = symbols('x y')

# 方法 1：方程式列表
eqs = [x + y - 5, 2*x - y - 1]
sol = linsolve(eqs, [x, y])
# {(2, 3)}

# 方法 2：增廣矩陣
M = Matrix([[1, 1, 5], [2, -1, 1]])
sol = linsolve(M, [x, y])

# 方法 3：A*x = b 形式
A = Matrix([[1, 1], [2, -1]])
b = Matrix([5, 1])
sol = linsolve((A, b), [x, y])
```

### 欠定和超定系統

```python
# 欠定（無窮多解）
A = Matrix([[1, 2, 3]])
b = Matrix([6])
sol = A.solve(b)  # 返回參數解

# 超定（最小平方）
A = Matrix([[1, 2], [3, 4], [5, 6]])
b = Matrix([1, 2, 3])
sol = A.solve_least_squares(b)
```

## 符號矩陣

### 使用符號元素

```python
from sympy import symbols, Matrix

a, b, c, d = symbols('a b c d')
M = Matrix([[a, b], [c, d]])

# 所有運算都符號運作
M.det()  # a*d - b*c
M.inv()  # Matrix([[d/(a*d - b*c), -b/(a*d - b*c)], ...])
M.eigenvals()  # 符號特徵值
```

### 矩陣函式

```python
from sympy import exp, sin, cos, Matrix

M = Matrix([[0, 1], [-1, 0]])

# 矩陣指數
exp(M)

# 三角函式
sin(M)
cos(M)
```

## 可變 vs 不可變矩陣

```python
from sympy import Matrix, ImmutableMatrix

# 可變（預設）
M = Matrix([[1, 2], [3, 4]])
M[0, 0] = 5  # 允許

# 不可變（用作字典鍵等）
I = ImmutableMatrix([[1, 2], [3, 4]])
# I[0, 0] = 5  # 錯誤：ImmutableMatrix 無法修改
```

## 稀疏矩陣

對於含有許多零元素的大型矩陣：

```python
from sympy import SparseMatrix

# 建立稀疏矩陣
S = SparseMatrix(1000, 1000, {(0, 0): 1, (100, 100): 2})
# 僅儲存非零元素

# 將密集矩陣轉換為稀疏矩陣
M = Matrix([[1, 0, 0], [0, 2, 0]])
S = SparseMatrix(M)
```

## 常見線性代數模式

### 模式 1：對多個 b 向量求解 Ax = b

```python
A = Matrix([[1, 2], [3, 4]])
A_inv = A.inv()

b1 = Matrix([5, 6])
b2 = Matrix([7, 8])

x1 = A_inv * b1
x2 = A_inv * b2
```

### 模式 2：基底變換

```python
# 給定舊基底中的向量，轉換到新基底
old_basis = [Matrix([1, 0]), Matrix([0, 1])]
new_basis = [Matrix([1, 1]), Matrix([1, -1])]

# 基底變換矩陣
P = Matrix.hstack(*new_basis)
P_inv = P.inv()

# 將向量 v 從舊基底轉換到新基底
v = Matrix([3, 4])
v_new = P_inv * v
```

### 模式 3：矩陣條件數

```python
# 估計條件數（最大與最小奇異值的比值）
M = Matrix([[1, 2], [3, 4]])
eigenvals = M.eigenvals()
cond = max(eigenvals.keys()) / min(eigenvals.keys())
```

### 模式 4：投影矩陣

```python
# 投影到 A 的行空間
A = Matrix([[1, 0], [0, 1], [1, 1]])
P = A * (A.T * A).inv() * A.T
# P 是投影到 A 行空間的投影矩陣
```

## 重要注意事項

1. **零檢定：**SymPy 的符號零檢定可能影響精確度。對於數值工作，考慮使用 `evalf()` 或數值函式庫。

2. **效能：**對於大型數值矩陣，考慮使用 `lambdify` 轉換為 NumPy 或直接使用數值線性代數函式庫。

3. **符號計算：**含符號元素的矩陣運算對於大型矩陣可能計算量很大。

4. **假設：**使用符號假設（例如 `real=True`、`positive=True`）幫助 SymPy 正確簡化矩陣運算式。
