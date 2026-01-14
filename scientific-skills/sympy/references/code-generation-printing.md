# SymPy 程式碼生成和列印

本文件涵蓋 SymPy 在各種語言中生成可執行程式碼、將運算式轉換為不同輸出格式，以及自訂列印行為的能力。

## 程式碼生成

### 轉換為 NumPy 函式

```python
from sympy import symbols, sin, cos, lambdify
import numpy as np

x, y = symbols('x y')
expr = sin(x) + cos(y)

# 建立 NumPy 函式
f = lambdify((x, y), expr, 'numpy')

# 與 NumPy 陣列一起使用
x_vals = np.linspace(0, 2*np.pi, 100)
y_vals = np.linspace(0, 2*np.pi, 100)
result = f(x_vals, y_vals)
```

### Lambdify 選項

```python
from sympy import lambdify, exp, sqrt

# 不同的後端
f_numpy = lambdify(x, expr, 'numpy')      # NumPy
f_scipy = lambdify(x, expr, 'scipy')      # SciPy
f_mpmath = lambdify(x, expr, 'mpmath')    # mpmath（任意精度）
f_math = lambdify(x, expr, 'math')        # Python math 模組

# 自訂函式對應
custom_funcs = {'sin': lambda x: x}  # 將 sin 替換為恆等函式
f = lambdify(x, sin(x), modules=[custom_funcs, 'numpy'])

# 多個運算式
exprs = [x**2, x**3, x**4]
f = lambdify(x, exprs, 'numpy')
# 返回結果的元組
```

### 生成 C/C++ 程式碼

```python
from sympy.utilities.codegen import codegen
from sympy import symbols

x, y = symbols('x y')
expr = x**2 + y**2

# 生成 C 程式碼
[(c_name, c_code), (h_name, h_header)] = codegen(
    ('distance_squared', expr),
    'C',
    header=False,
    empty=False
)

print(c_code)
# 輸出有效的 C 函式
```

### 生成 Fortran 程式碼

```python
from sympy.utilities.codegen import codegen

[(f_name, f_code), (h_name, h_interface)] = codegen(
    ('my_function', expr),
    'F95',  # Fortran 95
    header=False
)

print(f_code)
```

### 進階程式碼生成

```python
from sympy.utilities.codegen import CCodeGen, make_routine
from sympy import MatrixSymbol, Matrix

# 矩陣運算
A = MatrixSymbol('A', 3, 3)
expr = A + A.T

# 建立程序
routine = make_routine('matrix_sum', expr)

# 生成程式碼
gen = CCodeGen()
code = gen.write([routine], prefix='my_module')
```

### 程式碼列印器

```python
from sympy.printing.c import C99CodePrinter, C89CodePrinter
from sympy.printing.fortran import FCodePrinter
from sympy.printing.cxx import CXX11CodePrinter

# C 程式碼
c_printer = C99CodePrinter()
c_code = c_printer.doprint(expr)

# Fortran 程式碼
f_printer = FCodePrinter()
f_code = f_printer.doprint(expr)

# C++ 程式碼
cxx_printer = CXX11CodePrinter()
cxx_code = cxx_printer.doprint(expr)
```

## 列印和輸出格式

### 美觀列印

```python
from sympy import init_printing, pprint, pretty, symbols
from sympy import Integral, sqrt, pi

# 初始化美觀列印（用於 Jupyter notebooks 和終端機）
init_printing()

x = symbols('x')
expr = Integral(sqrt(1/x), (x, 0, pi))

# 在終端機美觀列印
pprint(expr)
#   π
#   ⌠
#   ⎮   1
#   ⎮  ───  dx
#   ⎮  √x
#   ⌡
#   0

# 取得美觀字串
s = pretty(expr)
print(s)
```

### LaTeX 輸出

```python
from sympy import latex, symbols, Integral, sin, sqrt

x, y = symbols('x y')
expr = Integral(sin(x)**2, (x, 0, pi))

# 轉換為 LaTeX
latex_str = latex(expr)
print(latex_str)
# \int\limits_{0}^{\pi} \sin^{2}{\left(x \right)}\, dx

# 自訂 LaTeX 格式
latex_str = latex(expr, mode='equation')  # 包裹在 equation 環境
latex_str = latex(expr, mode='inline')    # 行內數學

# 對於矩陣
from sympy import Matrix
M = Matrix([[1, 2], [3, 4]])
latex(M)  # \left[\begin{matrix}1 & 2\\3 & 4\end{matrix}\right]
```

### MathML 輸出

```python
from sympy.printing.mathml import mathml, print_mathml
from sympy import sin, pi

expr = sin(pi/4)

# Content MathML
mathml_str = mathml(expr)

# Presentation MathML
mathml_str = mathml(expr, printer='presentation')

# 列印到控制台
print_mathml(expr)
```

### 字串表示

```python
from sympy import symbols, sin, pi, srepr, sstr

x = symbols('x')
expr = sin(x)**2

# 標準字串（在 Python 中看到的）
str(expr)  # 'sin(x)**2'

# 字串表示（更美觀）
sstr(expr)  # 'sin(x)**2'

# 可重現表示
srepr(expr)  # "Pow(sin(Symbol('x')), Integer(2))"
# 這可以用 eval() 重建運算式
```

### 自訂列印

```python
from sympy.printing.str import StrPrinter

class MyPrinter(StrPrinter):
    def _print_Symbol(self, expr):
        return f"<{expr.name}>"

    def _print_Add(self, expr):
        return " PLUS ".join(self._print(arg) for arg in expr.args)

printer = MyPrinter()
x, y = symbols('x y')
print(printer.doprint(x + y))  # "<x> PLUS <y>"
```

## Python 程式碼生成

### autowrap - 編譯和匯入

```python
from sympy.utilities.autowrap import autowrap
from sympy import symbols

x, y = symbols('x y')
expr = x**2 + y**2

# 自動編譯 C 程式碼並建立 Python 包裝器
f = autowrap(expr, backend='cython')
# 或 backend='f2py' 用於 Fortran

# 像普通函式一樣使用
result = f(3, 4)  # 25
```

### ufuncify - 建立 NumPy ufuncs

```python
from sympy.utilities.autowrap import ufuncify
import numpy as np

x, y = symbols('x y')
expr = x**2 + y**2

# 建立通用函式
f = ufuncify((x, y), expr)

# 與 NumPy 廣播一起工作
x_arr = np.array([1, 2, 3])
y_arr = np.array([4, 5, 6])
result = f(x_arr, y_arr)  # [17, 29, 45]
```

## 運算式樹操作

### 遍歷運算式樹

```python
from sympy import symbols, sin, cos, preorder_traversal, postorder_traversal

x, y = symbols('x y')
expr = sin(x) + cos(y)

# 前序遍歷（父節點先於子節點）
for arg in preorder_traversal(expr):
    print(arg)

# 後序遍歷（子節點先於父節點）
for arg in postorder_traversal(expr):
    print(arg)

# 取得所有子運算式
subexprs = list(preorder_traversal(expr))
```

### 樹中的運算式替換

```python
from sympy import Wild, symbols, sin, cos

x, y = symbols('x y')
a = Wild('a')

expr = sin(x) + cos(y)

# 模式匹配和替換
new_expr = expr.replace(sin(a), a**2)  # sin(x) -> x**2
```

## Jupyter Notebook 整合

### 顯示數學

```python
from sympy import init_printing, display
from IPython.display import display as ipy_display

# 為 Jupyter 初始化列印
init_printing(use_latex='mathjax')  # 或 'png'、'svg'

# 美觀顯示運算式
expr = Integral(sin(x)**2, x)
display(expr)  # 在 notebook 中渲染為 LaTeX

# 多個輸出
ipy_display(expr1, expr2, expr3)
```

### 互動式小工具

```python
from sympy import symbols, sin
from IPython.display import display
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt
import numpy as np

x = symbols('x')
expr = sin(x)

@interact(a=FloatSlider(min=0, max=10, step=0.1, value=1))
def plot_expr(a):
    f = lambdify(x, a * expr, 'numpy')
    x_vals = np.linspace(-np.pi, np.pi, 100)
    plt.plot(x_vals, f(x_vals))
    plt.show()
```

## 表示間的轉換

### 字串到 SymPy

```python
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols

x, y = symbols('x y')

# 解析字串為運算式
expr = parse_expr('x**2 + 2*x + 1')
expr = parse_expr('sin(x) + cos(y)')

# 含變換
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application
)

transformations = standard_transformations + (implicit_multiplication_application,)
expr = parse_expr('2x', transformations=transformations)  # 將 '2x' 視為 2*x
```

### LaTeX 到 SymPy

```python
from sympy.parsing.latex import parse_latex

# 解析 LaTeX
expr = parse_latex(r'\frac{x^2}{y}')
# 返回：x**2/y

expr = parse_latex(r'\int_0^\pi \sin(x) dx')
```

### Mathematica 到 SymPy

```python
from sympy.parsing.mathematica import parse_mathematica

# 解析 Mathematica 程式碼
expr = parse_mathematica('Sin[x]^2 + Cos[y]^2')
# 返回 SymPy 運算式
```

## 匯出結果

### 匯出到檔案

```python
from sympy import symbols, sin
import json

x = symbols('x')
expr = sin(x)**2

# 匯出為 LaTeX 到檔案
with open('output.tex', 'w') as f:
    f.write(latex(expr))

# 匯出為字串
with open('output.txt', 'w') as f:
    f.write(str(expr))

# 匯出為 Python 程式碼
with open('output.py', 'w') as f:
    f.write(f"from numpy import sin\n")
    f.write(f"def f(x):\n")
    f.write(f"    return {lambdify(x, expr, 'numpy')}\n")
```

### Pickle SymPy 物件

```python
import pickle
from sympy import symbols, sin

x = symbols('x')
expr = sin(x)**2 + x

# 儲存
with open('expr.pkl', 'wb') as f:
    pickle.dump(expr, f)

# 載入
with open('expr.pkl', 'rb') as f:
    loaded_expr = pickle.load(f)
```

## 數值評估和精度

### 高精度評估

```python
from sympy import symbols, pi, sqrt, E, exp, sin
from mpmath import mp

x = symbols('x')

# 標準精度
pi.evalf()  # 3.14159265358979

# 高精度（1000 位數）
pi.evalf(1000)

# 使用 mpmath 設定全域精度
mp.dps = 50  # 50 位小數
expr = exp(pi * sqrt(163))
float(expr.evalf())

# 對於運算式
result = (sqrt(2) + sqrt(3)).evalf(100)
```

### 數值替換

```python
from sympy import symbols, sin, cos

x, y = symbols('x y')
expr = sin(x) + cos(y)

# 數值評估
result = expr.evalf(subs={x: 1.5, y: 2.3})

# 含單位
from sympy.physics.units import meter, second
distance = 100 * meter
time = 10 * second
speed = distance / time
speed.evalf()
```

## 常見模式

### 模式 1：生成和執行程式碼

```python
from sympy import symbols, lambdify
import numpy as np

# 1. 定義符號運算式
x, y = symbols('x y')
expr = x**2 + y**2

# 2. 生成函式
f = lambdify((x, y), expr, 'numpy')

# 3. 使用數值資料執行
data_x = np.random.rand(1000)
data_y = np.random.rand(1000)
results = f(data_x, data_y)
```

### 模式 2：建立 LaTeX 文件

```python
from sympy import symbols, Integral, latex
from sympy.abc import x

# 定義數學內容
expr = Integral(x**2, (x, 0, 1))
result = expr.doit()

# 生成 LaTeX 文件
latex_doc = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\begin{{document}}

我們計算積分：
\\begin{{equation}}
{latex(expr)} = {latex(result)}
\\end{{equation}}

\\end{{document}}
"""

with open('document.tex', 'w') as f:
    f.write(latex_doc)
```

### 模式 3：互動式計算

```python
from sympy import symbols, simplify, expand
from sympy.parsing.sympy_parser import parse_expr

x, y = symbols('x y')

# 互動式輸入
user_input = input("輸入運算式：")
expr = parse_expr(user_input)

# 處理
simplified = simplify(expr)
expanded = expand(expr)

# 顯示
print(f"簡化：{simplified}")
print(f"展開：{expanded}")
print(f"LaTeX：{latex(expr)}")
```

### 模式 4：批量程式碼生成

```python
from sympy import symbols, lambdify
from sympy.utilities.codegen import codegen

# 多個函式
x = symbols('x')
functions = {
    'f1': x**2,
    'f2': x**3,
    'f3': x**4
}

# 為所有函式生成 C 程式碼
for name, expr in functions.items():
    [(c_name, c_code), _] = codegen((name, expr), 'C')
    with open(f'{name}.c', 'w') as f:
        f.write(c_code)
```

### 模式 5：效能最佳化

```python
from sympy import symbols, sin, cos, cse
import numpy as np

x, y = symbols('x y')

# 含重複子運算式的複雜運算式
expr = sin(x + y)**2 + cos(x + y)**2 + sin(x + y)

# 公共子運算式消除
replacements, reduced = cse(expr)
# replacements: [(x0, sin(x + y)), (x1, cos(x + y))]
# reduced: [x0**2 + x1**2 + x0]

# 生成最佳化程式碼
for var, subexpr in replacements:
    print(f"{var} = {subexpr}")
print(f"result = {reduced[0]}")
```

## 重要注意事項

1. **NumPy 相容性：**使用 `lambdify` 與 NumPy 時，確保您的運算式使用 NumPy 中可用的函式。

2. **效能：**對於數值工作，始終使用 `lambdify` 或程式碼生成而非在迴圈中使用 `subs()` + `evalf()`。

3. **精度：**需要時使用 `mpmath` 進行任意精度算術。

4. **程式碼生成注意事項：**生成的程式碼可能無法處理所有邊界情況。徹底測試。

5. **編譯：**`autowrap` 和 `ufuncify` 需要 C/Fortran 編譯器，可能需要在您的系統上進行配置。

6. **解析：**解析使用者輸入時，驗證和清理以避免程式碼注入漏洞。

7. **Jupyter：**為獲得 Jupyter notebooks 中的最佳結果，在會話開始時呼叫 `init_printing()`。
