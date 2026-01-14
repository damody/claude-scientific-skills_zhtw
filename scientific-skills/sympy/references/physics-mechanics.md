# SymPy 物理和力學

本文件涵蓋 SymPy 的物理模組，包括古典力學、量子力學、向量分析、單位、光學、連續介質力學和控制系統。

## 向量分析

### 建立參考座標系和向量

```python
from sympy.physics.vector import ReferenceFrame, dynamicsymbols

# 建立參考座標系
N = ReferenceFrame('N')  # 慣性座標系
B = ReferenceFrame('B')  # 本體座標系

# 建立向量
v = 3*N.x + 4*N.y + 5*N.z

# 時變量
t = dynamicsymbols._t
x = dynamicsymbols('x')  # 時間的函數
v = x.diff(t) * N.x  # 速度向量
```

### 向量運算

```python
from sympy.physics.vector import dot, cross

v1 = 3*N.x + 4*N.y
v2 = 1*N.x + 2*N.y + 3*N.z

# 內積
d = dot(v1, v2)

# 外積
c = cross(v1, v2)

# 大小
mag = v1.magnitude()

# 正規化
v1_norm = v1.normalize()
```

### 座標系方位

```python
# 相對於 N 旋轉座標系 B
from sympy import symbols, cos, sin
theta = symbols('theta')

# 繞 z 軸簡單旋轉
B.orient(N, 'Axis', [theta, N.z])

# 方向餘弦矩陣（DCM）
dcm = N.dcm(B)

# B 在 N 中的角速度
omega = B.ang_vel_in(N)
```

### 點和運動學

```python
from sympy.physics.vector import Point

# 建立點
O = Point('O')  # 原點
P = Point('P')

# 設定位置
P.set_pos(O, 3*N.x + 4*N.y)

# 設定速度
P.set_vel(N, 5*N.x + 2*N.y)

# 取得 P 在座標系 N 中的速度
v = P.vel(N)

# 取得加速度
a = P.acc(N)
```

## 古典力學

### 拉格朗日力學

```python
from sympy import symbols, Function
from sympy.physics.mechanics import dynamicsymbols, LagrangesMethod

# 定義廣義座標
q = dynamicsymbols('q')
qd = dynamicsymbols('q', 1)  # q 點（速度）

# 定義拉格朗日量（L = T - V）
from sympy import Rational
m, g, l = symbols('m g l')
T = Rational(1, 2) * m * (l * qd)**2  # 動能
V = m * g * l * (1 - cos(q))           # 位能
L = T - V

# 應用拉格朗日方法
LM = LagrangesMethod(L, [q])
LM.form_lagranges_equations()
eqs = LM.rhs()  # 運動方程式的右邊
```

### Kane 方法

```python
from sympy.physics.mechanics import KanesMethod, ReferenceFrame, Point
from sympy.physics.vector import dynamicsymbols

# 定義系統
N = ReferenceFrame('N')
q = dynamicsymbols('q')
u = dynamicsymbols('u')  # 廣義速度

# 建立 Kane 方程式
kd = [u - q.diff()]  # 運動微分方程式
KM = KanesMethod(N, [q], [u], kd)

# 定義力和物體
# ...（定義粒子、力等）
# KM.kanes_equations(bodies, loads)
```

### 系統本體和慣量

```python
from sympy.physics.mechanics import RigidBody, Inertia, Point, ReferenceFrame
from sympy import symbols

# 質量和慣量參數
m = symbols('m')
Ixx, Iyy, Izz = symbols('I_xx I_yy I_zz')

# 建立參考座標系和質心
A = ReferenceFrame('A')
P = Point('P')

# 定義慣量並矢
I = Inertia(A, Ixx, Iyy, Izz)

# 建立剛體
body = RigidBody('Body', P, A, m, (I, P))
```

### 關節框架

```python
from sympy.physics.mechanics import Body, PinJoint, PrismaticJoint

# 建立本體
parent = Body('P')
child = Body('C')

# 建立銷（旋轉）關節
pin = PinJoint('pin', parent, child)

# 建立滑動（移動）關節
slider = PrismaticJoint('slider', parent, child, axis=parent.frame.z)
```

### 線性化

```python
# 在平衡點附近線性化運動方程式
operating_point = {q: 0, u: 0}  # 平衡點
A, B = KM.linearize(q_ind=[q], u_ind=[u],
                     A_and_B=True,
                     op_point=operating_point)
# A：狀態矩陣，B：輸入矩陣
```

## 量子力學

### 態和算符

```python
from sympy.physics.quantum import Ket, Bra, Operator, Dagger

# 定義態
psi = Ket('psi')
phi = Ket('phi')

# 左矢態
bra_psi = Bra('psi')

# 算符
A = Operator('A')
B = Operator('B')

# Hermitian 共軛
A_dag = Dagger(A)

# 內積
inner = bra_psi * psi
```

### 對易子和反對易子

```python
from sympy.physics.quantum import Commutator, AntiCommutator

# 對易子 [A, B] = AB - BA
comm = Commutator(A, B)
comm.doit()

# 反對易子 {A, B} = AB + BA
anti = AntiCommutator(A, B)
anti.doit()
```

### 量子諧振子

```python
from sympy.physics.quantum.qho_1d import RaisingOp, LoweringOp, NumberOp

# 產生和湮滅算符
a_dag = RaisingOp('a')  # 產生算符
a = LoweringOp('a')      # 湮滅算符
N = NumberOp('N')        # 數算符

# 數態
from sympy.physics.quantum.qho_1d import Ket as QHOKet
n = QHOKet('n')
```

### 自旋系統

```python
from sympy.physics.quantum.spin import (
    JzKet, JxKet, JyKet,  # 自旋態
    Jz, Jx, Jy,            # 自旋算符
    J2                     # 總角動量平方
)

# 自旋 1/2 態
from sympy import Rational
psi = JzKet(Rational(1, 2), Rational(1, 2))  # |1/2, 1/2⟩

# 應用算符
result = Jz * psi
```

### 量子閘

```python
from sympy.physics.quantum.gate import (
    H,      # Hadamard 閘
    X, Y, Z,  # Pauli 閘
    CNOT,    # 受控非閘
    SWAP     # 交換閘
)

# 將閘應用於量子態
from sympy.physics.quantum.qubit import Qubit
q = Qubit('01')
result = H(0) * q  # 將 Hadamard 應用於量子位元 0
```

### 量子演算法

```python
from sympy.physics.quantum.grover import grover_iteration, OracleGate

# Grover 演算法元件可用
# from sympy.physics.quantum.shor import <components>
# Shor 演算法元件可用
```

## 單位和維度

### 使用單位

```python
from sympy.physics.units import (
    meter, kilogram, second,
    newton, joule, watt,
    convert_to
)

# 定義量
distance = 5 * meter
mass = 10 * kilogram
time = 2 * second

# 計算力
force = mass * distance / time**2

# 轉換單位
force_in_newtons = convert_to(force, newton)
```

### 單位系統

```python
from sympy.physics.units import SI, gravitational_constant, speed_of_light

# SI 單位
print(SI._base_units)  # 基本 SI 單位

# 物理常數
G = gravitational_constant
c = speed_of_light
```

### 自訂單位

```python
from sympy.physics.units import Quantity, meter, second

# 定義自訂單位
parsec = Quantity('parsec')
parsec.set_global_relative_scale_factor(3.0857e16 * meter, meter)
```

### 維度分析

```python
from sympy.physics.units import Dimension, length, time, mass

# 檢查維度
from sympy.physics.units import convert_to, meter, second
velocity = 10 * meter / second
print(velocity.dimension)  # Dimension(length/time)
```

## 光學

### 高斯光學

```python
from sympy.physics.optics import (
    BeamParameter,
    FreeSpace,
    FlatRefraction,
    CurvedRefraction,
    ThinLens
)

# 高斯光束參數
q = BeamParameter(wavelen=532e-9, z=0, w=1e-3)

# 通過自由空間傳播
q_new = FreeSpace(1) * q

# 薄透鏡
q_focused = ThinLens(f=0.1) * q
```

### 波和偏振

```python
from sympy.physics.optics import TWave

# 平面波
wave = TWave(amplitude=1, frequency=5e14, phase=0)

# 介質性質（折射率等）
from sympy.physics.optics import Medium
medium = Medium('glass', permittivity=2.25)
```

## 連續介質力學

### 梁分析

```python
from sympy.physics.continuum_mechanics.beam import Beam
from sympy import symbols

# 定義梁
E, I = symbols('E I', positive=True)  # 楊氏模數、慣性矩
length = 10

beam = Beam(length, E, I)

# 施加載荷
from sympy.physics.continuum_mechanics.beam import Beam
beam.apply_load(-1000, 5, -1)  # 在 x=5 處施加 -1000 的點載荷

# 計算反力
beam.solve_for_reaction_loads()

# 取得剪力、彎矩、撓度
x = symbols('x')
shear = beam.shear_force()
moment = beam.bending_moment()
deflection = beam.deflection()
```

### 桁架分析

```python
from sympy.physics.continuum_mechanics.truss import Truss

# 建立桁架
truss = Truss()

# 添加節點
truss.add_node(('A', 0, 0), ('B', 4, 0), ('C', 2, 3))

# 添加構件
truss.add_member(('AB', 'A', 'B'), ('BC', 'B', 'C'))

# 施加載荷
truss.apply_load(('C', 1000, 270))  # 在節點 C 施加 1000 N 於 270° 方向

# 求解
truss.solve()
```

### 纜索分析

```python
from sympy.physics.continuum_mechanics.cable import Cable

# 建立纜索
cable = Cable(('A', 0, 10), ('B', 10, 10))

# 施加載荷
cable.apply_load(-1, 5)  # 分布載荷

# 求解張力和形狀
cable.solve()
```

## 控制系統

### 轉移函數和狀態空間

```python
from sympy.physics.control import TransferFunction, StateSpace
from sympy.abc import s

# 轉移函數
tf = TransferFunction(s + 1, s**2 + 2*s + 1, s)

# 狀態空間表示
A = [[0, 1], [-1, -2]]
B = [[0], [1]]
C = [[1, 0]]
D = [[0]]

ss = StateSpace(A, B, C, D)

# 表示間轉換
ss_from_tf = tf.to_statespace()
tf_from_ss = ss.to_TransferFunction()
```

### 系統分析

```python
# 極點和零點
poles = tf.poles()
zeros = tf.zeros()

# 穩定性
is_stable = tf.is_stable()

# 步階響應、脈衝響應等
# （通常需要數值評估）
```

## 生物力學

### 肌腱模型

```python
from sympy.physics.biomechanics import (
    MusculotendonDeGroote2016,
    FirstOrderActivationDeGroote2016
)

# 建立肌腱模型
mt = MusculotendonDeGroote2016('muscle')

# 活化動力學
activation = FirstOrderActivationDeGroote2016('muscle_activation')
```

## 高能物理

### 粒子物理

```python
# Gamma 矩陣和 Dirac 方程式
from sympy.physics.hep.gamma_matrices import GammaMatrix

gamma0 = GammaMatrix(0)
gamma1 = GammaMatrix(1)
```

## 常見物理模式

### 模式 1：設定力學問題

```python
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point
from sympy import symbols

# 1. 定義參考座標系
N = ReferenceFrame('N')

# 2. 定義廣義座標
q = dynamicsymbols('q')
q_dot = dynamicsymbols('q', 1)

# 3. 定義點和向量
O = Point('O')
P = Point('P')

# 4. 設定運動學
P.set_pos(O, length * N.x)
P.set_vel(N, length * q_dot * N.x)

# 5. 定義力並應用拉格朗日或 Kane 方法
```

### 模式 2：量子態操作

```python
from sympy.physics.quantum import Ket, Operator, qapply

# 定義態
psi = Ket('psi')

# 定義算符
H = Operator('H')  # Hamiltonian

# 應用算符
result = qapply(H * psi)
```

### 模式 3：單位轉換工作流程

```python
from sympy.physics.units import convert_to, meter, foot, second, minute

# 定義含單位的量
distance = 100 * meter
time = 5 * minute

# 執行計算
speed = distance / time

# 轉換為所需單位
speed_m_per_s = convert_to(speed, meter/second)
speed_ft_per_min = convert_to(speed, foot/minute)
```

### 模式 4：梁撓度分析

```python
from sympy.physics.continuum_mechanics.beam import Beam
from sympy import symbols

E, I = symbols('E I', positive=True, real=True)
beam = Beam(10, E, I)

# 施加邊界條件
beam.apply_support(0, 'pin')
beam.apply_support(10, 'roller')

# 施加載荷
beam.apply_load(-1000, 5, -1)  # 點載荷
beam.apply_load(-50, 0, 0, 10)  # 分布載荷

# 求解
beam.solve_for_reaction_loads()

# 取得特定位置的結果
x = 5
deflection_at_mid = beam.deflection().subs(symbols('x'), x)
```

## 重要注意事項

1. **時變變數：**在力學問題中使用 `dynamicsymbols()` 表示時變量。

2. **單位：**始終使用 `sympy.physics.units` 模組明確指定物理計算的單位。

3. **參考座標系：**清楚定義參考座標系及其相對方位以進行向量分析。

4. **數值評估：**許多物理計算需要數值評估。使用 `evalf()` 或轉換為 NumPy 進行數值工作。

5. **假設：**使用適當的符號假設（例如 `positive=True`、`real=True`）幫助 SymPy 正確簡化物理運算式。
