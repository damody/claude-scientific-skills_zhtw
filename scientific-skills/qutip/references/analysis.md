# QuTiP 分析和測量

## 期望值

### 基本期望值

```python
from qutip import *
import numpy as np

# 單一運算子
psi = coherent(N, 2)
n_avg = expect(num(N), psi)

# 多個運算子
ops = [num(N), destroy(N), create(N)]
results = expect(ops, psi)  # 返回列表
```

### 密度矩陣的期望值

```python
# 適用於純態和密度矩陣
rho = thermal_dm(N, 2)
n_avg = expect(num(N), rho)
```

### 變異數

```python
# 計算可觀測量的變異數
var_n = variance(num(N), psi)

# 手動計算
var_n = expect(num(N)**2, psi) - expect(num(N), psi)**2
```

### 時間相依期望值

```python
# 時間演化期間
result = mesolve(H, psi0, tlist, c_ops, e_ops=[num(N)])
n_t = result.expect[0]  # 每個時間的 ⟨n⟩ 陣列
```

## 熵度量

### von Neumann 熵

```python
from qutip import entropy_vn

# 密度矩陣熵
rho = thermal_dm(N, 2)
S = entropy_vn(rho)  # 返回 S = -Tr(ρ log₂ ρ)
```

### 線性熵

```python
from qutip import entropy_linear

# 線性熵 S_L = 1 - Tr(ρ²)
S_L = entropy_linear(rho)
```

### 糾纏熵

```python
# 用於雙分系統
psi = bell_state('00')
rho = psi.proj()

# 對子系統 B 取偏跡以取得約化密度矩陣
rho_A = ptrace(rho, 0)

# 糾纏熵
S_ent = entropy_vn(rho_A)
```

### 互資訊

```python
from qutip import entropy_mutual

# 對於雙分態 ρ_AB
I = entropy_mutual(rho, [0, 1])  # I(A:B) = S(A) + S(B) - S(AB)
```

### 條件熵

```python
from qutip import entropy_conditional

# S(A|B) = S(AB) - S(B)
S_cond = entropy_conditional(rho, 0)  # 給定子系統 1 的子系統 0 的熵
```

## 保真度和距離度量

### 態保真度

```python
from qutip import fidelity

# 兩個態之間的保真度
psi1 = coherent(N, 2)
psi2 = coherent(N, 2.1)

F = fidelity(psi1, psi2)  # 返回 [0, 1] 中的值
```

### 過程保真度

```python
from qutip import process_fidelity

# 兩個過程（超運算子）之間的保真度
U_ideal = (-1j * H * t).expm()
U_actual = mesolve(H, basis(N, 0), [0, t], c_ops).states[-1]

F_proc = process_fidelity(U_ideal, U_actual)
```

### 跡距離

```python
from qutip import tracedist

# 跡距離 D = (1/2) Tr|ρ₁ - ρ₂|
rho1 = coherent_dm(N, 2)
rho2 = thermal_dm(N, 2)

D = tracedist(rho1, rho2)  # 返回 [0, 1] 中的值
```

### Hilbert-Schmidt 距離

```python
from qutip import hilbert_dist

# Hilbert-Schmidt 距離
D_HS = hilbert_dist(rho1, rho2)
```

### Bures 距離

```python
from qutip import bures_dist

# Bures 距離
D_B = bures_dist(rho1, rho2)
```

### Bures 角

```python
from qutip import bures_angle

# Bures 角
angle = bures_angle(rho1, rho2)
```

## 糾纏度量

### 糾纏度（Concurrence）

```python
from qutip import concurrence

# 用於兩量子位元態
psi = bell_state('00')
rho = psi.proj()

C = concurrence(rho)  # 最大糾纏態 C = 1
```

### 負性（Negativity）

```python
from qutip import negativity

# 負性（部分轉置準則）
N_ent = negativity(rho, 0)  # 對子系統 0 的部分轉置

# 對數負性
from qutip import logarithmic_negativity
E_N = logarithmic_negativity(rho, 0)
```

### 糾纏能力

```python
from qutip import entangling_power

# 用於么正閘
U = cnot()
E_pow = entangling_power(U)
```

## 純度度量

### 純度

```python
# 純度 P = Tr(ρ²)
P = (rho * rho).tr()

# 純態：P = 1
# 最大混合態：P = 1/d
```

### 檢查態屬性

```python
# 態是純態嗎？
is_pure = abs((rho * rho).tr() - 1.0) < 1e-10

# 運算子是厄米的嗎？
H.isherm

# 運算子是么正的嗎？
U.check_isunitary()
```

## 測量

### 投影測量

```python
from qutip import measurement

# 在計算基底中測量
psi = (basis(2, 0) + basis(2, 1)).unit()

# 執行測量
result, state_after = measurement.measure(psi, None)  # 隨機結果

# 特定測量運算子
M = basis(2, 0).proj()
prob = measurement.measure_povm(psi, [M, qeye(2) - M])
```

### 測量統計

```python
from qutip import measurement_statistics

# 取得所有可能結果和機率
outcomes, probabilities = measurement_statistics(psi, [M0, M1])
```

### 可觀測量測量

```python
from qutip import measure_observable

# 測量可觀測量並取得結果 + 坍縮態
result, state_collapsed = measure_observable(psi, sigmaz())
```

### POVM 測量

```python
from qutip import measure_povm

# 正值運算子度量
E_0 = Qobj([[0.8, 0], [0, 0.2]])
E_1 = Qobj([[0.2, 0], [0, 0.8]])

result, state_after = measure_povm(psi, [E_0, E_1])
```

## 相干度量

### l1-範數相干

```python
from qutip import coherence_l1norm

# 非對角元素的 l1-範數
C_l1 = coherence_l1norm(rho)
```

## 關聯函數

### 雙時關聯

```python
from qutip import correlation_2op_1t, correlation_2op_2t

# 單時關聯 ⟨A(t+τ)B(t)⟩
A = destroy(N)
B = create(N)
taulist = np.linspace(0, 10, 200)

corr = correlation_2op_1t(H, rho0, taulist, c_ops, A, B)

# 雙時關聯 ⟨A(t)B(τ)⟩
tlist = np.linspace(0, 10, 100)
corr_2t = correlation_2op_2t(H, rho0, tlist, taulist, c_ops, A, B)
```

### 三運算子關聯

```python
from qutip import correlation_3op_1t

# ⟨A(t)B(t+τ)C(t)⟩
C_op = num(N)
corr_3 = correlation_3op_1t(H, rho0, taulist, c_ops, A, B, C_op)
```

### 四運算子關聯

```python
from qutip import correlation_4op_1t

# ⟨A(0)B(τ)C(τ)D(0)⟩
D_op = create(N)
corr_4 = correlation_4op_1t(H, rho0, taulist, c_ops, A, B, C_op, D_op)
```

## 頻譜分析

### FFT 頻譜

```python
from qutip import spectrum_correlation_fft

# 從關聯函數計算功率譜
w, S = spectrum_correlation_fft(taulist, corr)
```

### 直接頻譜計算

```python
from qutip import spectrum

# 發射/吸收頻譜
wlist = np.linspace(0, 2, 200)
spec = spectrum(H, wlist, c_ops, A, B)
```

### 偽模式

```python
from qutip import spectrum_pi

# 帶偽模式分解的頻譜
spec_pi = spectrum_pi(H, rho0, wlist, c_ops, A, B)
```

## 穩態分析

### 尋找穩態

```python
from qutip import steadystate

# 尋找穩態 ∂ρ/∂t = 0
rho_ss = steadystate(H, c_ops)

# 不同方法
rho_ss = steadystate(H, c_ops, method='direct')  # 預設
rho_ss = steadystate(H, c_ops, method='eigen')   # 本徵值
rho_ss = steadystate(H, c_ops, method='svd')     # SVD
rho_ss = steadystate(H, c_ops, method='power')   # 冪法
```

### 穩態屬性

```python
# 驗證它是穩態
L = liouvillian(H, c_ops)
assert (L * operator_to_vector(rho_ss)).norm() < 1e-10

# 計算穩態期望值
n_ss = expect(num(N), rho_ss)
```

## 量子 Fisher 資訊

```python
from qutip import qfisher

# 量子 Fisher 資訊
F_Q = qfisher(rho, num(N))  # 相對於生成元 num(N)
```

## 矩陣分析

### 本徵分析

```python
# 本徵值和本徵向量
evals, ekets = H.eigenstates()

# 只有本徵值
evals = H.eigenenergies()

# 基態
E0, psi0 = H.groundstate()
```

### 矩陣函數

```python
# 矩陣指數
U = (H * t).expm()

# 矩陣對數
log_rho = rho.logm()

# 矩陣平方根
sqrt_rho = rho.sqrtm()

# 矩陣冪
rho_squared = rho ** 2
```

### 奇異值分解

```python
# 運算子的 SVD
U, S, Vh = H.svd()
```

### 置換

```python
from qutip import permute

# 置換子系統
rho_permuted = permute(rho, [1, 0])  # 交換子系統
```

## 偏操作

### 偏跡

```python
# 約化到子系統
rho_A = ptrace(rho_AB, 0)  # 保留子系統 0
rho_B = ptrace(rho_AB, 1)  # 保留子系統 1

# 保留多個子系統
rho_AC = ptrace(rho_ABC, [0, 2])  # 保留 0 和 2，對 1 取跡
```

### 偏轉置

```python
from qutip import partial_transpose

# 偏轉置（用於糾纏偵測）
rho_pt = partial_transpose(rho, [0, 1])  # 轉置子系統 0

# 檢查是否糾纏（PPT 準則）
evals = rho_pt.eigenenergies()
is_entangled = any(evals < -1e-10)
```

## 量子態層析

### 態重建

```python
from qutip_qip.tomography import state_tomography

# 準備測量結果
# measurements = ... （實驗資料）

# 重建密度矩陣
rho_reconstructed = state_tomography(measurements, basis='Pauli')
```

### 過程層析

```python
from qutip_qip.tomography import qpt

# 表徵量子過程
chi = qpt(U_gate, method='lstsq')  # Chi 矩陣表示
```

## 隨機量子物件

用於測試和蒙地卡羅模擬。

```python
# 隨機態向量
psi_rand = rand_ket(N)

# 隨機密度矩陣
rho_rand = rand_dm(N)

# 隨機厄米運算子
H_rand = rand_herm(N)

# 隨機么正
U_rand = rand_unitary(N)

# 具有特定屬性
rho_rank2 = rand_dm(N, rank=2)  # 秩為 2 的密度矩陣
H_sparse = rand_herm(N, density=0.1)  # 10% 非零元素
```

## 有用的檢查

```python
# 檢查運算子是否厄米
H.isherm

# 檢查態是否正規化
abs(psi.norm() - 1.0) < 1e-10

# 檢查密度矩陣是否物理
rho.tr() ≈ 1 and all(rho.eigenenergies() >= 0)

# 檢查運算子是否對易
commutator(A, B).norm() < 1e-10
```
