# QuTiP 時間演化和動力學求解器

## 概述

QuTiP 提供多個用於量子動力學的求解器：
- `sesolve` - 薛丁格方程（么正演化）
- `mesolve` - 主方程（帶耗散的開放系統）
- `mcsolve` - 蒙地卡羅（量子軌跡）
- `brmesolve` - Bloch-Redfield 主方程
- `fmmesolve` - Floquet-Markov 主方程
- `ssesolve/smesolve` - 隨機薛丁格/主方程

## 薛丁格方程求解器（sesolve）

用於么正演化的封閉量子系統。

### 基本用法

```python
from qutip import *
import numpy as np

# 系統設定
N = 10
psi0 = basis(N, 0)  # 初始態
H = num(N)  # 哈密頓量

# 時間點
tlist = np.linspace(0, 10, 100)

# 求解
result = sesolve(H, psi0, tlist)

# 存取結果
states = result.states  # 每個時間的態列表
final_state = result.states[-1]
```

### 帶期望值

```python
# 要計算期望值的運算子
e_ops = [num(N), destroy(N), create(N)]

result = sesolve(H, psi0, tlist, e_ops=e_ops)

# 存取期望值
n_t = result.expect[0]  # ⟨n⟩(t)
a_t = result.expect[1]  # ⟨a⟩(t)
```

### 時間相依哈密頓量

```python
# 方法 1：基於字串（更快，需要 Cython）
H = [num(N), [destroy(N) + create(N), 'cos(w*t)']]
args = {'w': 1.0}
result = sesolve(H, psi0, tlist, args=args)

# 方法 2：基於函數
def drive(t, args):
    return np.exp(-t/args['tau']) * np.sin(args['w'] * t)

H = [num(N), [destroy(N) + create(N), drive]]
args = {'w': 1.0, 'tau': 5.0}
result = sesolve(H, psi0, tlist, args=args)

# 方法 3：QobjEvo（最靈活）
from qutip import QobjEvo
H_td = QobjEvo([num(N), [destroy(N) + create(N), drive]], args=args)
result = sesolve(H_td, psi0, tlist)
```

## 主方程求解器（mesolve）

用於帶耗散和退相干的開放量子系統。

### 基本用法

```python
# 系統哈密頓量
H = num(N)

# 坍縮運算子（Lindblad 運算子）
kappa = 0.1  # 衰減率
c_ops = [np.sqrt(kappa) * destroy(N)]

# 初始態
psi0 = coherent(N, 2.0)

# 求解
result = mesolve(H, psi0, tlist, c_ops, e_ops=[num(N)])

# 結果是密度矩陣演化
rho_t = result.states  # 密度矩陣列表
n_t = result.expect[0]  # ⟨n⟩(t)
```

### 多重耗散通道

```python
# 光子損失
kappa = 0.1
# 退相位
gamma = 0.05
# 熱激發
nth = 0.5  # 熱光子數

c_ops = [
    np.sqrt(kappa * (1 + nth)) * destroy(N),  # 熱衰減
    np.sqrt(kappa * nth) * create(N),  # 熱激發
    np.sqrt(gamma) * num(N)  # 純退相位
]

result = mesolve(H, psi0, tlist, c_ops)
```

### 時間相依耗散

```python
# 時間相依衰減率
def kappa_t(t, args):
    return args['k0'] * (1 + np.sin(args['w'] * t))

c_ops = [[np.sqrt(1.0) * destroy(N), kappa_t]]
args = {'k0': 0.1, 'w': 1.0}

result = mesolve(H, psi0, tlist, c_ops, args=args)
```

## 蒙地卡羅求解器（mcsolve）

模擬開放系統的量子軌跡。

### 基本用法

```python
# 與 mesolve 相同設定
H = num(N)
c_ops = [np.sqrt(0.1) * destroy(N)]
psi0 = coherent(N, 2.0)

# 軌跡數
ntraj = 500

result = mcsolve(H, psi0, tlist, c_ops, e_ops=[num(N)], ntraj=ntraj)

# 軌跡平均結果
n_avg = result.expect[0]
n_std = result.std_expect[0]  # 標準差

# 個別軌跡（如果 options.store_states=True）
options = Options(store_states=True)
result = mcsolve(H, psi0, tlist, c_ops, ntraj=ntraj, options=options)
trajectories = result.states  # 軌跡列表的列表
```

### 光子計數

```python
# 追蹤量子跳躍
result = mcsolve(H, psi0, tlist, c_ops, ntraj=ntraj, options=options)

# 存取跳躍時間和引起跳躍的運算子
for traj in result.col_times:
    print(f"跳躍時間: {traj}")

for traj in result.col_which:
    print(f"跳躍運算子索引: {traj}")
```

## Bloch-Redfield 求解器（brmesolve）

用於久期近似中的弱系統-熱浴耦合。

```python
# 系統哈密頓量
H = sigmaz()

# 耦合運算子和譜密度
a_ops = [[sigmax(), lambda w: 0.1 * w if w > 0 else 0]]  # 歐姆熱浴

psi0 = basis(2, 0)
result = brmesolve(H, psi0, tlist, a_ops, e_ops=[sigmaz(), sigmax()])
```

## Floquet 求解器（fmmesolve）

用於時間週期哈密頓量。

```python
# 時間週期哈密頓量
w_d = 1.0  # 驅動頻率
H0 = sigmaz()
H1 = sigmax()
H = [H0, [H1, 'cos(w*t)']]
args = {'w': w_d}

# Floquet 模式和準能量
T = 2 * np.pi / w_d  # 週期
f_modes, f_energies = floquet_modes(H, T, args)

# Floquet 基底中的初始態
psi0 = basis(2, 0)

# Floquet 基底中的耗散
c_ops = [np.sqrt(0.1) * sigmam()]

result = fmmesolve(H, psi0, tlist, c_ops, e_ops=[num(2)], T=T, args=args)
```

## 隨機求解器

### 隨機薛丁格方程（ssesolve）

```python
# 擴散運算子
sc_ops = [np.sqrt(0.1) * destroy(N)]

# 外差偵測
result = ssesolve(H, psi0, tlist, sc_ops=sc_ops, e_ops=[num(N)],
                   ntraj=500, noise=1)  # noise=1 用於外差
```

### 隨機主方程（smesolve）

```python
result = smesolve(H, psi0, tlist, c_ops=[], sc_ops=sc_ops,
                   e_ops=[num(N)], ntraj=500)
```

## 傳播子

### 時間演化運算子

```python
# 演化運算子 U(t) 使得 ψ(t) = U(t)ψ(0)
U = (-1j * H * t).expm()
psi_t = U * psi0

# 對於主方程（超運算子傳播子）
L = liouvillian(H, c_ops)
U_super = (L * t).expm()
rho_t = vector_to_operator(U_super * operator_to_vector(rho0))
```

### 傳播子函數

```python
# 生成多個時間的傳播子
U_list = propagator(H, tlist, c_ops)

# 應用於態
psi_t = [U_list[i] * psi0 for i in range(len(tlist))]
```

## 穩態解

### 直接穩態

```python
# 尋找劉維爾運算子的穩態
rho_ss = steadystate(H, c_ops)

# 驗證它是穩態
L = liouvillian(H, c_ops)
assert (L * operator_to_vector(rho_ss)).norm() < 1e-10
```

### 偽逆方法

```python
# 對於簡併穩態
rho_ss = steadystate(H, c_ops, method='direct')
# 或 'eigen'、'svd'、'power'
```

## 關聯函數

### 雙時關聯

```python
# ⟨A(t+τ)B(t)⟩
A = destroy(N)
B = create(N)

# 發射譜
taulist = np.linspace(0, 10, 200)
corr = correlation_2op_1t(H, None, taulist, c_ops, A, B)

# 功率譜
w, S = spectrum_correlation_fft(taulist, corr)
```

### 多時關聯

```python
# ⟨A(t3)B(t2)C(t1)⟩
corr = correlation_3op_1t(H, None, taulist, c_ops, A, B, C)
```

## 求解器選項

```python
from qutip import Options

options = Options()
options.nsteps = 10000  # 最大內部步數
options.atol = 1e-8  # 絕對容差
options.rtol = 1e-6  # 相對容差
options.method = 'adams'  # 或 'bdf' 用於剛性問題
options.store_states = True  # 儲存所有態
options.store_final_state = True  # 只儲存最終態

result = mesolve(H, psi0, tlist, c_ops, options=options)
```

### 進度條

```python
options.progress_bar = True
result = mesolve(H, psi0, tlist, c_ops, options=options)
```

## 儲存和載入結果

```python
# 儲存結果
result.save("my_simulation.dat")

# 載入結果
from qutip import Result
loaded_result = Result.load("my_simulation.dat")
```

## 高效模擬技巧

1. **稀疏矩陣**：QuTiP 自動使用稀疏矩陣
2. **小希爾伯特空間**：盡可能截斷
3. **時間相依項**：字串格式最快（需要編譯）
4. **平行軌跡**：mcsolve 自動平行化
5. **收斂**：透過變化 `ntraj`、`nsteps`、容差來檢查
6. **求解器選擇**：
   - 純態：使用 `sesolve`（更快）
   - 混合態/耗散：使用 `mesolve`
   - 雜訊/測量：使用 `mcsolve`
   - 弱耦合：使用 `brmesolve`
   - 週期驅動：使用 Floquet 方法
