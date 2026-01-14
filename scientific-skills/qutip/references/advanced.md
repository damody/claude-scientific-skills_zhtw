# QuTiP 進階功能

## Floquet 理論

用於時間週期哈密頓量 H(t + T) = H(t)。

### Floquet 模式和準能量

```python
from qutip import *
import numpy as np

# 時間週期哈密頓量
w_d = 1.0  # 驅動頻率
T = 2 * np.pi / w_d  # 週期

H0 = sigmaz()
H1 = sigmax()
H = [H0, [H1, 'cos(w*t)']]
args = {'w': w_d}

# 計算 Floquet 模式和準能量
f_modes, f_energies = floquet_modes(H, T, args)

print("準能量:", f_energies)
print("Floquet 模式:", f_modes)
```

### 時間 t 處的 Floquet 態

```python
# 取得特定時間的 Floquet 態
t = 1.0
f_states_t = floquet_states(f_modes, f_energies, t)
```

### Floquet 態分解

```python
# 在 Floquet 基底中分解初始態
psi0 = basis(2, 0)
f_coeff = floquet_state_decomposition(f_modes, f_energies, psi0)
```

### Floquet-Markov 主方程

```python
# 帶耗散的時間演化
c_ops = [np.sqrt(0.1) * sigmam()]
tlist = np.linspace(0, 20, 200)

result = fmmesolve(H, psi0, tlist, c_ops, e_ops=[sigmaz()], T=T, args=args)

# 繪製結果
import matplotlib.pyplot as plt
plt.plot(tlist, result.expect[0])
plt.xlabel('時間')
plt.ylabel('⟨σz⟩')
plt.show()
```

### Floquet 張量

```python
# Floquet 張量（廣義 Bloch-Redfield）
A_ops = [[sigmaz(), lambda w: 0.1 * w if w > 0 else 0]]

# 建構 Floquet 張量
R, U = floquet_markov_mesolve(H, psi0, tlist, A_ops, e_ops=[sigmaz()],
                               T=T, args=args)
```

### 有效哈密頓量

```python
# 時間平均有效哈密頓量
H_eff = floquet_master_equation_steadystate(H, c_ops, T, args)
```

## 層級運動方程（HEOM）

用於強系統-熱浴耦合的非馬可夫開放量子系統。

### 基本 HEOM 設定

```python
from qutip import heom

# 系統哈密頓量
H_sys = sigmaz()

# 熱浴關聯函數（指數）
Q = sigmax()  # 系統-熱浴耦合運算子
ck_real = [0.1]  # 耦合強度
vk_real = [0.5]  # 熱浴頻率

# HEOM 熱浴
bath = heom.BosonicBath(Q, ck_real, vk_real)

# 初始態
rho0 = basis(2, 0) * basis(2, 0).dag()

# 建立 HEOM 求解器
max_depth = 5
hsolver = heom.HEOMSolver(H_sys, [bath], max_depth=max_depth)

# 時間演化
tlist = np.linspace(0, 10, 100)
result = hsolver.run(rho0, tlist)

# 提取約化系統密度矩陣
rho_sys = [r.extract_state(0) for r in result.states]
```

### 多重熱浴

```python
# 定義多重熱浴
bath1 = heom.BosonicBath(sigmax(), [0.1], [0.5])
bath2 = heom.BosonicBath(sigmay(), [0.05], [1.0])

hsolver = heom.HEOMSolver(H_sys, [bath1, bath2], max_depth=5)
```

### Drude-Lorentz 譜密度

```python
# 在凝聚態物理中常見
from qutip.nonmarkov.heom import DrudeLorentzBath

lam = 0.1  # 重組能
gamma = 0.5  # 熱浴截止頻率
T = 1.0  # 溫度（能量單位）
Nk = 2  # Matsubara 項數

bath = DrudeLorentzBath(Q, lam, gamma, T, Nk)
```

### HEOM 選項

```python
options = heom.HEOMSolver.Options(
    nsteps=2000,
    store_states=True,
    rtol=1e-7,
    atol=1e-9
)

hsolver = heom.HEOMSolver(H_sys, [bath], max_depth=5, options=options)
```

## 置換不變性

用於全同粒子系統（例如自旋系綜）。

### Dicke 態

```python
from qutip import dicke

# N 個自旋的 Dicke 態 |j, m⟩
N = 10  # 自旋數
j = N/2  # 總角動量
m = 0   # z 分量

psi = dicke(N, j, m)
```

### 置換不變運算子

```python
from qutip.piqs import jspin

# 集體自旋運算子
N = 10
Jx = jspin(N, 'x')
Jy = jspin(N, 'y')
Jz = jspin(N, 'z')
Jp = jspin(N, '+')
Jm = jspin(N, '-')
```

### PIQS 動力學

```python
from qutip.piqs import Dicke

# 設定 Dicke 模型
N = 10
emission = 1.0
dephasing = 0.5
pumping = 0.0
collective_emission = 0.0

system = Dicke(N=N, emission=emission, dephasing=dephasing,
               pumping=pumping, collective_emission=collective_emission)

# 初始態
psi0 = dicke(N, N/2, N/2)  # 所有自旋向上

# 時間演化
tlist = np.linspace(0, 10, 100)
result = system.solve(psi0, tlist, e_ops=[Jz])
```

## 非馬可夫蒙地卡羅

具有記憶效應的量子軌跡。

```python
from qutip import nm_mcsolve

# 非馬可夫熱浴關聯
def bath_correlation(t1, t2):
    tau = abs(t2 - t1)
    return np.exp(-tau / 2.0) * np.cos(tau)

# 系統設定
H = sigmaz()
c_ops = [sigmax()]
psi0 = basis(2, 0)
tlist = np.linspace(0, 10, 100)

# 帶記憶求解
result = nm_mcsolve(H, psi0, tlist, c_ops, sc_ops=[],
                     bath_corr=bath_correlation, ntraj=500,
                     e_ops=[sigmaz()])
```

## 帶測量的隨機求解器

### 連續測量

```python
# 同位相偵測
sc_ops = [np.sqrt(0.1) * destroy(N)]  # 測量運算子

result = ssesolve(H, psi0, tlist, sc_ops=sc_ops,
                   e_ops=[num(N)], ntraj=100,
                   noise=11)  # 11 用於同位相

# 外差偵測
result = ssesolve(H, psi0, tlist, sc_ops=sc_ops,
                   e_ops=[num(N)], ntraj=100,
                   noise=12)  # 12 用於外差
```

### 光子計數

```python
# 量子跳躍時間
result = mcsolve(H, psi0, tlist, c_ops, ntraj=50,
                 options=Options(store_states=True))

# 提取測量時間
for i, jump_times in enumerate(result.col_times):
    print(f"軌跡 {i} 跳躍時間: {jump_times}")
    print(f"哪個運算子: {result.col_which[i]}")
```

## Krylov 子空間方法

對大型系統高效。

```python
from qutip import krylovsolve

# 使用 Krylov 求解器
result = krylovsolve(H, psi0, tlist, krylov_dim=10, e_ops=[num(N)])
```

## Bloch-Redfield 主方程

用於弱系統-熱浴耦合。

```python
# 熱浴譜密度
def ohmic_spectrum(w):
    if w >= 0:
        return 0.1 * w  # 歐姆式
    else:
        return 0

# 耦合運算子和譜
a_ops = [[sigmax(), ohmic_spectrum]]

# 求解
result = brmesolve(H, psi0, tlist, a_ops, e_ops=[sigmaz()])
```

### 溫度相依熱浴

```python
def thermal_spectrum(w):
    # 玻色-愛因斯坦分佈
    T = 1.0  # 溫度
    if abs(w) < 1e-10:
        return 0.1 * T
    n_th = 1 / (np.exp(abs(w)/T) - 1)
    if w >= 0:
        return 0.1 * w * (n_th + 1)
    else:
        return 0.1 * abs(w) * n_th

a_ops = [[sigmax(), thermal_spectrum]]
result = brmesolve(H, psi0, tlist, a_ops, e_ops=[sigmaz()])
```

## 超運算子和量子通道

### 超運算子表示

```python
# 劉維爾運算子
L = liouvillian(H, c_ops)

# 表示之間的轉換
from qutip import (spre, spost, sprepost,
                    super_to_choi, choi_to_super,
                    super_to_kraus, kraus_to_super)

# 超運算子形式
L_spre = spre(H)  # 左乘
L_spost = spost(H)  # 右乘
L_sprepost = sprepost(H, H.dag())

# Choi 矩陣
choi = super_to_choi(L)

# Kraus 運算子
kraus = super_to_kraus(L)
```

### 量子通道

```python
# 去極化通道
p = 0.1  # 錯誤機率
K0 = np.sqrt(1 - 3*p/4) * qeye(2)
K1 = np.sqrt(p/4) * sigmax()
K2 = np.sqrt(p/4) * sigmay()
K3 = np.sqrt(p/4) * sigmaz()

kraus_ops = [K0, K1, K2, K3]
E = kraus_to_super(kraus_ops)

# 應用通道
rho_out = E * operator_to_vector(rho_in)
rho_out = vector_to_operator(rho_out)
```

### 振幅阻尼

```python
# T1 衰減
gamma = 0.1
K0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
K1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])

E_damping = kraus_to_super([K0, K1])
```

### 相位阻尼

```python
# T2 退相位
gamma = 0.1
K0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma/2)]])
K1 = Qobj([[0, 0], [0, np.sqrt(gamma/2)]])

E_dephasing = kraus_to_super([K0, K1])
```

## 量子軌跡分析

### 提取個別軌跡

```python
options = Options(store_states=True, store_final_state=False)
result = mcsolve(H, psi0, tlist, c_ops, ntraj=100, options=options)

# 存取個別軌跡
for i in range(len(result.states)):
    trajectory = result.states[i]  # 軌跡 i 的態列表
    # 分析軌跡
```

### 軌跡統計

```python
# 平均值和標準差
result = mcsolve(H, psi0, tlist, c_ops, e_ops=[num(N)], ntraj=500)

n_mean = result.expect[0]
n_std = result.std_expect[0]

# 最終時間的光子數分佈
final_states = [result.states[i][-1] for i in range(len(result.states))]
```

## 進階時間相依項

### QobjEvo

```python
from qutip import QobjEvo

# 使用 QobjEvo 的時間相依哈密頓量
def drive(t, args):
    return args['A'] * np.exp(-t/args['tau']) * np.sin(args['w'] * t)

H0 = num(N)
H1 = destroy(N) + create(N)
args = {'A': 1.0, 'w': 1.0, 'tau': 5.0}

H_td = QobjEvo([H0, [H1, drive]], args=args)

# 可以更新 args 而不重新建立
H_td.arguments({'A': 2.0, 'w': 1.5, 'tau': 10.0})
```

### 編譯的時間相依項

```python
# 最快方法（需要 Cython）
H = [num(N), [destroy(N) + create(N), 'A * exp(-t/tau) * sin(w*t)']]
args = {'A': 1.0, 'w': 1.0, 'tau': 5.0}

# QuTiP 編譯此以加速
result = sesolve(H, psi0, tlist, args=args)
```

### 回呼函數

```python
# 進階控制
def time_dependent_coeff(t, args):
    # 如果需要可存取求解器狀態
    return complex_function(t, args)

H = [H0, [H1, time_dependent_coeff]]
```

## 平行處理

### 平行映射

```python
from qutip import parallel_map

# 定義任務
def simulate(gamma):
    c_ops = [np.sqrt(gamma) * destroy(N)]
    result = mesolve(H, psi0, tlist, c_ops, e_ops=[num(N)])
    return result.expect[0]

# 平行執行
gamma_values = np.linspace(0, 1, 20)
results = parallel_map(simulate, gamma_values, num_cpus=4)
```

### 序列映射（用於除錯）

```python
from qutip import serial_map

# 相同介面但序列執行
results = serial_map(simulate, gamma_values)
```

## 檔案 I/O

### 儲存/載入量子物件

```python
# 儲存
H.save('hamiltonian.qu')
psi.save('state.qu')

# 載入
H_loaded = qload('hamiltonian.qu')
psi_loaded = qload('state.qu')
```

### 儲存/載入結果

```python
# 儲存模擬結果
result = mesolve(H, psi0, tlist, c_ops, e_ops=[num(N)])
result.save('simulation.dat')

# 載入結果
from qutip import Result
loaded_result = Result.load('simulation.dat')
```

### 匯出到 MATLAB

```python
# 匯出到 .mat 檔
H.matlab_export('hamiltonian.mat', 'H')
```

## 求解器選項

### 微調求解器

```python
options = Options()

# 積分參數
options.nsteps = 10000  # 最大內部步數
options.rtol = 1e-8     # 相對容差
options.atol = 1e-10    # 絕對容差

# 方法選擇
options.method = 'adams'  # 非剛性（預設）
# options.method = 'bdf'  # 剛性問題

# 儲存選項
options.store_states = True
options.store_final_state = True

# 進度
options.progress_bar = True

# 隨機數種子（用於可重現性）
options.seeds = 12345

result = mesolve(H, psi0, tlist, c_ops, options=options)
```

### 除錯

```python
# 啟用詳細輸出
options.verbose = True

# 記憶體追蹤
options.num_cpus = 1  # 更容易除錯
```

## 效能技巧

1. **使用稀疏矩陣**：QuTiP 自動執行此操作
2. **最小化希爾伯特空間**：盡可能截斷
3. **選擇正確的求解器**：
   - 純態：`sesolve` 比 `mesolve` 快
   - 隨機：`mcsolve` 用於量子跳躍
   - 週期：Floquet 方法
4. **時間相依項**：字串格式最快
5. **期望值**：只計算需要的可觀測量
6. **平行軌跡**：`mcsolve` 使用所有 CPU
7. **Krylov 方法**：用於非常大的系統
8. **記憶體**：盡可能使用 `store_final_state` 而不是 `store_states`
