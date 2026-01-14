# QuTiP 核心概念

## 量子物件（Qobj）

QuTiP 中的所有量子物件都由 `Qobj` 類別表示：

```python
from qutip import *

# 建立量子物件
psi = basis(2, 0)  # 二能階系統的基態
rho = fock_dm(5, 2)  # n=2 Fock 態的密度矩陣
H = sigmaz()  # Pauli Z 運算子
```

關鍵屬性：
- `.dims` - 維度結構
- `.shape` - 矩陣維度
- `.type` - 類型（ket、bra、oper、super）
- `.isherm` - 檢查是否厄米
- `.dag()` - 厄米共軛
- `.tr()` - 跡
- `.norm()` - 範數

## 態

### 基態

```python
# Fock（數）態
n = 2  # 激發級別
N = 10  # 希爾伯特空間維度
psi = basis(N, n)  # 或 fock(N, n)

# 相干態
alpha = 1 + 1j
coherent(N, alpha)

# 熱態（密度矩陣）
n_avg = 2.0  # 平均光子數
thermal_dm(N, n_avg)
```

### 自旋態

```python
# 自旋-1/2 態
spin_state(1/2, 1/2)  # 自旋向上
spin_coherent(1/2, theta, phi)  # 相干自旋態

# 多量子位元計算基底
basis([2,2,2], [0,1,0])  # 3 量子位元的 |010⟩
```

### 複合態

```python
# 張量積
psi1 = basis(2, 0)
psi2 = basis(2, 1)
tensor(psi1, psi2)  # |01⟩

# Bell 態
bell_state('00')  # (|00⟩ + |11⟩)/√2
maximally_mixed_dm(2)  # 最大混合態
```

## 運算子

### 產生/湮滅

```python
N = 10
a = destroy(N)  # 湮滅運算子
a_dag = create(N)  # 產生運算子
num = num(N)  # 數運算子 (a†a)
```

### Pauli 矩陣

```python
sigmax()  # σx
sigmay()  # σy
sigmaz()  # σz
sigmap()  # σ+ = (σx + iσy)/2
sigmam()  # σ- = (σx - iσy)/2
```

### 角動量

```python
# 任意 j 的自旋運算子
j = 1  # 自旋-1
jmat(j, 'x')  # Jx
jmat(j, 'y')  # Jy
jmat(j, 'z')  # Jz
jmat(j, '+')  # J+
jmat(j, '-')  # J-
```

### 位移和壓縮

```python
alpha = 1 + 1j
displace(N, alpha)  # 位移運算子 D(α)

z = 0.5  # 壓縮參數
squeeze(N, z)  # 壓縮運算子 S(z)
```

## 張量積和組合

### 建構複合系統

```python
# 運算子的張量積
H1 = sigmaz()
H2 = sigmax()
H_total = tensor(H1, H2)

# 單位運算子
qeye([2, 2])  # 兩量子位元的單位矩陣

# 部分應用
# 3 量子位元系統的 σz ⊗ I
tensor(sigmaz(), qeye(2), qeye(2))
```

### 偏跡

```python
# 複合系統態
rho = bell_state('00').proj()  # |Φ+⟩⟨Φ+|

# 對子系統取跡
rho_A = ptrace(rho, 0)  # 對子系統 0 取跡
rho_B = ptrace(rho, 1)  # 對子系統 1 取跡
```

## 期望值和測量

```python
# 期望值
psi = coherent(N, alpha)
expect(num, psi)  # ⟨n⟩

# 對於多個運算子
ops = [a, a_dag, num]
expect(ops, psi)  # 返回列表

# 變異數
variance(num, psi)  # Var(n) = ⟨n²⟩ - ⟨n⟩²
```

## 超運算子和劉維爾運算子

### Lindblad 形式

```python
# 系統哈密頓量
H = num

# 坍縮運算子（耗散）
c_ops = [np.sqrt(0.1) * a]  # 衰減率 0.1

# 劉維爾超運算子
L = liouvillian(H, c_ops)

# 替代：顯式形式
L = -1j * (spre(H) - spost(H)) + lindblad_dissipator(a, a)
```

### 超運算子表示

```python
# Kraus 表示
kraus_to_super(kraus_ops)

# Choi 矩陣
choi_to_super(choi_matrix)

# Chi（過程）矩陣
chi_to_super(chi_matrix)

# 轉換
super_to_choi(L)
choi_to_kraus(choi_matrix)
```

## 量子閘（需要 qutip-qip）

```python
from qutip_qip.operations import *

# 單量子位元閘
hadamard_transform()  # Hadamard
rx(np.pi/2)  # X 旋轉
ry(np.pi/2)  # Y 旋轉
rz(np.pi/2)  # Z 旋轉
phasegate(np.pi/4)  # 相位閘
snot()  # Hadamard（替代）

# 雙量子位元閘
cnot()  # CNOT
swap()  # SWAP
iswap()  # iSWAP
sqrtswap()  # √SWAP
berkeley()  # Berkeley 閘
swapalpha(alpha)  # SWAP^α

# 三量子位元閘
fredkin()  # 受控 SWAP
toffoli()  # 受控 CNOT

# 擴展到多量子位元系統
N = 3  # 總量子位元數
target = 1
controls = [0, 2]
gate_expand_2toN(cnot(), N, [controls[0], target])
```

## 常見哈密頓量

### Jaynes-Cummings 模型

```python
# 腔模式
N = 10
a = tensor(destroy(N), qeye(2))

# 原子
sm = tensor(qeye(N), sigmam())

# 哈密頓量
wc = 1.0  # 腔頻率
wa = 1.0  # 原子頻率
g = 0.05  # 耦合強度
H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
```

### 驅動系統

```python
# 時間相依哈密頓量
H0 = sigmaz()
H1 = sigmax()

def drive(t, args):
    return np.sin(args['w'] * t)

H = [H0, [H1, drive]]
args = {'w': 1.0}
```

### 自旋鏈

```python
# 海森堡鏈
N_spins = 5
J = 1.0  # 交換耦合

# 建構哈密頓量
H = 0
for i in range(N_spins - 1):
    # σᵢˣσᵢ₊₁ˣ + σᵢʸσᵢ₊₁ʸ + σᵢᶻσᵢ₊₁ᶻ
    H += J * (
        tensor_at([sigmax()], i, N_spins) * tensor_at([sigmax()], i+1, N_spins) +
        tensor_at([sigmay()], i, N_spins) * tensor_at([sigmay()], i+1, N_spins) +
        tensor_at([sigmaz()], i, N_spins) * tensor_at([sigmaz()], i+1, N_spins)
    )
```

## 有用的工具函數

```python
# 生成隨機量子物件
rand_ket(N)  # 隨機 ket
rand_dm(N)  # 隨機密度矩陣
rand_herm(N)  # 隨機厄米運算子
rand_unitary(N)  # 隨機么正

# 對易子和反對易子
commutator(A, B)  # [A, B]
anti_commutator(A, B)  # {A, B}

# 矩陣指數
(-1j * H * t).expm()  # e^(-iHt)

# 本徵值和本徵向量
H.eigenstates()  # 返回 (本徵值, 本徵向量)
H.eigenenergies()  # 只返回本徵值
H.groundstate()  # 基態能量和態
```
