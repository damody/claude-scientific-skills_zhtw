# QuTiP 視覺化

## Bloch 球

在 Bloch 球上視覺化量子位元態。

### 基本用法

```python
from qutip import *
import matplotlib.pyplot as plt

# 建立 Bloch 球
b = Bloch()

# 添加態
psi = (basis(2, 0) + basis(2, 1)).unit()
b.add_states(psi)

# 添加向量
b.add_vectors([1, 0, 0])  # X 軸

# 顯示
b.show()
```

### 多個態

```python
# 添加多個態
states = [(basis(2, 0) + basis(2, 1)).unit(),
          (basis(2, 0) + 1j*basis(2, 1)).unit()]
b.add_states(states)

# 添加點
b.add_points([[0, 1, 0], [0, -1, 0]])

# 自訂顏色
b.point_color = ['r', 'g']
b.point_marker = ['o', 's']
b.point_size = [20, 20]

b.show()
```

### 動畫

```python
# 動畫態演化
states = result.states  # 來自 sesolve/mesolve

b = Bloch()
b.vector_color = ['r']
b.view = [-40, 30]  # 檢視角度

# 建立動畫
from matplotlib.animation import FuncAnimation

def animate(i):
    b.clear()
    b.add_states(states[i])
    b.make_sphere()
    return b.axes

anim = FuncAnimation(b.fig, animate, frames=len(states),
                      interval=50, blit=False, repeat=True)
plt.show()
```

### 自訂

```python
b = Bloch()

# 球面外觀
b.sphere_color = '#FFDDDD'
b.sphere_alpha = 0.1
b.frame_alpha = 0.1

# 軸
b.xlabel = ['$|+\\\\rangle$', '$|-\\\\rangle$']
b.ylabel = ['$|+i\\\\rangle$', '$|-i\\\\rangle$']
b.zlabel = ['$|0\\\\rangle$', '$|1\\\\rangle$']

# 字體大小
b.font_size = 20
b.font_color = 'black'

# 檢視角度
b.view = [-60, 30]

# 儲存圖形
b.save('bloch.png')
```

## Wigner 函數

相空間準機率分佈。

### 基本計算

```python
# 建立態
psi = coherent(N, alpha)

# 計算 Wigner 函數
xvec = np.linspace(-5, 5, 200)
W = wigner(psi, xvec, xvec)

# 繪圖
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cont = ax.contourf(xvec, xvec, W, 100, cmap='RdBu')
ax.set_xlabel('Re(α)')
ax.set_ylabel('Im(α)')
plt.colorbar(cont, ax=ax)
plt.show()
```

### 特殊色彩映射

```python
# Wigner 色彩映射強調負值
from qutip import wigner_cmap

W = wigner(psi, xvec, xvec)

fig, ax = plt.subplots()
cont = ax.contourf(xvec, xvec, W, 100, cmap=wigner_cmap(W))
ax.set_title('Wigner 函數')
plt.colorbar(cont)
plt.show()
```

### 3D 表面圖

```python
from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(xvec, xvec)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, W, cmap='RdBu', alpha=0.8)
ax.set_xlabel('Re(α)')
ax.set_ylabel('Im(α)')
ax.set_zlabel('W(α)')
plt.show()
```

### 比較態

```python
# 比較不同態
states = [coherent(N, 2), fock(N, 2), thermal_dm(N, 2)]
titles = ['相干態', 'Fock 態', '熱態']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (state, title) in enumerate(zip(states, titles)):
    W = wigner(state, xvec, xvec)
    cont = axes[i].contourf(xvec, xvec, W, 100, cmap='RdBu')
    axes[i].set_title(title)
    axes[i].set_xlabel('Re(α)')
    if i == 0:
        axes[i].set_ylabel('Im(α)')

plt.tight_layout()
plt.show()
```

## Q 函數（Husimi）

平滑的相空間分佈（始終為正）。

### 基本用法

```python
from qutip import qfunc

Q = qfunc(psi, xvec, xvec)

fig, ax = plt.subplots()
cont = ax.contourf(xvec, xvec, Q, 100, cmap='viridis')
ax.set_xlabel('Re(α)')
ax.set_ylabel('Im(α)')
ax.set_title('Q 函數')
plt.colorbar(cont)
plt.show()
```

### 高效批次計算

```python
from qutip import QFunc

# 用於在多點計算 Q 函數
qf = QFunc(rho)
Q = qf.eval(xvec, xvec)
```

## Fock 態機率分佈

視覺化光子數分佈。

### 基本直方圖

```python
from qutip import plot_fock_distribution

# 單一態
psi = coherent(N, 2)
fig, ax = plot_fock_distribution(psi)
ax.set_title('相干態')
plt.show()
```

### 比較分佈

```python
states = {
    '相干態': coherent(20, 2),
    '熱態': thermal_dm(20, 2),
    'Fock 態': fock(20, 2)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (title, state) in zip(axes, states.items()):
    plot_fock_distribution(state, fig=fig, ax=ax)
    ax.set_title(title)
    ax.set_ylim([0, 0.3])

plt.tight_layout()
plt.show()
```

### 時間演化

```python
# 顯示光子分佈演化
result = mesolve(H, psi0, tlist, c_ops)

# 在不同時間繪圖
times_to_plot = [0, 5, 10, 15]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, t_idx in zip(axes, times_to_plot):
    plot_fock_distribution(result.states[t_idx], fig=fig, ax=ax)
    ax.set_title(f't = {tlist[t_idx]:.1f}')
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()
```

## 矩陣視覺化

### Hinton 圖

用加權方塊視覺化矩陣結構。

```python
from qutip import hinton

# 密度矩陣
rho = bell_state('00').proj()

hinton(rho)
plt.title('Bell 態密度矩陣')
plt.show()
```

### 矩陣直方圖

矩陣元素的 3D 柱狀圖。

```python
from qutip import matrix_histogram

# 顯示實部和虛部
H = sigmaz()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

matrix_histogram(H.full(), xlabels=['0', '1'], ylabels=['0', '1'],
                 fig=fig, ax=axes[0])
axes[0].set_title('實部')

matrix_histogram(H.full(), bar_type='imag', xlabels=['0', '1'],
                 ylabels=['0', '1'], fig=fig, ax=axes[1])
axes[1].set_title('虛部')

plt.tight_layout()
plt.show()
```

### 複數相位圖

```python
# 視覺化複數矩陣元素
rho = coherent_dm(10, 2)

# 繪製複數元素
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 絕對值
matrix_histogram(rho.full(), bar_type='abs', fig=fig, ax=axes[0])
axes[0].set_title('絕對值')

# 相位
matrix_histogram(rho.full(), bar_type='phase', fig=fig, ax=axes[1])
axes[1].set_title('相位')

plt.tight_layout()
plt.show()
```

## 能階圖

```python
# 視覺化能量本徵值
H = num(N) + 0.1 * (create(N) + destroy(N))**2

# 取得本徵值和本徵向量
evals, ekets = H.eigenstates()

# 繪製能階
fig, ax = plt.subplots(figsize=(8, 6))

for i, E in enumerate(evals[:10]):
    ax.hlines(E, 0, 1, linewidth=2)
    ax.text(1.1, E, f'|{i}⟩', va='center')

ax.set_ylabel('能量')
ax.set_xlim([-0.2, 1.5])
ax.set_xticks([])
ax.set_title('能譜')
plt.show()
```

## 量子過程層析

視覺化量子通道/閘作用。

```python
from qutip.qip.operations import cnot
from qutip_qip.tomography import qpt, qpt_plot_combined

# 定義過程（例如 CNOT 閘）
U = cnot()

# 執行 QPT
chi = qpt(U, method='choicm')

# 視覺化
fig = qpt_plot_combined(chi)
plt.show()
```

## 時間相依期望值

```python
# 期望值的標準繪圖
result = mesolve(H, psi0, tlist, c_ops, e_ops=[num(N)])

fig, ax = plt.subplots()
ax.plot(tlist, result.expect[0])
ax.set_xlabel('時間')
ax.set_ylabel('⟨n⟩')
ax.set_title('光子數演化')
ax.grid(True)
plt.show()
```

### 多個可觀測量

```python
# 繪製多個期望值
e_ops = [a.dag() * a, a + a.dag(), 1j * (a - a.dag())]
labels = ['⟨n⟩', '⟨X⟩', '⟨P⟩']

result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)

fig, axes = plt.subplots(3, 1, figsize=(8, 9))

for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(tlist, result.expect[i])
    ax.set_ylabel(label)
    ax.grid(True)

axes[-1].set_xlabel('時間')
plt.tight_layout()
plt.show()
```

## 關聯函數和頻譜

```python
# 雙時關聯函數
taulist = np.linspace(0, 10, 200)
corr = correlation_2op_1t(H, rho0, taulist, c_ops, a.dag(), a)

# 繪製關聯
fig, ax = plt.subplots()
ax.plot(taulist, np.real(corr))
ax.set_xlabel('τ')
ax.set_ylabel('⟨a†(τ)a(0)⟩')
ax.set_title('關聯函數')
plt.show()

# 功率譜
from qutip import spectrum_correlation_fft

w, S = spectrum_correlation_fft(taulist, corr)

fig, ax = plt.subplots()
ax.plot(w, S)
ax.set_xlabel('頻率')
ax.set_ylabel('S(ω)')
ax.set_title('功率譜')
plt.show()
```

## 儲存圖形

```python
# 高解析度儲存
fig.savefig('my_plot.png', dpi=300, bbox_inches='tight')
fig.savefig('my_plot.pdf', bbox_inches='tight')
fig.savefig('my_plot.svg', bbox_inches='tight')
```
