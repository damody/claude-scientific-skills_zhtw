# Qiskit 中的視覺化

Qiskit 提供全面的視覺化工具，用於量子電路、測量結果和量子態。

## 安裝

安裝視覺化相依性：

```bash
uv pip install "qiskit[visualization]" matplotlib
```

## 電路視覺化

### 文字繪製

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# 簡單文字輸出
print(qc.draw())

# 更詳細的文字
print(qc.draw('text', fold=-1))  # 不折疊長電路
```

### Matplotlib 繪製

```python
# 高品質 matplotlib 圖形
qc.draw('mpl')

# 儲存到檔案
fig = qc.draw('mpl')
fig.savefig('circuit.png', dpi=300, bbox_inches='tight')
```

### LaTeX 繪製

```python
# 生成 LaTeX 電路圖
qc.draw('latex')

# 儲存 LaTeX 原始碼
latex_source = qc.draw('latex_source')
with open('circuit.tex', 'w') as f:
    f.write(latex_source)
```

## 自訂電路繪製

### 樣式選項

```python
from qiskit.visualization import circuit_drawer

# 反轉量子位元順序
qc.draw('mpl', reverse_bits=True)

# 折疊長電路
qc.draw('mpl', fold=20)  # 在 20 列處折疊

# 顯示閒置線路
qc.draw('mpl', idle_wires=False)

# 添加初始態
qc.draw('mpl', initial_state=True)
```

### 顏色自訂

```python
style = {
    'displaycolor': {
        'h': ('#FA74A6', '#000000'),     # Hadamard：粉紅色
        'cx': ('#A8D0DB', '#000000'),    # CNOT：淺藍色
        'measure': ('#F7E7B4', '#000000') # 測量：黃色
    }
}

qc.draw('mpl', style=style)
```

## 結果視覺化

### 計數直方圖

```python
from qiskit.visualization import plot_histogram
from qiskit.primitives import StatevectorSampler

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
counts = result[0].data.meas.get_counts()

# 繪製直方圖
plot_histogram(counts)

# 比較多個實驗
counts1 = {'00': 500, '11': 524}
counts2 = {'00': 480, '11': 544}
plot_histogram([counts1, counts2], legend=['執行 1', '執行 2'])

# 儲存圖形
fig = plot_histogram(counts)
fig.savefig('histogram.png', dpi=300, bbox_inches='tight')
```

### 直方圖選項

```python
# 自訂顏色
plot_histogram(counts, color=['#1f77b4', '#ff7f0e'])

# 按值排序
plot_histogram(counts, sort='value')

# 設定柱狀標籤
plot_histogram(counts, bar_labels=True)

# 設定目標分佈（用於比較）
target = {'00': 0.5, '11': 0.5}
plot_histogram(counts, target=target)
```

## 態視覺化

### Bloch 球

在 Bloch 球上視覺化單量子位元態：

```python
from qiskit.visualization import plot_bloch_vector
from qiskit.quantum_info import Statevector
import numpy as np

# 視覺化特定態向量
# 態 |+⟩：|0⟩ 和 |1⟩ 的等權疊加
state = Statevector.from_label('+')
plot_bloch_vector(state.to_bloch())

# 自訂向量
plot_bloch_vector([0, 1, 0])  # X 軸上的 |+⟩ 態
```

### 多量子位元 Bloch 球

```python
from qiskit.visualization import plot_bloch_multivector

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

state = Statevector.from_instruction(qc)
plot_bloch_multivector(state)
```

### 態城市圖

以 3D 城市形式視覺化態振幅：

```python
from qiskit.visualization import plot_state_city
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(3)
qc.h(range(3))
state = Statevector.from_instruction(qc)

plot_state_city(state)

# 自訂
plot_state_city(state, color=['#FF6B6B', '#4ECDC4'])
```

### QSphere

在球面上視覺化量子態：

```python
from qiskit.visualization import plot_state_qsphere

state = Statevector.from_instruction(qc)
plot_state_qsphere(state)
```

### Hinton 圖

顯示態振幅：

```python
from qiskit.visualization import plot_state_hinton

state = Statevector.from_instruction(qc)
plot_state_hinton(state)
```

## 密度矩陣視覺化

```python
from qiskit.visualization import plot_state_density
from qiskit.quantum_info import DensityMatrix

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

state = DensityMatrix.from_instruction(qc)
plot_state_density(state)
```

## 閘映射視覺化

視覺化後端耦合圖：

```python
from qiskit.visualization import plot_gate_map
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

# 顯示量子位元連接性
plot_gate_map(backend)

# 顯示錯誤率
plot_gate_map(backend, plot_error_rates=True)
```

## 錯誤圖視覺化

顯示後端錯誤率：

```python
from qiskit.visualization import plot_error_map

plot_error_map(backend)
```

## 電路屬性顯示

```python
from qiskit.visualization import plot_circuit_layout

# 顯示電路如何映射到物理量子位元
transpiled_qc = transpile(qc, backend=backend)
plot_circuit_layout(transpiled_qc, backend)
```

## 脈衝視覺化

用於脈衝級控制：

```python
from qiskit import pulse
from qiskit.visualization import pulse_drawer

# 建立脈衝排程
with pulse.build(backend) as schedule:
    pulse.play(pulse.Gaussian(duration=160, amp=0.1, sigma=40), pulse.drive_channel(0))

# 視覺化
schedule.draw()
```

## 互動式小工具（Jupyter）

### 電路組合器小工具

```python
from qiskit.tools.jupyter import QuantumCircuitComposer

composer = QuantumCircuitComposer()
composer.show()
```

### 互動式態視覺化

```python
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 啟用互動模式
plt.ion()
plot_histogram(counts)
plt.show()
```

## 比較圖

### 多直方圖

```python
# 比較來自不同後端的結果
counts_sim = {'00': 500, '11': 524}
counts_hw = {'00': 480, '01': 20, '10': 24, '11': 500}

plot_histogram(
    [counts_sim, counts_hw],
    legend=['模擬器', '硬體'],
    figsize=(12, 6)
)
```

### 轉譯前後比較

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

# 原始電路
qc.draw('mpl', ax=ax1)
ax1.set_title('原始電路')

# 轉譯後電路
qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
qc_transpiled.draw('mpl', ax=ax2)
ax2.set_title('轉譯後電路')

plt.tight_layout()
plt.show()
```

## 儲存視覺化

### 儲存為各種格式

```python
# PNG
fig = qc.draw('mpl')
fig.savefig('circuit.png', dpi=300, bbox_inches='tight')

# PDF
fig.savefig('circuit.pdf', bbox_inches='tight')

# SVG（向量圖形）
fig.savefig('circuit.svg', bbox_inches='tight')

# 直方圖
hist_fig = plot_histogram(counts)
hist_fig.savefig('results.png', dpi=300, bbox_inches='tight')
```

## 樣式最佳實踐

### 出版品質圖形

```python
import matplotlib.pyplot as plt

# 設定 matplotlib 樣式
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# 建立高品質視覺化
fig = qc.draw('mpl', style='iqp')
fig.savefig('publication_circuit.png', dpi=600, bbox_inches='tight')
```

### 可用樣式

```python
# 預設樣式
qc.draw('mpl')

# IQP 樣式（IBM Quantum）
qc.draw('mpl', style='iqp')

# 色盲友善
qc.draw('mpl', style='bw')  # 黑白
```

## 視覺化疑難排解

### 常見問題

**問題**："No module named 'matplotlib'"
```bash
uv pip install matplotlib
```

**問題**：電路太大無法顯示
```python
# 使用折疊
qc.draw('mpl', fold=50)

# 或匯出到檔案而不是顯示
fig = qc.draw('mpl')
fig.savefig('large_circuit.png', dpi=150, bbox_inches='tight')
```

**問題**：Jupyter notebook 不顯示圖形
```python
# 在 notebook 開頭添加魔術命令
%matplotlib inline
```

**問題**：LaTeX 視覺化不起作用
```bash
# 安裝 LaTeX 支援
uv pip install pylatexenc
```
