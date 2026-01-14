# 量子電路建構

## 建立量子電路

使用 `QuantumCircuit` 類別建立電路：

```python
from qiskit import QuantumCircuit

# 建立具有 3 個量子位元的電路
qc = QuantumCircuit(3)

# 建立具有 3 個量子位元和 3 個古典位元的電路
qc = QuantumCircuit(3, 3)
```

## 單量子位元閘

### Pauli 閘

```python
qc.x(0)   # 量子位元 0 上的 NOT/Pauli-X 閘
qc.y(1)   # 量子位元 1 上的 Pauli-Y 閘
qc.z(2)   # 量子位元 2 上的 Pauli-Z 閘
```

### Hadamard 閘

建立疊加態：

```python
qc.h(0)   # 量子位元 0 上的 Hadamard 閘
```

### 相位閘

```python
qc.s(0)   # S 閘（√Z）
qc.t(0)   # T 閘（√S）
qc.p(π/4, 0)   # 自訂角度的相位閘
```

### 旋轉閘

```python
from math import pi

qc.rx(pi/2, 0)   # 繞 X 軸旋轉
qc.ry(pi/4, 1)   # 繞 Y 軸旋轉
qc.rz(pi/3, 2)   # 繞 Z 軸旋轉
```

## 多量子位元閘

### CNOT（受控 NOT）

```python
qc.cx(0, 1)   # CNOT，控制=0，目標=1
```

### 受控閘

```python
qc.cy(0, 1)   # 受控 Y
qc.cz(0, 1)   # 受控 Z
qc.ch(0, 1)   # 受控 Hadamard
```

### SWAP 閘

```python
qc.swap(0, 1)   # 交換量子位元 0 和 1
```

### Toffoli（CCX）閘

```python
qc.ccx(0, 1, 2)   # Toffoli，控制=0,1，目標=2
```

## 測量

添加測量以讀取量子位元狀態：

```python
# 測量所有量子位元
qc.measure_all()

# 將特定量子位元測量到特定古典位元
qc.measure(0, 0)   # 將量子位元 0 測量到古典位元 0
qc.measure([0, 1], [0, 1])   # 將量子位元 0,1 測量到位元 0,1
```

## 電路組合

### 合併電路

```python
qc1 = QuantumCircuit(2)
qc1.h(0)

qc2 = QuantumCircuit(2)
qc2.cx(0, 1)

# 組合電路
qc_combined = qc1.compose(qc2)
```

### 張量積

```python
qc1 = QuantumCircuit(1)
qc1.h(0)

qc2 = QuantumCircuit(1)
qc2.x(0)

# 從較小電路建立較大電路
qc_tensor = qc1.tensor(qc2)   # 產生 2 量子位元電路
```

## 屏障和標籤

```python
qc.barrier()   # 在電路中添加視覺屏障
qc.barrier([0, 1])   # 在特定量子位元上添加屏障

# 添加標籤以增加清晰度
qc.barrier(label="初始化")
```

## 電路屬性

```python
print(qc.num_qubits)   # 量子位元數量
print(qc.num_clbits)   # 古典位元數量
print(qc.depth())      # 電路深度
print(qc.size())       # 總閘數
print(qc.count_ops())  # 閘計數字典
```

## 範例：Bell 態

在兩個量子位元之間建立糾纏：

```python
qc = QuantumCircuit(2)
qc.h(0)           # 量子位元 0 上的疊加
qc.cx(0, 1)       # 糾纏量子位元 0 和 1
qc.measure_all()  # 測量兩個量子位元
```

## 範例：量子傅立葉變換（QFT）

```python
from math import pi

def qft(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            qc.cp(pi/2**(k-j), k, j)
    return qc

# 建立 3 量子位元 QFT
qc_qft = qft(3)
```

## 參數化電路

為變分演算法建立帶參數的電路：

```python
from qiskit.circuit import Parameter

theta = Parameter('θ')
qc = QuantumCircuit(1)
qc.ry(theta, 0)

# 綁定參數值
qc_bound = qc.assign_parameters({theta: pi/4})
```

## 電路操作

```python
# 電路的逆
qc_inverse = qc.inverse()

# 將閘分解為基礎閘
qc_decomposed = qc.decompose()

# 繪製電路（返回字串或圖表）
print(qc.draw())
print(qc.draw('mpl'))   # Matplotlib 圖形
```
