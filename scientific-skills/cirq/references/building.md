# 建構量子電路

本指南涵蓋 Cirq 中的電路建構，包括量子位元、閘、操作和電路模式。

## 基本電路建構

### 建立電路

```python
import cirq

# 建立電路
circuit = cirq.Circuit()

# 建立量子位元
q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)
q2 = cirq.LineQubit(0)

# 將閘加入電路
circuit.append([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
])
```

### 量子位元類型

**GridQubit**：適用於硬體拓撲的 2D 網格
```python
qubits = cirq.GridQubit.square(2)  # 2x2 網格
qubit = cirq.GridQubit(row=0, col=1)
```

**LineQubit**：1D 線性拓撲
```python
qubits = cirq.LineQubit.range(5)  # 一列 5 個量子位元
qubit = cirq.LineQubit(3)
```

**NamedQubit**：自訂命名的量子位元
```python
qubit = cirq.NamedQubit('my_qubit')
```

## 常見閘和操作

### 單量子位元閘

```python
# Pauli 閘
cirq.X(qubit)  # NOT 閘
cirq.Y(qubit)
cirq.Z(qubit)

# Hadamard
cirq.H(qubit)

# 旋轉閘
cirq.rx(angle)(qubit)  # 繞 X 軸旋轉
cirq.ry(angle)(qubit)  # 繞 Y 軸旋轉
cirq.rz(angle)(qubit)  # 繞 Z 軸旋轉

# 相位閘
cirq.S(qubit)  # √Z 閘
cirq.T(qubit)  # ⁴√Z 閘
```

### 雙量子位元閘

```python
# CNOT（受控 NOT）
cirq.CNOT(control, target)
cirq.CX(control, target)  # 別名

# CZ（受控 Z）
cirq.CZ(q0, q1)

# SWAP
cirq.SWAP(q0, q1)

# iSWAP
cirq.ISWAP(q0, q1)

# 受控旋轉
cirq.CZPowGate(exponent=0.5)(q0, q1)
```

### 測量操作

```python
# 測量單一量子位元
cirq.measure(qubit, key='m')

# 測量多個量子位元
cirq.measure(q0, q1, q2, key='result')

# 測量電路中的所有量子位元
circuit.append(cirq.measure(*qubits, key='final'))
```

## 進階電路建構

### 參數化閘

```python
import sympy

# 建立符號參數
theta = sympy.Symbol('theta')
phi = sympy.Symbol('phi')

# 在閘中使用
circuit = cirq.Circuit(
    cirq.rx(theta)(q0),
    cirq.ry(phi)(q1),
    cirq.CNOT(q0, q1)
)

# 稍後解析參數
resolved = cirq.resolve_parameters(circuit, {'theta': 0.5, 'phi': 1.2})
```

### 透過么正矩陣自訂閘

```python
import numpy as np

# 定義么正矩陣
unitary = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
]) / np.sqrt(2)

# 從么正矩陣建立閘
gate = cirq.MatrixGate(unitary)
operation = gate(q0, q1)
```

### 閘分解

```python
# 定義帶分解的自訂閘
class MyGate(cirq.Gate):
    def _num_qubits_(self):
        return 1

    def _decompose_(self, qubits):
        q = qubits[0]
        return [cirq.H(q), cirq.T(q), cirq.H(q)]

    def _circuit_diagram_info_(self, args):
        return 'MyGate'

# 使用自訂閘
my_gate = MyGate()
circuit.append(my_gate(q0))
```

## 電路組織

### Moments

電路被組織成 moments（平行操作）：

```python
# 明確的 moment 建構
circuit = cirq.Circuit(
    cirq.Moment([cirq.H(q0), cirq.H(q1)]),
    cirq.Moment([cirq.CNOT(q0, q1)]),
    cirq.Moment([cirq.measure(q0, key='m0'), cirq.measure(q1, key='m1')])
)

# 存取 moments
for i, moment in enumerate(circuit):
    print(f"Moment {i}: {moment}")
```

### 電路操作

```python
# 連接電路
circuit3 = circuit1 + circuit2

# 插入操作
circuit.insert(index, operation)

# 使用策略附加
circuit.append(operations, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
```

## 電路模式

### Bell 態製備

```python
def bell_state_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    )
```

### GHZ 態

```python
def ghz_circuit(qubits):
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    return circuit
```

### 量子傅立葉轉換

```python
def qft_circuit(qubits):
    circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        circuit.append(cirq.H(q))
        for j in range(i + 1, len(qubits)):
            circuit.append(cirq.CZPowGate(exponent=1/2**(j-i))(qubits[j], q))

    # 反轉量子位元順序
    for i in range(len(qubits) // 2):
        circuit.append(cirq.SWAP(qubits[i], qubits[len(qubits) - i - 1]))

    return circuit
```

## 電路匯入/匯出

### OpenQASM

```python
# 匯出為 QASM
qasm_str = circuit.to_qasm()

# 從 QASM 匯入
from cirq.contrib.qasm_import import circuit_from_qasm
circuit = circuit_from_qasm(qasm_str)
```

### 電路 JSON

```python
import json

# 序列化
json_str = cirq.to_json(circuit)

# 反序列化
circuit = cirq.read_json(json_text=json_str)
```

## 使用 Qudits

Qudits 是高維量子系統（qutrits、ququarts 等）：

```python
# 建立 qutrit（3 能級系統）
qutrit = cirq.LineQid(0, dimension=3)

# 自訂 qutrit 閘
class QutritXGate(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

gate = QutritXGate()
circuit = cirq.Circuit(gate(qutrit))
```

## 可觀測量

從 Pauli 運算子建立可觀測量：

```python
# 單一 Pauli 可觀測量
obs = cirq.Z(q0)

# Pauli 字串
obs = cirq.X(q0) * cirq.Y(q1) * cirq.Z(q2)

# 線性組合
from cirq import PauliSum
obs = 0.5 * cirq.X(q0) + 0.3 * cirq.Z(q1)
```

## 最佳實務

1. **使用適當的量子位元類型**：GridQubit 用於硬體拓撲，LineQubit 用於 1D 問題
2. **保持電路模組化**：建構可重用的電路函數
3. **使用符號參數**：用於參數掃描和優化
4. **清楚標記測量**：使用描述性鍵標記測量結果
5. **記錄自訂閘**：包含電路圖資訊以便視覺化
