# 電路轉換

本指南涵蓋使用 Cirq 轉換框架進行電路優化、編譯和操作。

## 轉換器框架

### 基本轉換器

```python
import cirq

# 範例電路
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2])
)

# 應用內建轉換器
from cirq.transformers import optimize_for_target_gateset

# 優化為特定閘集
optimized = optimize_for_target_gateset(
    circuit,
    gateset=cirq.SqrtIswapTargetGateset()
)
```

### 合併單量子位元閘

```python
from cirq.transformers import merge_single_qubit_gates_to_phxz

# 具有多個單量子位元閘的電路
circuit = cirq.Circuit(
    cirq.H(q),
    cirq.T(q),
    cirq.S(q),
    cirq.H(q)
)

# 合併為單一操作
merged = merge_single_qubit_gates_to_phxz(circuit)
```

### 移除可忽略操作

```python
from cirq.transformers import drop_negligible_operations

# 移除低於閾值的閘
circuit_with_small_rotations = cirq.Circuit(
    cirq.rz(1e-10)(q),  # 非常小的旋轉
    cirq.H(q)
)

cleaned = drop_negligible_operations(circuit_with_small_rotations, atol=1e-8)
```

## 自訂轉換器

### 轉換器裝飾器

```python
from cirq.transformers import transformer_api

@transformer_api.transformer
def remove_z_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """從電路中移除所有 Z 閘。"""
    new_moments = []
    for moment in circuit:
        new_ops = [op for op in moment if not isinstance(op.gate, cirq.ZPowGate)]
        if new_ops:
            new_moments.append(cirq.Moment(new_ops))
    return cirq.Circuit(new_moments)

# 使用自訂轉換器
transformed = remove_z_gates(circuit)
```

### 轉換器類別

```python
from cirq.transformers import transformer_primitives

class HToRyTransformer(transformer_primitives.Transformer):
    """將 H 閘替換為 Ry(π/2)。"""

    def __call__(self, circuit: cirq.Circuit, *, context=None) -> cirq.Circuit:
        def map_op(op: cirq.Operation, _) -> cirq.OP_TREE:
            if isinstance(op.gate, cirq.HPowGate):
                return cirq.ry(np.pi/2)(op.qubits[0])
            return op

        return transformer_primitives.map_operations(
            circuit,
            map_op,
            deep=True
        ).unfreeze(copy=False)

# 應用轉換器
transformer = HToRyTransformer()
result = transformer(circuit)
```

## 閘分解

### 分解為目標閘集

```python
from cirq.transformers import optimize_for_target_gateset

# 分解為 CZ + 單量子位元旋轉
target_gateset = cirq.CZTargetGateset()
decomposed = optimize_for_target_gateset(circuit, gateset=target_gateset)

# 分解為 √iSWAP 閘
sqrt_iswap_gateset = cirq.SqrtIswapTargetGateset()
decomposed = optimize_for_target_gateset(circuit, gateset=sqrt_iswap_gateset)
```

### 自訂閘分解

```python
class Toffoli(cirq.Gate):
    def _num_qubits_(self):
        return 3

    def _decompose_(self, qubits):
        """將 Toffoli 分解為基本閘。"""
        c1, c2, t = qubits
        return [
            cirq.H(t),
            cirq.CNOT(c2, t),
            cirq.T(t)**-1,
            cirq.CNOT(c1, t),
            cirq.T(t),
            cirq.CNOT(c2, t),
            cirq.T(t)**-1,
            cirq.CNOT(c1, t),
            cirq.T(c2),
            cirq.T(t),
            cirq.H(t),
            cirq.CNOT(c1, c2),
            cirq.T(c1),
            cirq.T(c2)**-1,
            cirq.CNOT(c1, c2)
        ]

# 使用分解
circuit = cirq.Circuit(Toffoli()(q0, q1, q2))
decomposed = cirq.decompose(circuit)
```

## 電路優化

### 彈出 Z 閘

```python
from cirq.transformers import eject_z

# 將 Z 閘移到電路末端
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.Z(q0),
    cirq.CNOT(q0, q1)
)

ejected = eject_z(circuit)
```

### 彈出相位閘

```python
from cirq.transformers import eject_phased_paulis

# 整合相位閘
optimized = eject_phased_paulis(circuit, atol=1e-8)
```

### 移除空 Moments

```python
from cirq.transformers import drop_empty_moments

# 移除沒有操作的 moments
cleaned = drop_empty_moments(circuit)
```

### 對齊測量

```python
from cirq.transformers import dephase_measurements

# 將測量移到末端並移除之後的操作
aligned = dephase_measurements(circuit)
```

## 電路編譯

### 為硬體編譯

```python
import cirq_google

# 取得裝置規格
device = cirq_google.Sycamore

# 將電路編譯到裝置
from cirq.transformers import optimize_for_target_gateset

compiled = optimize_for_target_gateset(
    circuit,
    gateset=cirq_google.SycamoreTargetGateset()
)

# 驗證編譯後的電路
device.validate_circuit(compiled)
```

### 雙量子位元閘編譯

```python
# 編譯為特定雙量子位元閘
from cirq import two_qubit_to_cz

# 將所有雙量子位元閘轉換為 CZ
cz_circuit = cirq.Circuit()
for moment in circuit:
    for op in moment:
        if len(op.qubits) == 2:
            cz_circuit.append(two_qubit_to_cz(op))
        else:
            cz_circuit.append(op)
```

## 量子位元路由

### 將電路路由到裝置拓撲

```python
from cirq.transformers import route_circuit

# 定義裝置連接性
device_graph = cirq.NamedTopology(
    {
        (0, 0): [(0, 1), (1, 0)],
        (0, 1): [(0, 0), (1, 1)],
        (1, 0): [(0, 0), (1, 1)],
        (1, 1): [(0, 1), (1, 0)]
    }
)

# 將邏輯量子位元路由到物理量子位元
routed_circuit = route_circuit(
    circuit,
    device_graph=device_graph,
    routing_algo=cirq.RouteCQC(device_graph)
)
```

### SWAP 網路插入

```python
# 手動插入 SWAP 進行路由
def insert_swaps(circuit, swap_locations):
    """在指定位置插入 SWAP 閘。"""
    new_circuit = cirq.Circuit()
    moment_idx = 0

    for i, moment in enumerate(circuit):
        if i in swap_locations:
            q0, q1 = swap_locations[i]
            new_circuit.append(cirq.SWAP(q0, q1))
        new_circuit.append(moment)

    return new_circuit
```

## 進階轉換

### 么正編譯

```python
import scipy.linalg

# 將任意么正編譯為閘序列
def compile_unitary(unitary, qubits):
    """使用 KAK 分解編譯 2x2 么正。"""
    from cirq.linalg import kak_decomposition

    decomp = kak_decomposition(unitary)
    operations = []

    # 加入前置單量子位元閘
    operations.append(cirq.MatrixGate(decomp.single_qubit_operations_before[0])(qubits[0]))
    operations.append(cirq.MatrixGate(decomp.single_qubit_operations_before[1])(qubits[1]))

    # 加入交互作用（雙量子位元）部分
    x, y, z = decomp.interaction_coefficients
    operations.append(cirq.XXPowGate(exponent=x/np.pi)(qubits[0], qubits[1]))
    operations.append(cirq.YYPowGate(exponent=y/np.pi)(qubits[0], qubits[1]))
    operations.append(cirq.ZZPowGate(exponent=z/np.pi)(qubits[0], qubits[1]))

    # 加入後置單量子位元閘
    operations.append(cirq.MatrixGate(decomp.single_qubit_operations_after[0])(qubits[0]))
    operations.append(cirq.MatrixGate(decomp.single_qubit_operations_after[1])(qubits[1]))

    return operations
```

### 電路簡化

```python
from cirq.transformers import (
    merge_k_qubit_unitaries,
    merge_single_qubit_gates_to_phxz
)

# 合併相鄰單量子位元閘
simplified = merge_single_qubit_gates_to_phxz(circuit)

# 合併相鄰 k 量子位元么正
simplified = merge_k_qubit_unitaries(circuit, k=2)
```

### 基於交換性的優化

```python
# 將 Z 閘通過 CNOT 交換
def commute_z_through_cnot(circuit):
    """將 Z 閘通過 CNOT 閘移動。"""
    new_moments = []

    for moment in circuit:
        ops = list(moment)
        # 尋找 CNOT 之前的 Z 閘
        z_ops = [op for op in ops if isinstance(op.gate, cirq.ZPowGate)]
        cnot_ops = [op for op in ops if isinstance(op.gate, cirq.CXPowGate)]

        # 應用交換規則
        # 控制端上的 Z 可交換，目標端上的 Z 反交換
        # （這裡是簡化的邏輯）

        new_moments.append(cirq.Moment(ops))

    return cirq.Circuit(new_moments)
```

## 轉換管線

### 組合多個轉換器

```python
from cirq.transformers import transformer_api

# 建構轉換管線
@transformer_api.transformer
def optimization_pipeline(circuit: cirq.Circuit) -> cirq.Circuit:
    # 步驟 1：合併單量子位元閘
    circuit = merge_single_qubit_gates_to_phxz(circuit)

    # 步驟 2：移除可忽略操作
    circuit = drop_negligible_operations(circuit)

    # 步驟 3：彈出 Z 閘
    circuit = eject_z(circuit)

    # 步驟 4：移除空 moments
    circuit = drop_empty_moments(circuit)

    return circuit

# 應用管線
optimized = optimization_pipeline(circuit)
```

## 驗證和分析

### 電路深度減少

```python
# 測量轉換前後的電路深度
print(f"原始深度：{len(circuit)}")
optimized = optimization_pipeline(circuit)
print(f"優化後深度：{len(optimized)}")
```

### 閘計數分析

```python
def count_gates(circuit):
    """依類型計數閘。"""
    counts = {}
    for moment in circuit:
        for op in moment:
            gate_type = type(op.gate).__name__
            counts[gate_type] = counts.get(gate_type, 0) + 1
    return counts

original_counts = count_gates(circuit)
optimized_counts = count_gates(optimized)
print(f"原始：{original_counts}")
print(f"優化後：{optimized_counts}")
```

## 最佳實務

1. **從高階轉換器開始**：先使用內建轉換器再寫自訂的
2. **串連轉換器**：按順序應用多個優化
3. **轉換後驗證**：確保電路正確性和裝置相容性
4. **測量改進**：追蹤深度和閘計數減少
5. **使用適當的閘集**：匹配目標硬體能力
6. **考慮交換性**：利用閘交換進行優化
7. **先在小電路測試**：擴展前驗證轉換器是否正確工作
