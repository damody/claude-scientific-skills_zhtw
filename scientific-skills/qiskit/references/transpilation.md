# 電路轉譯和最佳化

轉譯是將量子電路重寫以匹配特定量子裝置的拓撲和閘集，同時針對在雜訊量子電腦上執行進行最佳化的過程。

## 為何需要轉譯？

**問題**：抽象量子電路可能使用硬體上不可用的閘，並假設全對全的量子位元連接性。

**解決方案**：轉譯將電路轉換為：
1. 僅使用硬體原生閘（基礎閘）
2. 遵守物理量子位元連接性
3. 最小化電路深度和閘數
4. 針對雜訊裝置最佳化以減少錯誤

## 基本轉譯

### 簡單轉譯

```python
from qiskit import QuantumCircuit, transpile

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# 為特定後端轉譯
transpiled_qc = transpile(qc, backend=backend)
```

### 最佳化等級

選擇最佳化等級 0-3：

```python
# 等級 0：無最佳化（最快）
qc_0 = transpile(qc, backend=backend, optimization_level=0)

# 等級 1：輕度最佳化
qc_1 = transpile(qc, backend=backend, optimization_level=1)

# 等級 2：中度最佳化（預設）
qc_2 = transpile(qc, backend=backend, optimization_level=2)

# 等級 3：重度最佳化（最慢，最佳結果）
qc_3 = transpile(qc, backend=backend, optimization_level=3)
```

**Qiskit SDK v2.2** 提供比競爭對手**快 83 倍的轉譯速度**。

## 轉譯階段

轉譯器管線由六個階段組成：

### 1. Init（初始化）階段
- 驗證電路指令
- 將多量子位元閘轉換為標準形式

### 2. Layout（佈局）階段
- 將虛擬量子位元映射到物理量子位元
- 考慮量子位元連接性和錯誤率

```python
from qiskit.transpiler import CouplingMap

# 定義自訂耦合
coupling = CouplingMap([(0, 1), (1, 2), (2, 3)])
qc_transpiled = transpile(qc, coupling_map=coupling)
```

### 3. Routing（路由）階段
- 插入 SWAP 閘以滿足連接性約束
- 最小化額外的 SWAP 開銷

### 4. Translation（轉換）階段
- 將閘轉換為硬體基礎閘
- 典型基礎：{RZ, SX, X, CX}

```python
# 指定基礎閘
basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
qc_transpiled = transpile(qc, basis_gates=basis_gates)
```

### 5. Optimization（最佳化）階段
- 減少閘數和電路深度
- 應用閘消除和交換規則
- 使用**虛擬排列消除**（等級 2-3）
- 找到可分離的操作進行分解

### 6. Scheduling（排程）階段
- 為脈衝級控制添加時序資訊

## 進階最佳化功能

### 虛擬排列消除

在最佳化等級 2-3，Qiskit 分析交換結構，透過追蹤虛擬量子位元排列來消除不必要的 SWAP 閘。

### 閘消除

識別並移除相互抵消的閘對：
- X-X → I
- H-H → I
- CNOT-CNOT → I

### 數值分解

拆分可表示為可分離單量子位元操作的雙量子位元閘。

## 常用轉譯參數

### 初始佈局

指定要使用的物理量子位元：

```python
# 使用特定物理量子位元
initial_layout = [0, 2, 4]  # 將電路量子位元 0,1,2 映射到物理量子位元 0,2,4
qc_transpiled = transpile(qc, backend=backend, initial_layout=initial_layout)
```

### 近似度

用較少的閘換取精確度（0.0 = 最大近似，1.0 = 無近似）：

```python
# 允許 5% 近似誤差以減少閘數
qc_transpiled = transpile(qc, backend=backend, approximation_degree=0.95)
```

### 可重現性種子

```python
qc_transpiled = transpile(qc, backend=backend, seed_transpiler=42)
```

### 排程方法

```python
# 添加時序約束
qc_transpiled = transpile(
    qc,
    backend=backend,
    scheduling_method='alap'  # 盡可能晚執行
)
```

## 為模擬器轉譯

即使對於模擬器，轉譯也可以最佳化電路：

```python
from qiskit_aer import AerSimulator

simulator = AerSimulator()
qc_optimized = transpile(qc, backend=simulator, optimization_level=3)

# 比較閘數
print(f"原始: {qc.size()} 閘")
print(f"最佳化: {qc_optimized.size()} 閘")
```

## 目標感知轉譯

使用 `Target` 物件進行詳細的後端規格：

```python
from qiskit.transpiler import Target

# 使用目標規格轉譯
qc_transpiled = transpile(qc, target=backend.target)
```

## 轉譯後的電路分析

```python
qc_transpiled = transpile(qc, backend=backend, optimization_level=3)

# 分析結果
print(f"深度: {qc_transpiled.depth()}")
print(f"閘數: {qc_transpiled.size()}")
print(f"操作: {qc_transpiled.count_ops()}")

# 檢查雙量子位元閘數（主要錯誤來源）
two_qubit_gates = qc_transpiled.count_ops().get('cx', 0)
print(f"雙量子位元閘: {two_qubit_gates}")
```

**Qiskit 產生的電路比領先替代方案減少 29% 的雙量子位元閘**，顯著降低錯誤。

## 多電路轉譯

高效轉譯多個電路：

```python
circuits = [qc1, qc2, qc3]
transpiled_circuits = transpile(
    circuits,
    backend=backend,
    optimization_level=3
)
```

## 預轉譯最佳實踐

### 1. 設計時考慮硬體拓撲

設計電路時考慮後端耦合圖：

```python
# 檢查後端耦合
print(backend.coupling_map)

# 設計與耦合對齊的電路
```

### 2. 盡可能使用原生閘

某些後端支援 {CX, RZ, SX, X} 以外的閘：

```python
# 檢查可用的基礎閘
print(backend.configuration().basis_gates)
```

### 3. 最小化雙量子位元閘

雙量子位元閘的錯誤率顯著較高：
- 設計演算法以最小化 CNOT 閘
- 使用閘恆等式減少數量

### 4. 先用模擬器測試

```python
from qiskit_aer import AerSimulator

# 在本地測試轉譯
sim_backend = AerSimulator.from_backend(backend)
qc_test = transpile(qc, backend=sim_backend, optimization_level=3)
```

## 不同供應商的轉譯

### IBM Quantum

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")
qc_transpiled = transpile(qc, backend=backend)
```

### IonQ

```python
# IonQ 具有全對全連接性，不同的基礎閘
basis_gates = ['gpi', 'gpi2', 'ms']
qc_transpiled = transpile(qc, basis_gates=basis_gates)
```

### Amazon Braket

轉譯取決於特定裝置（Rigetti、IonQ 等）

## 效能技巧

1. **快取轉譯後的電路** - 轉譯很耗時，盡可能重複使用
2. **使用適當的最佳化等級** - 等級 3 較慢但對生產環境最佳
3. **利用 v2.2 速度改進** - 更新到最新 Qiskit 獲得 83 倍加速
4. **平行化轉譯** - 轉譯多個電路時 Qiskit 自動平行化

## 常見問題和解決方案

### 問題：轉譯後電路太深
**解決方案**：使用更高的最佳化等級或重新設計電路減少層數

### 問題：插入太多 SWAP 閘
**解決方案**：調整 initial_layout 以更好地匹配量子位元拓撲

### 問題：轉譯時間太長
**解決方案**：降低最佳化等級或更新到 Qiskit v2.2+ 獲得速度改進

### 問題：意外的閘分解
**解決方案**：檢查 basis_gates 並考慮指定自訂分解規則
