# Qiskit Patterns：四步驟工作流程

Qiskit Patterns 提供一個通用框架，用於在四個階段解決領域特定的量子運算問題：Map（映射）、Optimize（最佳化）、Execute（執行）和 Post-process（後處理）。

## 概述

Patterns 框架實現量子能力的無縫組合，並支援異構運算基礎設施（CPU/GPU/QPU）。可在本地、透過雲端服務或透過 Qiskit Serverless 執行。

## 四個步驟

```
問題 → [Map] → [Optimize] → [Execute] → [Post-process] → 解決方案
```

### 1. Map（映射）
將古典問題轉換為量子電路和運算子

### 2. Optimize（最佳化）
透過轉譯為目標硬體準備電路

### 3. Execute（執行）
使用基元在量子硬體上執行電路

### 4. Post-process（後處理）
透過古典計算提取和精煉結果

## 步驟 1：Map（映射）

### 目標
將領域特定問題轉換為量子表示（電路、運算子、哈密頓量）。

### 關鍵決策

**選擇輸出類型：**
- **Sampler**：用於位元串輸出（最佳化、搜尋）
- **Estimator**：用於期望值（化學、物理）

**設計電路結構：**
```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np

# 範例：用於 VQE 的參數化電路
def create_ansatz(num_qubits, depth):
    qc = QuantumCircuit(num_qubits)
    params = []

    for d in range(depth):
        # 旋轉層
        for i in range(num_qubits):
            theta = Parameter(f'θ_{d}_{i}')
            params.append(theta)
            qc.ry(theta, i)

        # 糾纏層
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    return qc, params

ansatz, params = create_ansatz(num_qubits=4, depth=2)
```

### 考慮事項

- **硬體拓撲**：設計時考慮後端耦合圖
- **閘效率**：最小化雙量子位元閘
- **測量基**：確定所需的測量

### 領域特定範例

**化學：分子哈密頓量**
```python
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

# 定義分子
driver = PySCFDriver(atom='H 0 0 0; H 0 0 0.735', basis='sto3g')
problem = driver.run()

# 映射到量子位元哈密頓量
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian)
```

**最佳化：QAOA 電路**
```python
from qiskit.circuit import QuantumCircuit, Parameter

def qaoa_circuit(graph, p):
    """為 MaxCut 問題建立 QAOA 電路"""
    num_qubits = len(graph.nodes())
    qc = QuantumCircuit(num_qubits)

    # 初始疊加
    qc.h(range(num_qubits))

    # 交替層
    betas = [Parameter(f'β_{i}') for i in range(p)]
    gammas = [Parameter(f'γ_{i}') for i in range(p)]

    for i in range(p):
        # 問題哈密頓量
        for edge in graph.edges():
            qc.cx(edge[0], edge[1])
            qc.rz(2 * gammas[i], edge[1])
            qc.cx(edge[0], edge[1])

        # 混合哈密頓量
        qc.rx(2 * betas[i], range(num_qubits))

    return qc
```

## 步驟 2：Optimize（最佳化）

### 目標
將抽象電路轉換為硬體相容的 ISA（指令集架構）電路。

### 轉譯

```python
from qiskit import transpile

# 基本轉譯
qc_isa = transpile(qc, backend=backend, optimization_level=3)

# 指定初始佈局
qc_isa = transpile(
    qc,
    backend=backend,
    optimization_level=3,
    initial_layout=[0, 2, 4, 6],  # 映射到特定物理量子位元
    seed_transpiler=42  # 可重現性
)
```

### 預最佳化技巧

1. **先用模擬器測試**：
```python
from qiskit_aer import AerSimulator

sim = AerSimulator.from_backend(backend)
qc_test = transpile(qc, sim, optimization_level=3)
print(f"估計深度: {qc_test.depth()}")
```

2. **分析轉譯結果**：
```python
print(f"原始閘數: {qc.size()}")
print(f"轉譯後閘數: {qc_isa.size()}")
print(f"雙量子位元閘: {qc_isa.count_ops().get('cx', 0)}")
```

3. **考慮電路切割**用於大型電路：
```python
# 對於超出可用硬體的電路
# 使用電路切割技術分割成較小的子電路
```

## 步驟 3：Execute（執行）

### 目標
使用基元在量子硬體上執行 ISA 電路。

### 使用 Sampler

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

# 先轉譯
qc_isa = transpile(qc, backend=backend, optimization_level=3)

# 執行
sampler = Sampler(backend)
job = sampler.run([qc_isa], shots=10000)
result = job.result()
counts = result[0].data.meas.get_counts()
```

### 使用 Estimator

```python
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

# 轉譯
qc_isa = transpile(qc, backend=backend, optimization_level=3)

# 定義可觀測量
observable = SparsePauliOp(["ZZZZ", "XXXX"])

# 執行
estimator = Estimator(backend)
job = estimator.run([(qc_isa, observable)])
result = job.result()
expectation_value = result[0].data.evs
```

### 執行模式

**Session 模式（迭代）：**
```python
from qiskit_ibm_runtime import Session

with Session(backend=backend) as session:
    sampler = Sampler(session=session)

    # 多次迭代
    for iteration in range(max_iterations):
        qc_iteration = update_circuit(params[iteration])
        qc_isa = transpile(qc_iteration, backend=backend)

        job = sampler.run([qc_isa], shots=1000)
        result = job.result()

        # 更新參數
        params[iteration + 1] = optimize_params(result)
```

**Batch 模式（平行）：**
```python
from qiskit_ibm_runtime import Batch

with Batch(backend=backend) as batch:
    sampler = Sampler(session=batch)

    # 一次提交所有工作
    jobs = []
    for qc in circuit_list:
        qc_isa = transpile(qc, backend=backend)
        job = sampler.run([qc_isa], shots=1000)
        jobs.append(job)

    # 收集結果
    results = [job.result() for job in jobs]
```

### 錯誤緩解

```python
from qiskit_ibm_runtime import Options

options = Options()
options.resilience_level = 2  # 0=無, 1=輕度, 2=中等, 3=重度
options.optimization_level = 3

sampler = Sampler(backend, options=options)
```

## 步驟 4：Post-process（後處理）

### 目標
使用古典計算從量子測量中提取有意義的結果。

### 結果處理

**對於 Sampler（位元串）：**
```python
counts = result[0].data.meas.get_counts()

# 轉換為機率
total_shots = sum(counts.values())
probabilities = {state: count/total_shots for state, count in counts.items()}

# 找到最可能的狀態
max_state = max(counts, key=counts.get)
print(f"最可能的狀態: {max_state} ({counts[max_state]}/{total_shots})")
```

**對於 Estimator（期望值）：**
```python
expectation_value = result[0].data.evs
std_dev = result[0].data.stds  # 標準差

print(f"能量: {expectation_value} ± {std_dev}")
```

### 領域特定後處理

**化學：基態能量**
```python
def post_process_chemistry(result, nuclear_repulsion):
    """提取基態能量"""
    electronic_energy = result[0].data.evs
    total_energy = electronic_energy + nuclear_repulsion
    return total_energy
```

**最佳化：MaxCut 解**
```python
def post_process_maxcut(counts, graph):
    """從測量結果中找到最佳切割"""
    def compute_cut_value(bitstring, graph):
        cut_value = 0
        for edge in graph.edges():
            if bitstring[edge[0]] != bitstring[edge[1]]:
                cut_value += 1
        return cut_value

    # 找到具有最大切割的位元串
    best_cut = 0
    best_string = None

    for bitstring, count in counts.items():
        cut = compute_cut_value(bitstring, graph)
        if cut > best_cut:
            best_cut = cut
            best_string = bitstring

    return best_string, best_cut
```

### 進階後處理

**錯誤緩解後處理：**
```python
# 應用額外的古典錯誤緩解
from qiskit.result import marginal_counts

# 邊緣化到相關量子位元
relevant_qubits = [0, 1, 2]
marginal = marginal_counts(counts, indices=relevant_qubits)
```

**統計分析：**
```python
import numpy as np

def analyze_results(results_list):
    """分析多次執行的統計資料"""
    energies = [r[0].data.evs for r in results_list]

    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    confidence_interval = 1.96 * std_energy / np.sqrt(len(energies))

    return {
        'mean': mean_energy,
        'std': std_energy,
        '95% CI': (mean_energy - confidence_interval, mean_energy + confidence_interval)
    }
```

**視覺化：**
```python
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 視覺化結果
plot_histogram(counts, figsize=(12, 6))
plt.title("測量結果")
plt.show()
```

## 完整範例：用於化學的 VQE

```python
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, Session
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import numpy as np

# 1. MAP：建立參數化電路
def create_ansatz(num_qubits):
    qc = QuantumCircuit(num_qubits)
    params = []

    for i in range(num_qubits):
        theta = f'θ_{i}'
        params.append(theta)
        qc.ry(theta, i)

    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc, params

# 定義哈密頓量（範例：H2 分子）
hamiltonian = SparsePauliOp(["IIZZ", "ZZII", "XXII", "IIXX"], coeffs=[0.3, 0.3, 0.1, 0.1])

# 2. OPTIMIZE：連接並準備
service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

ansatz, param_names = create_ansatz(num_qubits=4)

# 3. EXECUTE：執行 VQE
def cost_function(params):
    # 綁定參數
    bound_circuit = ansatz.assign_parameters({param_names[i]: params[i] for i in range(len(params))})

    # 轉譯
    qc_isa = transpile(bound_circuit, backend=backend, optimization_level=3)

    # 執行
    job = estimator.run([(qc_isa, hamiltonian)])
    result = job.result()
    energy = result[0].data.evs

    return energy

with Session(backend=backend) as session:
    estimator = Estimator(session=session)

    # 古典最佳化迴圈
    initial_params = np.random.random(len(param_names)) * 2 * np.pi
    result = minimize(cost_function, initial_params, method='COBYLA')

# 4. POST-PROCESS：提取基態能量
ground_state_energy = result.fun
optimized_params = result.x

print(f"基態能量: {ground_state_energy}")
print(f"最佳化參數: {optimized_params}")
```

## 最佳實踐

### 1. 先在本地迭代
在使用硬體前用模擬器測試完整工作流程：
```python
from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()
# 在本地測試工作流程
```

### 2. 對迭代演算法使用 Sessions
VQE、QAOA 和其他變分演算法從 sessions 中受益。

### 3. 選擇適當的 Shots
- 開發/測試：100-1000 shots
- 生產：10,000+ shots

### 4. 監控收斂
```python
energies = []

def cost_function_with_tracking(params):
    energy = cost_function(params)
    energies.append(energy)
    print(f"迭代 {len(energies)}: E = {energy}")
    return energy
```

### 5. 儲存結果
```python
import json

results_data = {
    'energy': float(ground_state_energy),
    'parameters': optimized_params.tolist(),
    'iterations': len(energies),
    'backend': backend.name
}

with open('vqe_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
```

## Qiskit Serverless

對於大規模工作流程，使用 Qiskit Serverless 進行分散式計算：

```python
from qiskit_serverless import ServerlessClient, QiskitFunction

client = ServerlessClient()

# 定義 serverless 函數
@QiskitFunction()
def run_vqe_serverless(hamiltonian, ansatz):
    # 您的 VQE 實作
    pass

# 遠端執行
job = run_vqe_serverless(hamiltonian, ansatz)
result = job.result()
```

## 常見工作流程模式

### 模式 1：參數掃描
```python
# Map → 最佳化一次 → 執行多次 → 後處理
qc_isa = transpile(parameterized_circuit, backend=backend)

with Batch(backend=backend) as batch:
    sampler = Sampler(session=batch)
    results = []

    for param_set in parameter_sweep:
        bound_qc = qc_isa.assign_parameters(param_set)
        job = sampler.run([bound_qc], shots=1000)
        results.append(job.result())
```

### 模式 2：迭代精煉
```python
# Map → (Optimize → Execute → Post-process) 重複
with Session(backend=backend) as session:
    estimator = Estimator(session=session)

    for iteration in range(max_iter):
        qc = update_circuit(params)
        qc_isa = transpile(qc, backend=backend)

        result = estimator.run([(qc_isa, observable)]).result()
        params = update_params(result)
```

### 模式 3：集成測量
```python
# Map → Optimize → 執行多個可觀測量 → 後處理
qc_isa = transpile(qc, backend=backend)

observables = [obs1, obs2, obs3, obs4]
jobs = [(qc_isa, obs) for obs in observables]

estimator = Estimator(backend)
result = estimator.run(jobs).result()
expectation_values = [r.data.evs for r in result]
```
