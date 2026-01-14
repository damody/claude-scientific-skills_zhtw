# 執行量子實驗

本指南涵蓋設計和執行量子實驗，包括參數掃描、資料收集和使用 ReCirq 框架。

## 實驗設計

### 基本實驗結構

```python
import cirq
import numpy as np
import pandas as pd

class QuantumExperiment:
    """量子實驗的基礎類別。"""

    def __init__(self, qubits, simulator=None):
        self.qubits = qubits
        self.simulator = simulator or cirq.Simulator()
        self.results = []

    def build_circuit(self, **params):
        """使用給定參數建構電路。"""
        raise NotImplementedError

    def run(self, params_list, repetitions=1000):
        """使用參數掃描執行實驗。"""
        for params in params_list:
            circuit = self.build_circuit(**params)
            result = self.simulator.run(circuit, repetitions=repetitions)
            self.results.append({
                'params': params,
                'result': result
            })
        return self.results

    def analyze(self):
        """分析實驗結果。"""
        raise NotImplementedError
```

### 參數掃描

```python
import sympy

# 定義參數
theta = sympy.Symbol('theta')
phi = sympy.Symbol('phi')

# 建立參數化電路
def parameterized_circuit(qubits, theta, phi):
    return cirq.Circuit(
        cirq.ry(theta)(qubits[0]),
        cirq.rz(phi)(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key='result')
    )

# 定義掃描
sweep = cirq.Product(
    cirq.Linspace('theta', 0, np.pi, 20),
    cirq.Linspace('phi', 0, 2*np.pi, 20)
)

# 執行掃描
circuit = parameterized_circuit(cirq.LineQubit.range(2), theta, phi)
results = cirq.Simulator().run_sweep(circuit, params=sweep, repetitions=1000)
```

### 資料收集

```python
def collect_experiment_data(circuit, sweep, simulator, repetitions=1000):
    """收集並組織實驗資料。"""

    data = []
    results = simulator.run_sweep(circuit, params=sweep, repetitions=repetitions)

    for params, result in zip(sweep, results):
        # 擷取參數
        param_dict = {k: v for k, v in params.param_dict.items()}

        # 擷取測量
        counts = result.histogram(key='result')

        # 以結構化格式儲存
        data.append({
            **param_dict,
            'counts': counts,
            'total': repetitions
        })

    return pd.DataFrame(data)

# 收集資料
df = collect_experiment_data(circuit, sweep, cirq.Simulator())

# 儲存到檔案
df.to_csv('experiment_results.csv', index=False)
```

## ReCirq 框架

ReCirq 提供可重現量子實驗的結構化框架。

### ReCirq 實驗結構

```python
"""
標準 ReCirq 實驗結構：

experiment_name/
├── __init__.py
├── experiment.py        # 主要實驗程式碼
├── tasks.py            # 資料生成任務
├── data_collection.py  # 平行資料收集
├── analysis.py         # 資料分析
└── plots.py           # 視覺化
"""
```

### 基於任務的資料收集

```python
from dataclasses import dataclass
from typing import List
import cirq

@dataclass
class ExperimentTask:
    """參數掃描中的單一任務。"""
    theta: float
    phi: float
    repetitions: int = 1000

    def build_circuit(self, qubits):
        """為此任務建構電路。"""
        return cirq.Circuit(
            cirq.ry(self.theta)(qubits[0]),
            cirq.rz(self.phi)(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key='result')
        )

    def run(self, qubits, simulator):
        """執行任務。"""
        circuit = self.build_circuit(qubits)
        result = simulator.run(circuit, repetitions=self.repetitions)
        return {
            'theta': self.theta,
            'phi': self.phi,
            'result': result
        }

# 建立任務
tasks = [
    ExperimentTask(theta=t, phi=p)
    for t in np.linspace(0, np.pi, 10)
    for p in np.linspace(0, 2*np.pi, 10)
]

# 執行任務
qubits = cirq.LineQubit.range(2)
simulator = cirq.Simulator()
results = [task.run(qubits, simulator) for task in tasks]
```

### 平行資料收集

```python
from multiprocessing import Pool
import functools

def run_task_parallel(task, qubits, simulator):
    """執行單一任務（用於平行執行）。"""
    return task.run(qubits, simulator)

def collect_data_parallel(tasks, qubits, simulator, n_workers=4):
    """使用平行處理收集資料。"""

    # 建立帶固定引數的部分函數
    run_func = functools.partial(
        run_task_parallel,
        qubits=qubits,
        simulator=simulator
    )

    # 平行執行
    with Pool(n_workers) as pool:
        results = pool.map(run_func, tasks)

    return results

# 使用平行收集
results = collect_data_parallel(tasks, qubits, cirq.Simulator(), n_workers=8)
```

## 常見量子演算法

### 變分量子本徵求解器（VQE）

```python
import scipy.optimize

def vqe_experiment(hamiltonian, ansatz_func, initial_params):
    """執行 VQE 以尋找基態能量。"""

    def cost_function(params):
        """能量期望值。"""
        circuit = ansatz_func(params)

        # 測量哈密頓量的期望值
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        energy = hamiltonian.expectation_from_state_vector(
            result.final_state_vector,
            qubit_map={q: i for i, q in enumerate(circuit.all_qubits())}
        )
        return energy.real

    # 優化參數
    result = scipy.optimize.minimize(
        cost_function,
        initial_params,
        method='COBYLA'
    )

    return result

# 範例：H2 分子
def h2_ansatz(params, qubits):
    """H2 的 UCC ansatz。"""
    theta = params[0]
    return cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.ry(theta)(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1])
    )

# 定義哈密頓量（簡化版）
qubits = cirq.LineQubit.range(2)
hamiltonian = cirq.PauliSum.from_pauli_strings([
    cirq.PauliString({qubits[0]: cirq.Z}),
    cirq.PauliString({qubits[1]: cirq.Z}),
    cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.Z})
])

# 執行 VQE
result = vqe_experiment(
    hamiltonian,
    lambda p: h2_ansatz(p, qubits),
    initial_params=[0.0]
)

print(f"基態能量：{result.fun}")
print(f"最優參數：{result.x}")
```

### 量子近似優化演算法（QAOA）

```python
def qaoa_circuit(graph, params, p_layers):
    """MaxCut 問題的 QAOA 電路。"""

    qubits = cirq.LineQubit.range(graph.number_of_nodes())
    circuit = cirq.Circuit()

    # 初始疊加
    circuit.append(cirq.H(q) for q in qubits)

    # QAOA 層
    for layer in range(p_layers):
        gamma = params[layer]
        beta = params[p_layers + layer]

        # 問題哈密頓量（成本）
        for edge in graph.edges():
            i, j = edge
            circuit.append(cirq.ZZPowGate(exponent=gamma)(qubits[i], qubits[j]))

        # 混合哈密頓量
        circuit.append(cirq.rx(2 * beta)(q) for q in qubits)

    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

# 執行 QAOA
import networkx as nx

graph = nx.cycle_graph(4)
p_layers = 2

def qaoa_cost(params):
    """評估 QAOA 成本函數。"""
    circuit = qaoa_circuit(graph, params, p_layers)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)

    # 計算 MaxCut 目標
    total_cost = 0
    counts = result.histogram(key='result')

    for bitstring, count in counts.items():
        cost = 0
        bits = [(bitstring >> i) & 1 for i in range(graph.number_of_nodes())]
        for edge in graph.edges():
            i, j = edge
            if bits[i] != bits[j]:
                cost += 1
        total_cost += cost * count

    return -total_cost / 1000  # 最大化切割

# 優化
initial_params = np.random.random(2 * p_layers) * np.pi
result = scipy.optimize.minimize(qaoa_cost, initial_params, method='COBYLA')

print(f"最優成本：{-result.fun}")
print(f"最優參數：{result.x}")
```

### 量子相位估計

```python
def qpe_circuit(unitary, eigenstate_prep, n_counting_qubits):
    """量子相位估計電路。"""

    counting_qubits = cirq.LineQubit.range(n_counting_qubits)
    target_qubit = cirq.LineQubit(n_counting_qubits)

    circuit = cirq.Circuit()

    # 製備本徵態
    circuit.append(eigenstate_prep(target_qubit))

    # 對計數量子位元應用 Hadamard
    circuit.append(cirq.H(q) for q in counting_qubits)

    # 受控么正
    for i, q in enumerate(counting_qubits):
        power = 2 ** (n_counting_qubits - 1 - i)
        # 應用受控 U^power
        for _ in range(power):
            circuit.append(cirq.ControlledGate(unitary)(q, target_qubit))

    # 對計數量子位元進行逆 QFT
    circuit.append(inverse_qft(counting_qubits))

    # 測量計數量子位元
    circuit.append(cirq.measure(*counting_qubits, key='phase'))

    return circuit

def inverse_qft(qubits):
    """逆量子傅立葉轉換。"""
    n = len(qubits)
    ops = []

    for i in range(n // 2):
        ops.append(cirq.SWAP(qubits[i], qubits[n - i - 1]))

    for i in range(n):
        for j in range(i):
            ops.append(cirq.CZPowGate(exponent=-1/2**(i-j))(qubits[j], qubits[i]))
        ops.append(cirq.H(qubits[i]))

    return ops
```

## 資料分析

### 統計分析

```python
def analyze_measurement_statistics(results):
    """分析測量統計資料。"""

    counts = results.histogram(key='result')
    total = sum(counts.values())

    # 計算機率
    probabilities = {state: count/total for state, count in counts.items()}

    # Shannon 熵
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)

    # 最可能的結果
    most_likely = max(counts.items(), key=lambda x: x[1])

    return {
        'probabilities': probabilities,
        'entropy': entropy,
        'most_likely_state': most_likely[0],
        'most_likely_probability': most_likely[1] / total
    }
```

### 期望值計算

```python
def calculate_expectation_value(circuit, observable, simulator):
    """計算可觀測量的期望值。"""

    # 移除測量
    circuit_no_measure = cirq.Circuit(
        m for m in circuit if not isinstance(m, cirq.MeasurementGate)
    )

    result = simulator.simulate(circuit_no_measure)
    state_vector = result.final_state_vector

    # 計算 ⟨ψ|O|ψ⟩
    expectation = observable.expectation_from_state_vector(
        state_vector,
        qubit_map={q: i for i, q in enumerate(circuit.all_qubits())}
    )

    return expectation.real
```

### 保真度估計

```python
def state_fidelity(state1, state2):
    """計算兩個狀態之間的保真度。"""
    return np.abs(np.vdot(state1, state2)) ** 2

def process_fidelity(result1, result2):
    """從測量結果計算過程保真度。"""

    counts1 = result1.histogram(key='result')
    counts2 = result2.histogram(key='result')

    # 正規化為機率
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    probs1 = {k: v/total1 for k, v in counts1.items()}
    probs2 = {k: v/total2 for k, v in counts2.items()}

    # 經典保真度（Bhattacharyya 係數）
    all_states = set(probs1.keys()) | set(probs2.keys())
    fidelity = sum(np.sqrt(probs1.get(s, 0) * probs2.get(s, 0))
                   for s in all_states) ** 2

    return fidelity
```

## 視覺化

### 繪製參數地形

```python
import matplotlib.pyplot as plt

def plot_parameter_landscape(theta_vals, phi_vals, energies):
    """繪製 2D 參數地形。"""

    plt.figure(figsize=(10, 8))
    plt.contourf(theta_vals, phi_vals, energies, levels=50, cmap='viridis')
    plt.colorbar(label='能量')
    plt.xlabel('θ')
    plt.ylabel('φ')
    plt.title('能量地形')
    plt.show()
```

### 繪製收斂圖

```python
def plot_optimization_convergence(optimization_history):
    """繪製優化收斂圖。"""

    iterations = range(len(optimization_history))
    energies = [result['energy'] for result in optimization_history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, 'b-', linewidth=2)
    plt.xlabel('迭代次數')
    plt.ylabel('能量')
    plt.title('優化收斂')
    plt.grid(True)
    plt.show()
```

### 繪製測量分佈

```python
def plot_measurement_distribution(results):
    """繪製測量結果分佈。"""

    counts = results.histogram(key='result')

    plt.figure(figsize=(12, 6))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('測量結果')
    plt.ylabel('計數')
    plt.title('測量分佈')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## 最佳實務

1. **清楚地結構化實驗**：使用 ReCirq 模式以確保可重現性
2. **分離任務**：將資料生成、收集和分析分開
3. **使用參數掃描**：系統性地探索參數空間
4. **儲存中間結果**：不要遺失昂貴的計算
5. **盡可能平行化**：對獨立任務使用多處理
6. **追蹤元資料**：記錄實驗條件、時間戳記、版本
7. **在模擬器上驗證**：在硬體執行前測試實驗程式碼
8. **實施錯誤處理**：為長時間運行的實驗編寫健壯的程式碼
9. **版本控制資料**：與程式碼一起追蹤實驗資料
10. **徹底記錄**：清晰的文件以確保可重現性

## 範例：完整實驗

```python
# 完整實驗工作流程
class VQEExperiment(QuantumExperiment):
    """完整的 VQE 實驗。"""

    def __init__(self, hamiltonian, ansatz, qubits):
        super().__init__(qubits)
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.history = []

    def build_circuit(self, params):
        return self.ansatz(params, self.qubits)

    def cost_function(self, params):
        circuit = self.build_circuit(params)
        result = self.simulator.simulate(circuit)
        energy = self.hamiltonian.expectation_from_state_vector(
            result.final_state_vector,
            qubit_map={q: i for i, q in enumerate(self.qubits)}
        )
        self.history.append({'params': params, 'energy': energy.real})
        return energy.real

    def run(self, initial_params):
        result = scipy.optimize.minimize(
            self.cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': 100}
        )
        return result

    def analyze(self):
        # 繪製收斂圖
        energies = [h['energy'] for h in self.history]
        plt.plot(energies)
        plt.xlabel('迭代次數')
        plt.ylabel('能量')
        plt.title('VQE 收斂')
        plt.show()

        return {
            'final_energy': self.history[-1]['energy'],
            'optimal_params': self.history[-1]['params'],
            'num_iterations': len(self.history)
        }

# 執行實驗
experiment = VQEExperiment(hamiltonian, h2_ansatz, qubits)
result = experiment.run(initial_params=[0.0])
analysis = experiment.analyze()
```
