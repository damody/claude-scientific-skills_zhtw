# 量子演算法和應用

Qiskit 支援廣泛的量子演算法，用於最佳化、化學、機器學習和物理模擬。

## 目錄

1. [最佳化演算法](#最佳化演算法)
2. [化學與材料科學](#化學與材料科學)
3. [機器學習](#機器學習)
4. [演算法庫](#演算法庫)

## 最佳化演算法

### 變分量子本徵求解器（VQE）

VQE 使用混合量子-古典方法找到哈密頓量的最小本徵值。

**使用情境：**
- 分子基態能量
- 組合最佳化
- 材料模擬

**實作：**
```python
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, Session
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import numpy as np

def vqe_algorithm(hamiltonian, ansatz, backend, initial_params):
    """
    執行 VQE 演算法

    參數：
        hamiltonian: 可觀測量（SparsePauliOp）
        ansatz: 參數化量子電路
        backend: 量子後端
        initial_params: 初始參數值
    """

    with Session(backend=backend) as session:
        estimator = Estimator(session=session)

        def cost_function(params):
            # 將參數綁定到電路
            bound_circuit = ansatz.assign_parameters(params)

            # 為硬體轉譯
            qc_isa = transpile(bound_circuit, backend=backend, optimization_level=3)

            # 計算期望值
            job = estimator.run([(qc_isa, hamiltonian)])
            result = job.result()
            energy = result[0].data.evs

            return energy

        # 古典最佳化
        result = minimize(
            cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': 100}
        )

    return result.fun, result.x

# 範例：H2 分子哈密頓量
hamiltonian = SparsePauliOp(
    ["IIII", "ZZII", "IIZZ", "ZZZI", "IZZI"],
    coeffs=[-0.8, 0.17, 0.17, -0.24, 0.17]
)

# 建立 ansatz
qc = QuantumCircuit(4)
# ... 定義 ansatz 結構 ...

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

energy, params = vqe_algorithm(hamiltonian, qc, backend, np.random.rand(10))
print(f"基態能量: {energy}")
```

### 量子近似最佳化演算法（QAOA）

QAOA 解決組合最佳化問題，如 MaxCut、TSP 和圖著色。

**使用情境：**
- MaxCut 問題
- 投資組合最佳化
- 車輛路徑規劃
- 排程問題

**實作：**
```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import networkx as nx

def qaoa_maxcut(graph, p, backend):
    """
    用於 MaxCut 問題的 QAOA

    參數：
        graph: NetworkX 圖
        p: QAOA 層數
        backend: 量子後端
    """
    num_qubits = len(graph.nodes())
    qc = QuantumCircuit(num_qubits)

    # 初始疊加
    qc.h(range(num_qubits))

    # 交替層
    betas = [Parameter(f'β_{i}') for i in range(p)]
    gammas = [Parameter(f'γ_{i}') for i in range(p)]

    for i in range(p):
        # 問題哈密頓量（MaxCut）
        for edge in graph.edges():
            u, v = edge
            qc.cx(u, v)
            qc.rz(2 * gammas[i], v)
            qc.cx(u, v)

        # 混合哈密頓量
        for qubit in range(num_qubits):
            qc.rx(2 * betas[i], qubit)

    qc.measure_all()
    return qc, betas + gammas

# 範例：4 節點圖上的 MaxCut
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

qaoa_circuit, params = qaoa_maxcut(G, p=2, backend=backend)

# 使用 Sampler 執行並最佳化參數
# ...（類似 VQE 最佳化迴圈）
```

### Grover 演算法

量子搜尋演算法，為非結構化搜尋提供二次加速。

**使用情境：**
- 資料庫搜尋
- SAT 求解
- 尋找標記項目

**實作：**
```python
from qiskit import QuantumCircuit

def grover_oracle(marked_states):
    """建立標記目標狀態的 oracle"""
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    for target in marked_states:
        # 翻轉目標狀態的相位
        for i, bit in enumerate(target):
            if bit == '0':
                qc.x(i)

        # 多控制 Z
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)

        for i, bit in enumerate(target):
            if bit == '0':
                qc.x(i)

    return qc

def grover_diffusion(num_qubits):
    """建立 Grover 擴散運算子"""
    qc = QuantumCircuit(num_qubits)

    qc.h(range(num_qubits))
    qc.x(range(num_qubits))

    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)

    qc.x(range(num_qubits))
    qc.h(range(num_qubits))

    return qc

def grover_algorithm(marked_states, num_iterations):
    """完整的 Grover 演算法"""
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    # 初始化疊加
    qc.h(range(num_qubits))

    # Grover 迭代
    oracle = grover_oracle(marked_states)
    diffusion = grover_diffusion(num_qubits)

    for _ in range(num_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    qc.measure_all()
    return qc

# 在 3 量子位元空間中搜尋狀態 |101⟩
marked = ['101']
iterations = int(np.pi/4 * np.sqrt(2**3))  # 最佳迭代次數
qc_grover = grover_algorithm(marked, iterations)
```

## 化學與材料科學

### 分子基態能量

**安裝 Qiskit Nature：**
```bash
uv pip install qiskit-nature qiskit-nature-pyscf
```

**範例：H2 分子**
```python
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# 定義分子
driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0
)

# 取得電子結構問題
problem = driver.run()

# 將費米子運算子映射到量子位元
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

# 建立初始態
num_particles = problem.num_particles
num_spatial_orbitals = problem.num_spatial_orbitals

init_state = HartreeFock(
    num_spatial_orbitals,
    num_particles,
    mapper
)

# 建立 ansatz
ansatz = UCCSD(
    num_spatial_orbitals,
    num_particles,
    mapper,
    initial_state=init_state
)

# 執行 VQE
energy, params = vqe_algorithm(
    hamiltonian,
    ansatz,
    backend,
    np.zeros(ansatz.num_parameters)
)

# 加上核排斥能
total_energy = energy + problem.nuclear_repulsion_energy
print(f"基態能量: {total_energy} Ha")
```

### 不同的分子映射器

```python
# Jordan-Wigner 映射
jw_mapper = JordanWignerMapper()
ham_jw = jw_mapper.map(problem.hamiltonian.second_q_op())

# Parity 映射（通常更高效）
parity_mapper = ParityMapper()
ham_parity = parity_mapper.map(problem.hamiltonian.second_q_op())

# Bravyi-Kitaev 映射
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
bk_mapper = BravyiKitaevMapper()
ham_bk = bk_mapper.map(problem.hamiltonian.second_q_op())
```

### 激發態

```python
from qiskit_nature.second_q.algorithms import QEOM

# 用於激發態的量子運動方程
qeom = QEOM(estimator, ansatz, 'sd')  # 單激發和雙激發
excited_states = qeom.solve(problem)
```

## 機器學習

### 量子核

量子電腦可以為機器學習計算核函數。

**安裝 Qiskit Machine Learning：**
```bash
uv pip install qiskit-machine-learning
```

**範例：使用量子核進行分類**
```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from sklearn.svm import SVC
import numpy as np

# 建立特徵映射
num_features = 2
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)

# 建立量子核
fidelity = ComputeUncompute(sampler=sampler)
qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# 與 scikit-learn 一起使用
X_train = np.random.rand(50, 2)
y_train = np.random.choice([0, 1], 50)

# 計算核矩陣
kernel_matrix = qkernel.evaluate(X_train)

# 使用量子核訓練 SVM
svc = SVC(kernel='precomputed')
svc.fit(kernel_matrix, y_train)

# 預測
X_test = np.random.rand(10, 2)
kernel_test = qkernel.evaluate(X_test, X_train)
predictions = svc.predict(kernel_test)
```

### 變分量子分類器（VQC）

```python
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import RealAmplitudes

# 建立特徵映射和 ansatz
feature_map = ZZFeatureMap(2)
ansatz = RealAmplitudes(2, reps=1)

# 建立 VQC
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer='COBYLA'
)

# 訓練
vqc.fit(X_train, y_train)

# 預測
predictions = vqc.predict(X_test)
accuracy = vqc.score(X_test, y_test)
```

### 量子神經網路（QNN）

```python
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import QuantumCircuit, Parameter

# 建立參數化電路
qc = QuantumCircuit(2)
params = [Parameter(f'θ_{i}') for i in range(4)]

# 網路結構
for i, param in enumerate(params[:2]):
    qc.ry(param, i)

qc.cx(0, 1)

for i, param in enumerate(params[2:]):
    qc.ry(param, i)

qc.measure_all()

# 建立 QNN
qnn = SamplerQNN(
    circuit=qc,
    sampler=sampler,
    input_params=[],  # 此範例無輸入參數
    weight_params=params
)

# 使用 PyTorch 或 TensorFlow 進行訓練
```

## 演算法庫

### Qiskit Algorithms

量子演算法的標準實作：

```bash
uv pip install qiskit-algorithms
```

**可用演算法：**
- 振幅估計
- 相位估計
- Shor 演算法
- 量子傅立葉變換
- HHL（線性系統）

**範例：量子相位估計**
```python
from qiskit.circuit.library import QFT
from qiskit_algorithms import PhaseEstimation

# 建立酉運算子
num_qubits = 3
unitary = QuantumCircuit(num_qubits)
# ... 定義酉運算子 ...

# 相位估計
pe = PhaseEstimation(num_evaluation_qubits=3, quantum_instance=backend)
result = pe.estimate(unitary=unitary, state_preparation=initial_state)
```

### Qiskit Optimization

最佳化問題求解器：

```bash
uv pip install qiskit-optimization
```

**支援的問題：**
- 二次規劃
- 整數規劃
- 線性規劃
- 約束滿足

**範例：投資組合最佳化**
```python
from qiskit_optimization.applications import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA

# 定義投資組合問題
returns = [0.1, 0.15, 0.12]  # 預期報酬
covariances = [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]
budget = 2  # 要選擇的資產數量

portfolio = PortfolioOptimization(
    expected_returns=returns,
    covariances=covariances,
    budget=budget
)

# 轉換為二次規劃
qp = portfolio.to_quadratic_program()

# 使用 QAOA 求解
qaoa = QAOA(sampler=sampler, optimizer='COBYLA', reps=2)
optimizer = MinimumEigenOptimizer(qaoa)

result = optimizer.solve(qp)
print(f"最佳投資組合: {result.x}")
```

## 物理模擬

### 時間演化

```python
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp

# 定義哈密頓量
hamiltonian = SparsePauliOp(["XX", "YY", "ZZ"], coeffs=[1.0, 1.0, 1.0])

# 時間演化
time = 1.0
evolution_gate = SuzukiTrotter(order=2, reps=10).synthesize(
    hamiltonian,
    time
)

qc = QuantumCircuit(2)
qc.append(evolution_gate, range(2))
```

### 偏微分方程

**使用情境：** 用於求解偏微分方程的量子演算法，具有潛在的指數加速。

```python
# 量子 PDE 求解器實作
# 需要進階技術如 HHL 演算法
# 和解向量的振幅編碼
```

## 基準測試

### Benchpress 工具包

量子演算法基準測試：

```python
# Benchpress 提供標準化基準
# 用於比較量子演算法效能

# 範例：
# - 量子體積電路
# - 隨機電路取樣
# - 應用特定基準
```

## 最佳實踐

### 1. 從簡單開始
從小規模問題實例開始以驗證您的方法：
```python
# 首先用 2-3 個量子位元測試
# 確認正確性後再擴展
```

### 2. 使用模擬器進行開發
```python
from qiskit.primitives import StatevectorSampler

# 使用本地模擬器開發
sampler_sim = StatevectorSampler()

# 生產環境切換到硬體
# sampler_hw = Sampler(backend)
```

### 3. 監控收斂
```python
convergence_data = []

def tracked_cost_function(params):
    energy = cost_function(params)
    convergence_data.append(energy)
    return energy

# 最佳化後繪製收斂圖
import matplotlib.pyplot as plt
plt.plot(convergence_data)
plt.xlabel('迭代次數')
plt.ylabel('能量')
plt.show()
```

### 4. 參數初始化
```python
# 盡可能使用問題特定的初始化
# 隨機初始化
initial_params = np.random.uniform(0, 2*np.pi, num_params)

# 或使用古典預處理
# initial_params = classical_solution_to_params(classical_result)
```

### 5. 儲存中間結果
```python
import json

checkpoint = {
    'iteration': iteration,
    'params': params.tolist(),
    'energy': energy,
    'timestamp': time.time()
}

with open(f'checkpoint_{iteration}.json', 'w') as f:
    json.dump(checkpoint, f)
```

## 資源和延伸閱讀

**官方文件：**
- [Qiskit 教科書](https://qiskit.org/learn)
- [Qiskit Nature 文件](https://qiskit.org/ecosystem/nature)
- [Qiskit Machine Learning 文件](https://qiskit.org/ecosystem/machine-learning)
- [Qiskit Optimization 文件](https://qiskit.org/ecosystem/optimization)

**研究論文：**
- VQE：Peruzzo 等人，Nature Communications (2014)
- QAOA：Farhi 等人，arXiv:1411.4028
- 量子核：Havlíček 等人，Nature (2019)
