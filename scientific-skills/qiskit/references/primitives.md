# Qiskit 基元

基元是執行量子電路的基本建構塊。Qiskit 提供兩種主要基元：**Sampler**（用於測量位元串）和 **Estimator**（用於計算期望值）。

## 基元類型

### Sampler
計算量子電路位元串的機率或準機率。當您需要以下情況時使用：
- 測量結果
- 輸出機率分佈
- 從量子態取樣

### Estimator
計算量子電路可觀測量的期望值。當您需要以下情況時使用：
- 能量計算
- 可觀測量測量
- 變分演算法最佳化

## V2 介面（目前標準）

Qiskit 使用 V2 基元（BaseSamplerV2、BaseEstimatorV2）作為目前標準。V1 基元是舊版。

## Sampler 基元

### StatevectorSampler（本地模擬）

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# 建立電路
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 使用 Sampler 執行
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()

# 存取結果
counts = result[0].data.meas.get_counts()
print(counts)  # 例如 {'00': 523, '11': 501}
```

### 多個電路

```python
qc1 = QuantumCircuit(2)
qc1.h(0)
qc1.measure_all()

qc2 = QuantumCircuit(2)
qc2.x(0)
qc2.measure_all()

# 執行多個電路
sampler = StatevectorSampler()
job = sampler.run([qc1, qc2], shots=1000)
results = job.result()

# 存取個別結果
counts1 = results[0].data.meas.get_counts()
counts2 = results[1].data.meas.get_counts()
```

### 使用參數

```python
from qiskit.circuit import Parameter

theta = Parameter('θ')
qc = QuantumCircuit(1)
qc.ry(theta, 0)
qc.measure_all()

# 使用參數值執行
sampler = StatevectorSampler()
param_values = [[0], [np.pi/4], [np.pi/2]]
result = sampler.run([(qc, param_values)], shots=1024).result()
```

## Estimator 基元

### StatevectorEstimator（本地模擬）

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# 建立電路（不含測量）
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 定義可觀測量
observable = SparsePauliOp(["ZZ", "XX"])

# 執行 Estimator
estimator = StatevectorEstimator()
result = estimator.run([(qc, observable)]).result()

# 存取期望值
exp_value = result[0].data.evs
print(f"期望值: {exp_value}")
```

### 多個可觀測量

```python
from qiskit.quantum_info import SparsePauliOp

qc = QuantumCircuit(2)
qc.h(0)

obs1 = SparsePauliOp(["ZZ"])
obs2 = SparsePauliOp(["XX"])

estimator = StatevectorEstimator()
result = estimator.run([(qc, obs1), (qc, obs2)]).result()

ev1 = result[0].data.evs
ev2 = result[1].data.evs
```

### 參數化 Estimator

```python
from qiskit.circuit import Parameter
import numpy as np

theta = Parameter('θ')
qc = QuantumCircuit(1)
qc.ry(theta, 0)

observable = SparsePauliOp(["Z"])

# 使用多個參數值執行
estimator = StatevectorEstimator()
param_values = [[0], [np.pi/4], [np.pi/2], [np.pi]]
result = estimator.run([(qc, observable, param_values)]).result()
```

## IBM Quantum Runtime 基元

要在真實硬體上執行，請使用 runtime 基元：

### Runtime Sampler

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 在真實硬體上執行
sampler = Sampler(backend)
job = sampler.run([qc], shots=1024)
result = job.result()
counts = result[0].data.meas.get_counts()
```

### Runtime Estimator

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

observable = SparsePauliOp(["ZZ"])

# 在真實硬體上執行
estimator = Estimator(backend)
job = estimator.run([(qc, observable)])
result = job.result()
exp_value = result[0].data.evs
```

## 用於迭代工作負載的 Sessions

Sessions 將多個工作分組以減少佇列等待時間：

```python
from qiskit_ibm_runtime import Session

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

with Session(backend=backend) as session:
    sampler = Sampler(session=session)

    # 在 session 中執行多個工作
    job1 = sampler.run([qc1], shots=1024)
    result1 = job1.result()

    job2 = sampler.run([qc2], shots=1024)
    result2 = job2.result()
```

## 用於平行工作的 Batch 模式

Batch 模式平行執行獨立工作：

```python
from qiskit_ibm_runtime import Batch

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

with Batch(backend=backend) as batch:
    sampler = Sampler(session=batch)

    # 提交多個獨立工作
    job1 = sampler.run([qc1], shots=1024)
    job2 = sampler.run([qc2], shots=1024)

    # 準備就緒時檢索結果
    result1 = job1.result()
    result2 = job2.result()
```

## 結果處理

### Sampler 結果

```python
result = sampler.run([qc], shots=1024).result()

# 取得計數
counts = result[0].data.meas.get_counts()

# 取得機率
probs = {k: v/1024 for k, v in counts.items()}

# 取得元資料
metadata = result[0].metadata
```

### Estimator 結果

```python
result = estimator.run([(qc, observable)]).result()

# 期望值
exp_val = result[0].data.evs

# 標準差（如可用）
std_dev = result[0].data.stds

# 元資料
metadata = result[0].metadata
```

## 與 V1 基元的差異

**V2 改進：**
- 更靈活的參數綁定
- 更好的結果結構
- 改進的效能
- 更簡潔的 API 設計

**從 V1 遷移：**
- 使用 `StatevectorSampler` 代替 `Sampler`
- 使用 `StatevectorEstimator` 代替 `Estimator`
- 結果存取從 `.result().quasi_dists[0]` 變更為 `.result()[0].data.meas.get_counts()`
