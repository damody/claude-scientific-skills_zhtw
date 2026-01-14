# 硬體後端和執行

Qiskit 是後端無關的，支援在模擬器和來自多個供應商的真實量子硬體上執行。

## 後端類型

### 本地模擬器
- 在您的機器上執行
- 不需要帳戶
- 非常適合開發和測試

### 雲端硬體
- IBM Quantum（100+ 量子位元系統）
- IonQ（離子阱）
- Amazon Braket（Rigetti、IonQ、Oxford Quantum Circuits）
- 其他供應商透過外掛程式支援

## IBM Quantum 後端

### 連接到 IBM Quantum

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# 首次：儲存憑證
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN"
)

# 後續會話：載入憑證
service = QiskitRuntimeService()
```

### 列出可用後端

```python
# 列出所有可用後端
backends = service.backends()
for backend in backends:
    print(f"{backend.name}: {backend.num_qubits} 量子位元")

# 按最小量子位元數篩選
backends_127q = service.backends(min_num_qubits=127)

# 取得特定後端
backend = service.backend("ibm_brisbane")
backend = service.least_busy()  # 取得最空閒的後端
```

### 後端屬性

```python
backend = service.backend("ibm_brisbane")

# 基本資訊
print(f"名稱: {backend.name}")
print(f"量子位元: {backend.num_qubits}")
print(f"版本: {backend.version}")
print(f"狀態: {backend.status()}")

# 耦合圖（量子位元連接性）
print(backend.coupling_map)

# 基礎閘
print(backend.configuration().basis_gates)

# 量子位元屬性
print(backend.qubit_properties(0))  # 量子位元 0 的屬性
```

### 檢查後端狀態

```python
status = backend.status()
print(f"運作中: {status.operational}")
print(f"待處理工作: {status.pending_jobs}")
print(f"狀態訊息: {status.status_msg}")
```

## 在 IBM Quantum 硬體上執行

### 使用 Runtime 基元

```python
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

# 建立並轉譯電路
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 為後端轉譯
transpiled_qc = transpile(qc, backend=backend, optimization_level=3)

# 使用 Sampler 執行
sampler = Sampler(backend)
job = sampler.run([transpiled_qc], shots=1024)

# 取得結果
result = job.result()
counts = result[0].data.meas.get_counts()
print(counts)
```

### 工作管理

```python
# 提交工作
job = sampler.run([qc], shots=1024)

# 取得工作 ID（儲存以便稍後檢索）
job_id = job.job_id()
print(f"工作 ID: {job_id}")

# 檢查工作狀態
print(job.status())

# 等待完成
result = job.result()

# 稍後檢索工作
service = QiskitRuntimeService()
retrieved_job = service.job(job_id)
result = retrieved_job.result()
```

### 工作佇列

```python
# 檢查佇列位置
job_status = job.status()
print(f"佇列位置: {job.queue_position()}")

# 如需要可取消工作
job.cancel()
```

## Session 模式

使用 session 進行迭代演算法（VQE、QAOA）以減少佇列時間：

```python
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

with Session(backend=backend) as session:
    sampler = Sampler(session=session)

    # 同一 session 中的多次迭代
    for iteration in range(10):
        # 參數化電路
        qc = create_parameterized_circuit(params[iteration])
        job = sampler.run([qc], shots=1024)
        result = job.result()

        # 根據結果更新參數
        params[iteration + 1] = optimize(result)
```

Session 的優點：
- 減少迭代之間的佇列等待時間
- 保證 session 期間後端可用性
- 更適合變分演算法

## Batch 模式

使用 batch 模式進行獨立的平行工作：

```python
from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

with Batch(backend=backend) as batch:
    sampler = Sampler(session=batch)

    # 提交多個獨立工作
    jobs = []
    for qc in circuit_list:
        job = sampler.run([qc], shots=1024)
        jobs.append(job)

    # 收集所有結果
    results = [job.result() for job in jobs]
```

## 本地模擬器

### StatevectorSampler（理想模擬）

```python
from qiskit.primitives import StatevectorSampler

sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
counts = result[0].data.meas.get_counts()
```

### Aer 模擬器（真實雜訊）

```python
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# 理想模擬
simulator = AerSimulator()

# 使用後端雜訊模型模擬
backend = service.backend("ibm_brisbane")
noisy_simulator = AerSimulator.from_backend(backend)

# 執行模擬
transpiled_qc = transpile(qc, simulator)
sampler = Sampler(simulator)
job = sampler.run([transpiled_qc], shots=1024)
result = job.result()
```

### Aer GPU 加速

```python
# 使用 GPU 加速模擬
simulator = AerSimulator(method='statevector', device='GPU')
```

## 第三方供應商

### IonQ

IonQ 提供具有全對全連接性的離子阱量子電腦：

```python
from qiskit_ionq import IonQProvider

provider = IonQProvider("YOUR_IONQ_API_TOKEN")

# 列出 IonQ 後端
backends = provider.backends()
backend = provider.get_backend("ionq_qpu")

# 執行電路
job = backend.run(qc, shots=1024)
result = job.result()
```

### Amazon Braket

```python
from qiskit_braket_provider import BraketProvider

provider = BraketProvider()

# 列出可用裝置
backends = provider.backends()

# 使用特定裝置
backend = provider.get_backend("Rigetti")
job = backend.run(qc, shots=1024)
result = job.result()
```

## 錯誤緩解

### 測量錯誤緩解

```python
from qiskit_ibm_runtime import SamplerV2 as Sampler, Options

# 配置錯誤緩解
options = Options()
options.resilience_level = 1  # 0=無, 1=最小, 2=中等, 3=重度

sampler = Sampler(backend, options=options)
job = sampler.run([qc], shots=1024)
result = job.result()
```

### 錯誤緩解等級

- **等級 0**：無緩解
- **等級 1**：讀出錯誤緩解
- **等級 2**：等級 1 + 閘錯誤緩解
- **等級 3**：等級 2 + 進階技術

**Qiskit 的 Samplomatic 套件**可透過機率性錯誤消除將取樣開銷減少高達 100 倍。

### 零雜訊外推（ZNE）

```python
options = Options()
options.resilience_level = 2
options.resilience.zne_mitigation = True

sampler = Sampler(backend, options=options)
```

## 監控使用量和成本

### 檢查帳戶使用量

```python
# 對於 IBM Quantum
service = QiskitRuntimeService()

# 檢查剩餘額度
print(service.usage())
```

### 估算工作成本

```python
from qiskit_ibm_runtime import EstimatorV2 as Estimator

backend = service.backend("ibm_brisbane")

# 估算工作成本
estimator = Estimator(backend)
# 成本取決於電路複雜度和 shots
```

## 最佳實踐

### 1. 始終在執行前轉譯

```python
# 不好：不轉譯就執行
job = sampler.run([qc], shots=1024)

# 好：先轉譯
qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
job = sampler.run([qc_transpiled], shots=1024)
```

### 2. 先用模擬器測試

```python
# 在使用硬體前用雜訊模擬器測試
noisy_sim = AerSimulator.from_backend(backend)
qc_test = transpile(qc, noisy_sim, optimization_level=3)

# 驗證結果看起來合理
# 然後在硬體上執行
```

### 3. 使用適當的 Shot 數量

```python
# 對於最佳化演算法：較少 shots（100-1000）
# 對於最終測量：較多 shots（10000+）

# 根據階段調整 shots
shots_optimization = 500
shots_final = 10000
```

### 4. 策略性地選擇後端

```python
# 測試：使用最空閒的後端
backend = service.least_busy(min_num_qubits=5)

# 生產：使用符合需求的後端
backend = service.backend("ibm_brisbane")  # 127 量子位元
```

### 5. 對變分演算法使用 Sessions

Sessions 非常適合 VQE、QAOA 和其他迭代演算法。

### 6. 監控工作狀態

```python
import time

job = sampler.run([qc], shots=1024)

while job.status().name not in ['DONE', 'ERROR', 'CANCELLED']:
    print(f"狀態: {job.status().name}")
    time.sleep(10)

result = job.result()
```

## 疑難排解

### 問題："Backend not found"
```python
# 列出可用後端
print([b.name for b in service.backends()])
```

### 問題："Invalid credentials"
```python
# 重新儲存憑證
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_TOKEN",
    overwrite=True
)
```

### 問題：長佇列時間
```python
# 使用最空閒的後端
backend = service.least_busy(min_num_qubits=5)

# 或對多個獨立工作使用 batch 模式
```

### 問題：工作因 "Circuit too large" 失敗
```python
# 降低電路複雜度
# 使用更高的轉譯最佳化
qc_opt = transpile(qc, backend=backend, optimization_level=3)
```

## 後端比較

| 供應商 | 連接性 | 閘集 | 備註 |
|--------|--------|------|------|
| IBM Quantum | 有限 | CX, RZ, SX, X | 100+ 量子位元系統，高品質 |
| IonQ | 全對全 | GPI, GPI2, MS | 離子阱，低錯誤率 |
| Rigetti | 有限 | CZ, RZ, RX | 超導量子位元 |
| Oxford Quantum Circuits | 有限 | ECR, RZ, SX | Coaxmon 技術 |
