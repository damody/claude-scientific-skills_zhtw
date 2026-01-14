# 硬體整合

本指南涵蓋透過 Cirq 的裝置介面和服務供應商在真實量子硬體上執行量子電路。

## 裝置表示

### 裝置類別

```python
import cirq

# 定義帶連接性的裝置
class MyDevice(cirq.Device):
    def __init__(self, qubits, connectivity):
        self.qubits = qubits
        self.connectivity = connectivity

    @property
    def metadata(self):
        return cirq.DeviceMetadata(
            self.qubits,
            self.connectivity
        )

    def validate_operation(self, operation):
        # 檢查操作是否在此裝置上有效
        if len(operation.qubits) == 2:
            q0, q1 = operation.qubits
            if (q0, q1) not in self.connectivity:
                raise ValueError(f"量子位元 {q0} 和 {q1} 未連接")
```

### 裝置約束

```python
# 檢查裝置元資料
device = cirq_google.Sycamore

# 取得量子位元拓撲
qubits = device.metadata.qubit_set
print(f"可用量子位元：{len(qubits)}")

# 檢查連接性
for q0 in qubits:
    neighbors = device.metadata.nx_graph.neighbors(q0)
    print(f"{q0} 連接到：{list(neighbors)}")

# 根據裝置驗證電路
try:
    device.validate_circuit(circuit)
    print("電路對裝置有效")
except ValueError as e:
    print(f"無效電路：{e}")
```

## 量子位元選擇

### 最佳量子位元選擇

```python
import cirq_google

# 取得校準指標
processor = cirq_google.get_engine().get_processor('weber')
calibration = processor.get_current_calibration()

# 尋找錯誤率最低的量子位元
def select_best_qubits(calibration, n_qubits):
    """選擇單量子位元閘保真度最佳的 n 個量子位元。"""
    qubit_fidelities = {}

    for qubit in calibration.keys():
        if 'single_qubit_rb_average_error_per_gate' in calibration[qubit]:
            error = calibration[qubit]['single_qubit_rb_average_error_per_gate']
            qubit_fidelities[qubit] = 1 - error

    # 依保真度排序
    best_qubits = sorted(
        qubit_fidelities.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n_qubits]

    return [q for q, _ in best_qubits]

best_qubits = select_best_qubits(calibration, n_qubits=10)
```

### 拓撲感知選擇

```python
def select_connected_qubits(device, n_qubits):
    """選擇形成路徑或網格的連接量子位元。"""
    graph = device.metadata.nx_graph

    # 尋找連接子圖
    import networkx as nx
    for node in graph.nodes():
        subgraph = nx.ego_graph(graph, node, radius=n_qubits)
        if len(subgraph) >= n_qubits:
            return list(subgraph.nodes())[:n_qubits]

    raise ValueError(f"找不到 {n_qubits} 個連接的量子位元")
```

## 服務供應商

### Google Quantum AI (Cirq-Google)

#### 設定

```python
import cirq_google

# 認證（需要 Google Cloud 專案）
# 設定環境變數：GOOGLE_CLOUD_PROJECT=your-project-id

# 取得量子引擎
engine = cirq_google.get_engine()

# 列出可用處理器
processors = engine.list_processors()
for processor in processors:
    print(f"處理器：{processor.processor_id}")
```

#### 在 Google 硬體上執行

```python
# 為 Google 裝置建立電路
import cirq_google

# 取得處理器
processor = engine.get_processor('weber')
device = processor.get_device()

# 在裝置量子位元上建立電路
qubits = sorted(device.metadata.qubit_set)[:5]
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CZ(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

# 驗證並執行
device.validate_circuit(circuit)
job = processor.run(circuit, repetitions=1000)

# 取得結果
results = job.results()[0]
print(results.histogram(key='result'))
```

### IonQ

#### 設定

```python
import cirq_ionq

# 設定 API 金鑰
# 選項 1：環境變數
# export IONQ_API_KEY=your_api_key

# 選項 2：在程式碼中
service = cirq_ionq.Service(api_key='your_api_key')
```

#### 在 IonQ 上執行

```python
import cirq_ionq

# 建立服務
service = cirq_ionq.Service(api_key='your_api_key')

# 建立電路（IonQ 使用通用量子位元）
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
    cirq.measure(*qubits, key='result')
)

# 在模擬器上執行
result = service.run(
    circuit=circuit,
    repetitions=1000,
    target='simulator'
)
print(result.histogram(key='result'))

# 在硬體上執行
result = service.run(
    circuit=circuit,
    repetitions=1000,
    target='qpu'
)
```

#### IonQ 作業管理

```python
# 建立作業
job = service.create_job(circuit, repetitions=1000, target='qpu')

# 檢查作業狀態
status = job.status()
print(f"作業狀態：{status}")

# 等待完成
job.wait_until_complete()

# 取得結果
results = job.results()
```

#### IonQ 校準資料

```python
# 取得當前校準
calibration = service.get_current_calibration()

# 存取指標
print(f"保真度：{calibration['fidelity']}")
print(f"時序：{calibration['timing']}")
```

### Azure Quantum

#### 設定

```python
from azure.quantum import Workspace
from azure.quantum.cirq import AzureQuantumService

# 建立工作區連接
workspace = Workspace(
    resource_id="/subscriptions/.../resourceGroups/.../providers/Microsoft.Quantum/Workspaces/...",
    location="eastus"
)

# 建立 Cirq 服務
service = AzureQuantumService(workspace)
```

#### 在 Azure Quantum 上執行（IonQ 後端）

```python
# 列出可用目標
targets = service.targets()
for target in targets:
    print(f"目標：{target.name}")

# 在 IonQ 模擬器上執行
result = service.run(
    circuit=circuit,
    repetitions=1000,
    target='ionq.simulator'
)

# 在 IonQ QPU 上執行
result = service.run(
    circuit=circuit,
    repetitions=1000,
    target='ionq.qpu'
)
```

#### 在 Azure Quantum 上執行（Honeywell 後端）

```python
# 在 Honeywell System Model H1 上執行
result = service.run(
    circuit=circuit,
    repetitions=1000,
    target='honeywell.hqs-lt-s1'
)

# 檢查 Honeywell 特定選項
target_info = service.get_target('honeywell.hqs-lt-s1')
print(f"目標資訊：{target_info}")
```

### AQT (Alpine Quantum Technologies)

#### 設定

```python
import cirq_aqt

# 設定 API 權杖
# export AQT_TOKEN=your_token

# 建立服務
service = cirq_aqt.AQTSampler(
    remote_host='https://gateway.aqt.eu',
    access_token='your_token'
)
```

#### 在 AQT 上執行

```python
# 建立電路
qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

# 在模擬器上執行
result = service.run(
    circuit,
    repetitions=1000,
    target='simulator'
)

# 在裝置上執行
result = service.run(
    circuit,
    repetitions=1000,
    target='device'
)
```

### Pasqal

#### 設定

```python
import cirq_pasqal

# 建立 Pasqal 裝置
device = cirq_pasqal.PasqalDevice(qubits=cirq.LineQubit.range(10))
```

#### 在 Pasqal 上執行

```python
# 建立取樣器
sampler = cirq_pasqal.PasqalSampler(
    remote_host='https://api.pasqal.cloud',
    access_token='your_token',
    device=device
)

# 執行電路
result = sampler.run(circuit, repetitions=1000)
```

## 硬體最佳實務

### 為硬體優化電路

```python
def optimize_for_hardware(circuit, device):
    """為特定硬體優化電路。"""
    from cirq.transformers import (
        optimize_for_target_gateset,
        merge_single_qubit_gates_to_phxz,
        drop_negligible_operations
    )

    # 取得裝置閘集
    if hasattr(device, 'gateset'):
        gateset = device.gateset
    else:
        gateset = cirq.CZTargetGateset()  # 預設

    # 優化
    circuit = merge_single_qubit_gates_to_phxz(circuit)
    circuit = drop_negligible_operations(circuit)
    circuit = optimize_for_target_gateset(circuit, gateset=gateset)

    return circuit
```

### 錯誤緩解

```python
def run_with_readout_error_mitigation(circuit, sampler, repetitions):
    """使用校準緩解讀取錯誤。"""

    # 測量讀取錯誤
    cal_circuits = []
    for state in range(2**len(circuit.qubits)):
        cal_circuit = cirq.Circuit()
        for i, q in enumerate(circuit.qubits):
            if state & (1 << i):
                cal_circuit.append(cirq.X(q))
        cal_circuit.append(cirq.measure(*circuit.qubits, key='m'))
        cal_circuits.append(cal_circuit)

    # 執行校準
    cal_results = [sampler.run(c, repetitions=1000) for c in cal_circuits]

    # 建構混淆矩陣
    # ...（實作細節）

    # 執行實際電路
    result = sampler.run(circuit, repetitions=repetitions)

    # 應用校正
    # ...（應用混淆矩陣的逆）

    return result
```

### 作業管理

```python
def submit_jobs_in_batches(circuits, sampler, batch_size=10):
    """分批提交多個電路。"""
    jobs = []

    for i in range(0, len(circuits), batch_size):
        batch = circuits[i:i+batch_size]
        job_ids = []

        for circuit in batch:
            job = sampler.run_async(circuit, repetitions=1000)
            job_ids.append(job)

        jobs.extend(job_ids)

    # 等待所有作業
    results = [job.result() for job in jobs]
    return results
```

## 裝置規格

### 檢查裝置能力

```python
def print_device_info(device):
    """列印裝置能力和約束。"""

    print(f"裝置：{device}")
    print(f"量子位元數量：{len(device.metadata.qubit_set)}")

    # 閘支援
    print("\n支援的閘：")
    if hasattr(device, 'gateset'):
        for gate in device.gateset.gates:
            print(f"  - {gate}")

    # 連接性
    print("\n連接性：")
    graph = device.metadata.nx_graph
    print(f"  邊數：{graph.number_of_edges()}")
    print(f"  平均度數：{sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")

    # 持續時間約束
    if hasattr(device, 'gate_durations'):
        print("\n閘持續時間：")
        for gate, duration in device.gate_durations.items():
            print(f"  {gate}: {duration}")
```

## 認證和存取

### 設定憑證

**Google Cloud：**
```bash
# 安裝 gcloud CLI
# 訪問：https://cloud.google.com/sdk/docs/install

# 認證
gcloud auth application-default login

# 設定專案
export GOOGLE_CLOUD_PROJECT=your-project-id
```

**IonQ：**
```bash
# 設定 API 金鑰
export IONQ_API_KEY=your_api_key
```

**Azure Quantum：**
```python
# 使用 Azure CLI 或工作區連接字串
# 參見：https://docs.microsoft.com/azure/quantum/
```

**AQT：**
```bash
# 向 AQT 請求存取權杖
export AQT_TOKEN=your_token
```

**Pasqal：**
```bash
# 向 Pasqal 請求 API 存取
export PASQAL_TOKEN=your_token
```

## 最佳實務

1. **提交前驗證電路**：使用 device.validate_circuit()
2. **為目標硬體優化**：分解為原生閘
3. **選擇最佳量子位元**：使用校準資料進行量子位元選擇
4. **監控作業狀態**：擷取結果前檢查作業完成狀態
5. **實施錯誤緩解**：使用讀取錯誤校正
6. **有效率地批次作業**：一起提交多個電路
7. **遵守速率限制**：遵循供應商特定的 API 限制
8. **儲存結果**：立即儲存昂貴的硬體結果
9. **先在模擬器測試**：在硬體執行前在模擬器上驗證
10. **保持電路淺層**：硬體的相干時間有限
